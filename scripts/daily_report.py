"""
每日选股报告

每天自动运行模型，生成选股报告并推送到飞书群。

用法：
    python daily_report.py                # 运行报告
    python daily_report.py --dry-run      # 测试运行（不推送）
"""
import sys
import os
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 飞书配置
FEISHU_APP_ID = os.getenv("FEISHU_APP_ID", "")
FEISHU_APP_SECRET = os.getenv("FEISHU_APP_SECRET", "")
FEISHU_CHAT_ID = os.getenv("FEISHU_CHAT_ID", "oc_5fe5ead630574921013c411e54270fa2")


def load_and_prepare_data() -> Optional[pd.DataFrame]:
    """加载并准备数据"""
    try:
        # 加载数据
        data_path = "memory/dataset/features_h10_full.parquet"
        if not os.path.exists(data_path):
            logger.error(f"数据文件不存在: {data_path}")
            return None
        
        df = pd.read_parquet(data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # 特征工程
        feature_cols = [c for c in df.columns if c not in ['date','exit_date','ts_code',
            'stock_return','benchmark_return','excess_return','industry']]
        for col in feature_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(0)
        
        # 添加交互特征
        base_features = ['mom_20', 'volatility_20', 'roe', 'dv_ratio', 'net_mf_ratio']
        for i in range(len(base_features)):
            for j in range(i+1, len(base_features)):
                f1, f2 = base_features[i], base_features[j]
                df[f'{f1}_x_{f2}'] = df[f1] * df[f2]
        
        df['price_momentum'] = df['mom_5'] * df['mom_20']
        df['vol_momentum'] = df['volatility_20'] * df['mom_20']
        df['value_momentum'] = df['dv_ratio'] * df['mom_20']
        df['quality_momentum'] = df['roe'] * df['mom_20']
        df['flow_momentum'] = df['net_mf_ratio'] * df['mom_20']
        df['mom_accel'] = df['mom_5'] - df['mom_20']
        df['mom_decel'] = df['mom_20'] - df['mom_60']
        df['reversal_strength'] = df['reversal_score'] * df['volatility_20']
        df['quality_value'] = df['roe'] * df['dv_ratio']
        
        # 截面排名
        key = ['mom_20','volatility_20','roe','dv_ratio','net_mf_ratio','bp','ep','mom_5','mom_60']
        for c in key:
            if c in df.columns:
                df[f'{c}_r'] = df.groupby('date')[c].rank(pct=True)
        
        # 更新特征列表
        feature_cols = [c for c in df.columns if c not in ['date','exit_date','ts_code',
            'stock_return','benchmark_return','excess_return','industry']]
        for col in feature_cols:
            df[col] = df[col].fillna(0)
        feature_cols = [c for c in feature_cols if df[c].std() > 0]
        
        return df, feature_cols
        
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        return None


def train_model(df: pd.DataFrame, feature_cols: List[str]) -> Optional[object]:
    """训练模型"""
    try:
        import lightgbm as lgb
        from sklearn.preprocessing import StandardScaler
        from scipy.stats import spearmanr
        
        # 使用最近12个月数据训练
        df['ym'] = df['date'].dt.to_period('M')
        months = sorted(df['ym'].unique())
        
        if len(months) < 12:
            logger.error("数据不足12个月")
            return None
        
        # 使用最近12个月训练
        train_months = months[-12:]
        train_df = df[df['ym'].isin(train_months)]
        
        # 特征选择
        def calc_ic(pred, actual):
            if len(pred) < 10:
                return 0
            ic, _ = spearmanr(pred, actual)
            return ic if not np.isnan(ic) else 0
        
        ics = {}
        for c in feature_cols:
            ic = calc_ic(train_df[c].values, train_df['excess_return'].values)
            if not np.isnan(ic):
                ics[c] = abs(ic)
        selected = sorted(ics, key=ics.get, reverse=True)[:30]
        
        # 训练 DoubleEnsemble
        sc = StandardScaler()
        X = sc.fit_transform(train_df[selected].values)
        y = train_df['excess_return'].values
        
        # 第一阶段
        model1 = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=4,
            min_child_samples=50,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=500,
            verbose=-1
        )
        model1.fit(X, y)
        
        # 第二阶段
        residuals = y - model1.predict(X)
        model2 = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=4,
            min_child_samples=50,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=501,
            verbose=-1
        )
        model2.fit(X, residuals)
        
        return {
            'model1': model1,
            'model2': model2,
            'scaler': sc,
            'selected_features': selected
        }
        
    except Exception as e:
        logger.error(f"训练模型失败: {e}")
        return None


def generate_predictions(df: pd.DataFrame, feature_cols: List[str], model: object) -> Optional[pd.DataFrame]:
    """生成预测"""
    try:
        # 获取最新日期
        latest_date = df['date'].max()
        logger.info(f"最新数据日期: {latest_date}")
        
        # 获取最新数据
        latest_df = df[df['date'] == latest_date].copy()
        
        if latest_df.empty:
            logger.error("没有最新数据")
            return None
        
        # 质量因子召回
        recalled = latest_df[(latest_df['roe'] > 0) & (latest_df['net_mf_ratio'] > 0)]
        
        if recalled.empty:
            logger.error("召回后没有股票")
            return None
        
        # 预测
        selected = model['selected_features']
        X = model['scaler'].transform(recalled[selected].values)
        
        pred1 = model['model1'].predict(X)
        pred2 = model['model2'].predict(X)
        predictions = pred1 + pred2
        
        # 构建结果
        result = recalled[['ts_code', 'industry', 'roe', 'dv_ratio', 'mom_20', 'volatility_20']].copy()
        result['predicted'] = predictions
        result['date'] = latest_date
        
        # 计算置信度（预测分数的百分位）
        result['confidence'] = result['predicted'].rank(pct=True)
        
        # 按预测分数排序
        result = result.sort_values('predicted', ascending=False)
        
        return result
        
    except Exception as e:
        logger.error(f"生成预测失败: {e}")
        return None


def format_report(predictions: pd.DataFrame, date: str) -> str:
    """格式化报告"""
    try:
        # Top 10 推荐股票
        top10 = predictions.head(10)
        
        content = f"📊 AlphaHelix 每日选股报告\n"
        content += f"📅 日期: {date}\n"
        content += f"🤖 模型: DoubleEnsemble\n"
        content += f"📈 持仓: Top10\n\n"
        
        content += "🎯 推荐持有:\n"
        for idx, (_, row) in enumerate(top10.iterrows(), 1):
            confidence = row['confidence'] * 100
            content += f"{idx}. {row['ts_code']}\n"
            content += f"   行业: {row['industry']}\n"
            content += f"   置信度: {confidence:.1f}%\n"
            content += f"   ROE: {row['roe']:.2%}\n"
            content += f"   股息率: {row['dv_ratio']:.2%}\n"
            content += f"   动量: {row['mom_20']:.2%}\n\n"
        
        # 关注股票（Top 11-20）
        if len(predictions) > 10:
            watch_stocks = predictions.iloc[10:20]
            content += "👀 建议关注:\n"
            for idx, (_, row) in enumerate(watch_stocks.iterrows(), 11):
                confidence = row['confidence'] * 100
                content += f"{idx}. {row['ts_code']} ({row['industry']}) - 置信度: {confidence:.1f}%\n"
        
        content += f"\n⏰ 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return content
        
    except Exception as e:
        logger.error(f"格式化报告失败: {e}")
        return f"报告生成失败: {e}"


def send_to_feishu(content: str, chat_id: str = None) -> bool:
    """发送到飞书"""
    try:
        from feishu_bot import send_message
        
        chat_id = chat_id or FEISHU_CHAT_ID
        
        if not chat_id:
            logger.error("未配置飞书 chat_id")
            return False
        
        result = send_message(chat_id, content)
        
        if result.get('status') == 'ok':
            logger.info(f"飞书推送成功: {result.get('message_id')}")
            return True
        else:
            logger.error(f"飞书推送失败: {result}")
            return False
            
    except Exception as e:
        logger.error(f"飞书推送异常: {e}")
        return False


def run_daily_report(dry_run: bool = False):
    """运行每日报告"""
    logger.info("=" * 50)
    logger.info("开始生成每日选股报告")
    logger.info("=" * 50)
    
    # 1. 加载数据
    logger.info("步骤 1: 加载数据...")
    result = load_and_prepare_data()
    if result is None:
        logger.error("加载数据失败")
        return False
    
    df, feature_cols = result
    logger.info(f"数据加载成功: {len(df)} 行, {len(feature_cols)} 特征")
    
    # 2. 训练模型
    logger.info("步骤 2: 训练模型...")
    model = train_model(df, feature_cols)
    if model is None:
        logger.error("训练模型失败")
        return False
    
    logger.info(f"模型训练成功: {len(model['selected_features'])} 特征")
    
    # 3. 生成预测
    logger.info("步骤 3: 生成预测...")
    predictions = generate_predictions(df, feature_cols, model)
    if predictions is None:
        logger.error("生成预测失败")
        return False
    
    logger.info(f"预测生成成功: {len(predictions)} 只股票")
    
    # 4. 格式化报告
    logger.info("步骤 4: 格式化报告...")
    date = datetime.now().strftime('%Y-%m-%d')
    report = format_report(predictions, date)
    
    # 5. 推送到飞书
    if dry_run:
        logger.info("测试模式，不推送飞书")
        print("\n" + "=" * 50)
        print("报告预览:")
        print("=" * 50)
        print(report)
        print("=" * 50)
    else:
        logger.info("步骤 5: 推送到飞书...")
        if send_to_feishu(report):
            logger.info("每日报告完成!")
        else:
            logger.error("飞书推送失败")
            return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="每日选股报告")
    parser.add_argument("--dry-run", action="store_true", help="测试运行（不推送飞书）")
    args = parser.parse_args()
    
    success = run_daily_report(dry_run=args.dry_run)
    
    if success:
        logger.info("每日报告完成!")
        sys.exit(0)
    else:
        logger.error("每日报告失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()
