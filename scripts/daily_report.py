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


def get_stock_names() -> Dict[str, str]:
    """获取股票名称映射"""
    try:
        import tushare as ts
        token = os.getenv('TUSHARE_TOKEN', '')
        pro = ts.pro_api(token)
        
        # 获取股票列表
        df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')
        
        # 构建映射
        name_map = dict(zip(df['ts_code'], df['name']))
        logger.info(f"获取股票名称: {len(name_map)} 只")
        
        return name_map
        
    except Exception as e:
        logger.warning(f"获取股票名称失败: {e}")
        return {}


def get_stock_price_info(ts_codes: List[str]) -> Dict[str, Dict]:
    """获取股票价格信息（当前价、7日涨跌、30日涨跌）"""
    try:
        import tushare as ts
        token = os.getenv('TUSHARE_TOKEN', '')
        pro = ts.pro_api(token)
        
        # 计算日期范围
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=40)).strftime('%Y%m%d')
        
        # 批量获取日线数据
        price_info = {}
        for ts_code in ts_codes:
            try:
                df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                if df is not None and len(df) >= 2:
                    df = df.sort_values('trade_date')
                    latest_close = df.iloc[-1]['close']
                    
                    # 7日涨跌
                    ret_7d = None
                    if len(df) >= 7:
                        close_7d = df.iloc[-7]['close']
                        ret_7d = (latest_close - close_7d) / close_7d
                    
                    # 30日涨跌
                    ret_30d = None
                    if len(df) >= 30:
                        close_30d = df.iloc[-30]['close']
                        ret_30d = (latest_close - close_30d) / close_30d
                    elif len(df) >= 2:
                        close_oldest = df.iloc[0]['close']
                        ret_30d = (latest_close - close_oldest) / close_oldest
                    
                    price_info[ts_code] = {
                        'price': latest_close,
                        'ret_7d': ret_7d,
                        'ret_30d': ret_30d,
                    }
            except Exception as e:
                continue
        
        logger.info(f"获取价格信息: {len(price_info)} 只")
        return price_info
        
    except Exception as e:
        logger.warning(f"获取价格信息失败: {e}")
        return {}


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
        
        # 使用最近1个月数据训练（短窗口避免特征方向漂移）
        df['ym'] = df['date'].dt.to_period('M')
        months = sorted(df['ym'].unique())
        
        if len(months) < 1:
            logger.error("数据不足")
            return None
        
        # 使用最近1个月训练
        train_months = months[-1:]
        train_df = df[df['ym'].isin(train_months)]
        
        # 特征选择（只选正IC特征）
        def calc_ic(pred, actual):
            if len(pred) < 10:
                return 0
            ic, _ = spearmanr(pred, actual)
            return ic if not np.isnan(ic) else 0
        
        ics = {}
        for c in feature_cols:
            ic = calc_ic(train_df[c].values, train_df['excess_return'].values)
            if not np.isnan(ic) and ic > 0:  # 只选正IC
                ics[c] = ic
        selected = sorted(ics, key=ics.get, reverse=True)[:15]  # 15个特征
        
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


def format_report(predictions: pd.DataFrame, date: str, name_map: Dict[str, str], price_info: Dict[str, Dict]) -> str:
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
            ts_code = row['ts_code']
            stock_name = name_map.get(ts_code, ts_code)
            confidence = row['confidence'] * 100
            info = price_info.get(ts_code, {})
            
            content += f"{idx}. {stock_name} ({ts_code})\n"
            content += f"   行业: {row['industry']}\n"
            content += f"   置信度: {confidence:.1f}%\n"
            
            # 价格信息
            price = info.get('price')
            if price:
                content += f"   现价: {price:.2f}\n"
            
            ret_7d = info.get('ret_7d')
            if ret_7d is not None:
                emoji = "📈" if ret_7d > 0 else "📉"
                content += f"   7日涨跌: {emoji} {ret_7d:+.2%}\n"
            
            ret_30d = info.get('ret_30d')
            if ret_30d is not None:
                emoji = "📈" if ret_30d > 0 else "📉"
                content += f"   30日涨跌: {emoji} {ret_30d:+.2%}\n"
            
            content += f"   ROE: {row['roe']:.2%}\n"
            content += f"   股息率: {row['dv_ratio']:.2%}\n\n"
        
        # 关注股票（Top 11-20）
        if len(predictions) > 10:
            watch_stocks = predictions.iloc[10:20]
            content += "👀 建议关注:\n"
            for idx, (_, row) in enumerate(watch_stocks.iterrows(), 11):
                ts_code = row['ts_code']
                stock_name = name_map.get(ts_code, ts_code)
                confidence = row['confidence'] * 100
                info = price_info.get(ts_code, {})
                
                ret_30d = info.get('ret_30d')
                ret_str = f" 30日:{ret_30d:+.1%}" if ret_30d else ""
                
                content += f"{idx}. {stock_name} ({ts_code}) - 置信度: {confidence:.1f}%{ret_str}\n"
        
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


def save_predictions(predictions: pd.DataFrame, date: str):
    """保存预测结果"""
    try:
        os.makedirs('memory/predictions', exist_ok=True)
        path = f'memory/predictions/predictions_{date}.parquet'
        predictions.to_parquet(path, index=False)
        logger.info(f"预测结果已保存: {path}")
    except Exception as e:
        logger.warning(f"保存预测结果失败: {e}")


def load_predictions(date: str) -> Optional[pd.DataFrame]:
    """加载预测结果"""
    try:
        path = f'memory/predictions/predictions_{date}.parquet'
        if os.path.exists(path):
            return pd.read_parquet(path)
        return None
    except Exception as e:
        logger.warning(f"加载预测结果失败: {e}")
        return None


def verify_yesterday_predictions(name_map: Dict[str, str]) -> Optional[str]:
    """验证前一天的预测"""
    try:
        # 获取昨天日期
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # 加载昨天的预测
        predictions = load_predictions(yesterday)
        if predictions is None:
            logger.info(f"未找到昨天的预测结果: {yesterday}")
            return None
        
        logger.info(f"验证昨天预测: {yesterday}, {len(predictions)} 只股票")
        
        # 获取今天的价格信息
        ts_codes = predictions['ts_code'].tolist()
        price_info = get_stock_price_info(ts_codes)
        
        # 计算每只股票的今日收益
        results = []
        for _, row in predictions.iterrows():
            ts_code = row['ts_code']
            info = price_info.get(ts_code, {})
            ret_1d = info.get('ret_7d')  # 用7日收益作为近似
            
            if ret_1d is not None:
                correct = ret_1d > 0
                results.append({
                    'ts_code': ts_code,
                    'name': name_map.get(ts_code, ts_code),
                    'predicted': row.get('predicted', 0),
                    'confidence': row.get('confidence', 0),
                    'return': ret_1d,
                    'correct': correct,
                })
        
        if not results:
            logger.info("无法获取今日收益")
            return None
        
        df_results = pd.DataFrame(results)
        
        # 计算胜率
        total = len(df_results)
        correct = df_results['correct'].sum()
        win_rate = correct / total if total > 0 else 0
        
        # 格式化报告
        content = f"📊 前一天预测验证报告\n"
        content += f"📅 验证日期: {datetime.now().strftime('%Y-%m-%d')}\n"
        content += f"📅 预测日期: {yesterday}\n"
        content += f"📈 预测股票: {total} 只\n\n"
        
        content += f"📊 统计结果:\n"
        content += f"   正确: {correct}/{total}\n"
        content += f"   胜率: {win_rate:.1%}\n\n"
        
        content += "📋 详细结果:\n"
        for _, row in df_results.iterrows():
            emoji = "✅" if row['correct'] else "❌"
            ret_str = f"{row['return']:+.2%}" if row['return'] else "N/A"
            content += f"{emoji} {row['name']} ({row['ts_code']}) - 预测:{row['confidence']:.0%} 实际:{ret_str}\n"
        
        content += f"\n⏰ 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return content
        
    except Exception as e:
        logger.error(f"验证预测失败: {e}")
        return None


def run_daily_report(dry_run: bool = False):
    """运行每日报告"""
    logger.info("=" * 50)
    logger.info("开始生成每日选股报告")
    logger.info("=" * 50)
    
    # 0. 获取股票名称
    logger.info("步骤 0: 获取股票名称...")
    name_map = get_stock_names()
    
    # 0.5 验证前一天预测
    logger.info("步骤 0.5: 验证前一天预测...")
    verify_report = verify_yesterday_predictions(name_map)
    if verify_report:
        if dry_run:
            print("\n" + "=" * 50)
            print("前一天预测验证:")
            print("=" * 50)
            print(verify_report)
            print("=" * 50)
        else:
            send_to_feishu(verify_report)
    
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
    
    # 3.5 保存预测结果
    today = datetime.now().strftime('%Y-%m-%d')
    save_predictions(predictions.head(20), today)
    
    # 4. 获取价格信息
    logger.info("步骤 4: 获取价格信息...")
    top20_codes = predictions.head(20)['ts_code'].tolist()
    price_info = get_stock_price_info(top20_codes)
    
    # 5. 格式化报告
    logger.info("步骤 5: 格式化报告...")
    report = format_report(predictions, today, name_map, price_info)
    
    # 6. 推送到飞书
    if dry_run:
        logger.info("测试模式，不推送飞书")
        print("\n" + "=" * 50)
        print("报告预览:")
        print("=" * 50)
        print(report)
        print("=" * 50)
    else:
        logger.info("步骤 6: 推送到飞书...")
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
