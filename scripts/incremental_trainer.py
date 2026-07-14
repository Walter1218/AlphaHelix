"""
增量训练模块

支持 DoubleEnsemble 模型的增量训练和持久化。

功能：
1. 模型保存/加载
2. 增量训练（用新数据继续训练旧模型）
3. 模型版本管理
4. 模型性能跟踪

用法：
    python incremental_trainer.py --train           # 首次训练
    python incremental_trainer.py --incremental     # 增量训练
    python incremental_trainer.py --evaluate        # 评估模型
"""
import sys
import os
import json
import logging
import argparse
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 模型目录
MODEL_DIR = "memory/models"


class IncrementalDoubleEnsemble:
    """
    增量式 DoubleEnsemble 模型
    
    支持：
    1. 首次训练
    2. 增量训练（用新数据继续训练旧模型）
    3. 模型保存/加载
    4. 模型版本管理
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        num_leaves: int = 15,
        max_depth: int = 4,
        min_child_samples: int = 50,
        reg_alpha: float = 1.0,
        reg_lambda: float = 1.0,
        random_state: int = 500,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        
        self.model1 = None
        self.model2 = None
        self.scaler = None
        self.selected_features = None
        self.version = 0
        self.training_history = []
    
    def train(self, X: np.ndarray, y: np.ndarray, features: List[str]):
        """
        首次训练
        
        Args:
            X: 特征矩阵
            y: 目标变量
            features: 特征列表
        """
        import lightgbm as lgb
        from sklearn.preprocessing import StandardScaler
        
        logger.info(f"开始首次训练: {X.shape[0]} 样本, {X.shape[1]} 特征")
        
        # 标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # 第一阶段
        self.model1 = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            verbose=-1
        )
        self.model1.fit(X_scaled, y)
        
        # 第二阶段
        residuals = y - self.model1.predict(X_scaled)
        self.model2 = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state + 1,
            verbose=-1
        )
        self.model2.fit(X_scaled, residuals)
        
        self.selected_features = features
        self.version = 1
        self.training_history.append({
            'version': self.version,
            'timestamp': datetime.now().isoformat(),
            'samples': X.shape[0],
            'features': X.shape[1],
            'type': 'initial'
        })
        
        logger.info(f"首次训练完成: 版本 {self.version}")
    
    def incremental_train(self, X_new: np.ndarray, y_new: np.ndarray, n_boost: int = 20):
        """
        增量训练
        
        Args:
            X_new: 新特征矩阵
            y_new: 新目标变量
            n_boost: 增量训练轮数
        """
        import lightgbm as lgb
        
        if self.model1 is None or self.model2 is None:
            raise ValueError("模型未初始化，请先调用 train()")
        
        logger.info(f"开始增量训练: {X_new.shape[0]} 新样本, {n_boost} 轮")
        
        # 标准化新数据
        X_scaled = self.scaler.transform(X_new)
        
        # 获取当前 booster
        booster1 = self.model1.booster_
        booster2 = self.model2.booster_
        
        # 增量训练第一阶段
        new_n_estimators = self.model1.n_estimators + n_boost
        self.model1 = lgb.LGBMRegressor(
            n_estimators=new_n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            verbose=-1
        )
        self.model1.fit(X_scaled, y_new, init_model=booster1)
        
        # 增量训练第二阶段
        residuals = y_new - self.model1.predict(X_scaled)
        new_n_estimators2 = self.model2.n_estimators + n_boost
        self.model2 = lgb.LGBMRegressor(
            n_estimators=new_n_estimators2,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state + 1,
            verbose=-1
        )
        self.model2.fit(X_scaled, residuals, init_model=booster2)
        
        self.version += 1
        self.training_history.append({
            'version': self.version,
            'timestamp': datetime.now().isoformat(),
            'samples': X_new.shape[0],
            'features': X_new.shape[1],
            'type': 'incremental',
            'n_boost': n_boost
        })
        
        logger.info(f"增量训练完成: 版本 {self.version}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if self.model1 is None or self.model2 is None:
            raise ValueError("模型未初始化")
        
        X_scaled = self.scaler.transform(X)
        pred1 = self.model1.predict(X_scaled)
        pred2 = self.model2.predict(X_scaled)
        return pred1 + pred2
    
    def save(self, path: str = None):
        """保存模型"""
        import lightgbm as lgb
        
        if path is None:
            os.makedirs(MODEL_DIR, exist_ok=True)
            path = os.path.join(MODEL_DIR, "double_ensemble.pkl")
        
        # 保存 LightGBM 模型
        model1_path = path.replace('.pkl', '_model1.txt')
        model2_path = path.replace('.pkl', '_model2.txt')
        
        self.model1.booster_.save_model(model1_path)
        self.model2.booster_.save_model(model2_path)
        
        # 保存元数据
        metadata = {
            'version': self.version,
            'selected_features': self.selected_features,
            'training_history': self.training_history,
            'model1_path': model1_path,
            'model2_path': model2_path,
            'model1_n_estimators': self.model1.n_estimators,
            'model2_n_estimators': self.model2.n_estimators,
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist(),
            'params': {
                'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate,
                'num_leaves': self.num_leaves,
                'max_depth': self.max_depth,
                'min_child_samples': self.min_child_samples,
                'reg_alpha': self.reg_alpha,
                'reg_lambda': self.reg_lambda,
                'random_state': self.random_state,
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"模型已保存: {path} (版本 {self.version})")
    
    @classmethod
    def load(cls, path: str = None) -> 'IncrementalDoubleEnsemble':
        """加载模型"""
        import lightgbm as lgb
        from sklearn.preprocessing import StandardScaler
        
        if path is None:
            path = os.path.join(MODEL_DIR, "double_ensemble.pkl")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        
        # 加载元数据
        with open(path, 'rb') as f:
            metadata = pickle.load(f)
        
        # 创建实例
        instance = cls(**metadata['params'])
        instance.version = metadata['version']
        instance.selected_features = metadata['selected_features']
        instance.training_history = metadata['training_history']
        
        # 加载 LightGBM 模型
        booster1 = lgb.Booster(model_file=metadata['model1_path'])
        booster2 = lgb.Booster(model_file=metadata['model2_path'])
        
        # 创建 LGBMRegressor 并设置 booster
        params1 = metadata['params'].copy()
        params1['n_estimators'] = metadata['model1_n_estimators']
        instance.model1 = lgb.LGBMRegressor(**params1)
        
        # 使用 dummy fit 初始化
        dummy_X = np.zeros((2, len(metadata['selected_features'])))
        dummy_y = [0, 0]
        instance.model1.fit(dummy_X, dummy_y)
        instance.model1._Booster = booster1
        instance.model1._n_features = len(metadata['selected_features'])
        instance.model1._n_features_in = len(metadata['selected_features'])
        
        params2 = metadata['params'].copy()
        params2['n_estimators'] = metadata['model2_n_estimators']
        instance.model2 = lgb.LGBMRegressor(**params2)
        instance.model2.fit(dummy_X, dummy_y)
        instance.model2._Booster = booster2
        instance.model2._n_features = len(metadata['selected_features'])
        instance.model2._n_features_in = len(metadata['selected_features'])
        
        # 恢复 scaler
        instance.scaler = StandardScaler()
        instance.scaler.mean_ = np.array(metadata['scaler_mean'])
        instance.scaler.scale_ = np.array(metadata['scaler_scale'])
        instance.scaler.n_features_in_ = len(metadata['selected_features'])
        
        logger.info(f"模型已加载: 版本 {instance.version}")
        return instance
    
    def get_info(self) -> Dict:
        """获取模型信息"""
        return {
            'version': self.version,
            'features': len(self.selected_features) if self.selected_features else 0,
            'training_history': self.training_history,
            'model1_estimators': self.model1.n_estimators if self.model1 else 0,
            'model2_estimators': self.model2.n_estimators if self.model2 else 0,
        }


def load_and_prepare_data(months_back: int = 1) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """加载并准备数据"""
    # 加载数据
    df = pd.read_parquet('memory/dataset/features_h10_full.parquet')
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
    
    # 使用最近 N 个月数据
    df['ym'] = df['date'].dt.to_period('M')
    months = sorted(df['ym'].unique())
    train_months = months[-months_back:]
    train_df = df[df['ym'].isin(train_months)]
    
    # 特征选择（用 IC）
    from scipy.stats import spearmanr
    
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
    
    X = train_df[selected].values
    y = train_df['excess_return'].values
    
    return X, y, selected, train_df


def run_initial_training():
    """运行首次训练"""
    logger.info("=" * 50)
    logger.info("开始首次训练")
    logger.info("=" * 50)
    
    # 加载数据
    X, y, features, df = load_and_prepare_data(months_back=12)
    logger.info(f"数据加载完成: {X.shape[0]} 样本, {len(features)} 特征")
    
    # 训练模型
    model = IncrementalDoubleEnsemble()
    model.train(X, y, features)
    
    # 保存模型
    model.save()
    
    # 输出模型信息
    info = model.get_info()
    logger.info(f"模型信息: {json.dumps(info, indent=2)}")
    
    return model


def run_incremental_training():
    """运行增量训练"""
    logger.info("=" * 50)
    logger.info("开始增量训练")
    logger.info("=" * 50)
    
    # 加载旧模型
    try:
        model = IncrementalDoubleEnsemble.load()
        logger.info(f"加载旧模型: 版本 {model.version}")
    except FileNotFoundError:
        logger.warning("未找到旧模型，将进行首次训练")
        return run_initial_training()
    
    # 加载新数据（最近1个月）
    df = pd.read_parquet('memory/dataset/features_h10_full.parquet')
    df['date'] = pd.to_datetime(df['date'])
    df['ym'] = df['date'].dt.to_period('M')
    
    months = sorted(df['ym'].unique())
    new_month = months[-1]  # 最近1个月
    new_df = df[df['ym'] == new_month].copy()
    
    # 特征工程（与训练时一致）
    feature_cols = model.selected_features
    
    # 确保特征存在
    for col in feature_cols:
        if col not in new_df.columns:
            new_df[col] = 0
    
    X_new = new_df[feature_cols].values
    y_new = new_df['excess_return'].values
    
    logger.info(f"新数据: {X_new.shape[0]} 样本")
    
    # 增量训练
    model.incremental_train(X_new, y_new, n_boost=20)
    
    # 保存模型
    model.save()
    
    # 输出模型信息
    info = model.get_info()
    logger.info(f"模型信息: {json.dumps(info, indent=2)}")
    
    return model


def run_evaluate():
    """评估模型"""
    logger.info("=" * 50)
    logger.info("开始评估模型")
    logger.info("=" * 50)
    
    # 加载模型
    try:
        model = IncrementalDoubleEnsemble.load()
        logger.info(f"加载模型: 版本 {model.version}")
    except FileNotFoundError:
        logger.error("未找到模型")
        return
    
    # 加载数据
    df = pd.read_parquet('memory/dataset/features_h10_full.parquet')
    df['date'] = pd.to_datetime(df['date'])
    df['ym'] = df['date'].dt.to_period('M')
    
    months = sorted(df['ym'].unique())
    test_month = months[-1]
    test_df = df[df['ym'] == test_month].copy()
    
    # 特征工程
    feature_cols = model.selected_features
    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = 0
    
    X_test = test_df[feature_cols].values
    y_test = test_df['excess_return'].values
    
    # 预测
    predictions = model.predict(X_test)
    
    # 计算指标
    from scipy.stats import spearmanr
    ic, _ = spearmanr(predictions, y_test)
    ic = ic if not np.isnan(ic) else 0
    
    # 计算胜率
    test_df['predicted'] = predictions
    test_df['rank'] = test_df.groupby('date')['predicted'].rank(ascending=False)
    top10 = test_df[test_df['rank'] <= 10]
    win_rate = (top10['excess_return'] > 0).mean()
    
    logger.info(f"评估结果:")
    logger.info(f"  IC: {ic:.4f}")
    logger.info(f"  胜率: {win_rate:.1%}")
    logger.info(f"  模型版本: {model.version}")
    logger.info(f"  训练历史: {len(model.training_history)} 次")


def main():
    parser = argparse.ArgumentParser(description="增量训练模块")
    parser.add_argument("--train", action="store_true", help="首次训练")
    parser.add_argument("--incremental", action="store_true", help="增量训练")
    parser.add_argument("--evaluate", action="store_true", help="评估模型")
    parser.add_argument("--info", action="store_true", help="显示模型信息")
    args = parser.parse_args()
    
    if args.train:
        run_initial_training()
    elif args.incremental:
        run_incremental_training()
    elif args.evaluate:
        run_evaluate()
    elif args.info:
        try:
            model = IncrementalDoubleEnsemble.load()
            info = model.get_info()
            print(json.dumps(info, indent=2))
        except FileNotFoundError:
            print("未找到模型")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
