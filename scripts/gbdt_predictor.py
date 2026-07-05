"""
AlphaHelix GBDT 预测器封装

- 加载 model_trainer.py 保存的模型 + 特征元数据
- 对 screen.py 输出的候选 DataFrame 做一致的特征工程
- 返回每只股票的未来收益预测得分

设计原则：
- 与训练流程使用同一套特征工程（winsorize / 行业市值中性化 / rank）
- 缺失特征自动补 0；多余特征忽略
- 支持 LightGBM / XGBoost
"""
import sys
import os
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import build_numeric_features

# 延迟导入树模型库，未安装时给出友好提示
_lightgbm = None
_xgboost = None


def _import_lightgbm():
    global _lightgbm
    if _lightgbm is None:
        try:
            import lightgbm as lgb
            _lightgbm = lgb
        except ImportError as e:
            raise ImportError("使用 GBDT 模式需要安装 lightgbm: pip install lightgbm") from e
    return _lightgbm


def _import_xgboost():
    global _xgboost
    if _xgboost is None:
        try:
            import xgboost as xgb
            _xgboost = xgb
        except ImportError as e:
            raise ImportError("使用 XGBoost 模式需要安装 xgboost: pip install xgboost") from e
    return _xgboost


class GBDTScorePredictor:
    """GBDT 打分器。线程不安全，建议每个进程只初始化一次。"""

    def __init__(self, model_path: str, model_type: Optional[str] = None,
                 feature_cols: Optional[list] = None):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        self.model_type = model_type
        self.feature_cols = feature_cols
        self._load_meta()
        self._load_model()

    def _load_meta(self):
        """加载特征元数据；如果构造时未提供，尝试从同目录 meta 文件读取。"""
        if self.model_type and self.feature_cols:
            return

        meta_path = str(self.model_path).replace(".txt", "_meta.json")
        # 兼容旧版 double-suffix 产物
        legacy_path = str(self.model_path).replace(".txt", "_meta_meta.json")
        for p in [meta_path, legacy_path]:
            if Path(p).exists():
                with open(p, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.model_type = self.model_type or meta.get("model_type", "lightgbm")
                self.feature_cols = self.feature_cols or meta.get("feature_cols")
                return

        if not self.model_type:
            raise ValueError(f"无法推断模型类型，请提供 model_type。已尝试 meta 路径: {meta_path}")
        if not self.feature_cols:
            raise ValueError(f"无法读取特征列表，请提供 feature_cols。已尝试 meta 路径: {meta_path}")

    def _load_model(self):
        if self.model_type == "lightgbm":
            lgb = _import_lightgbm()
            self.model = lgb.Booster(model_file=str(self.model_path))
        elif self.model_type == "xgboost":
            xgb = _import_xgboost()
            self.model = xgb.Booster(model_file=str(self.model_path))
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """对原始候选 DataFrame 做与训练一致的特征工程。"""
        # 确保行业/市值字段存在，供中性化使用
        if "industry" not in df.columns:
            df["industry"] = "未知"
        if "total_mv" not in df.columns:
            df["total_mv"] = np.nan

        # 只对模型期望的特征做处理
        df = build_numeric_features(
            df,
            feature_cols=self.feature_cols,
            neutralize=True,
            rank=True,
            winsorize=True,
        )
        return df

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """输入 screen.py 的候选 DataFrame，返回预测得分数组。"""
        df = self.prepare_features(df)

        # 补齐缺失列
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0

        X = df[self.feature_cols].astype(float).values
        if self.model_type == "lightgbm":
            return self.model.predict(X, num_iteration=self.model.best_iteration)
        elif self.model_type == "xgboost":
            xgb = _import_xgboost()
            dmatrix = xgb.DMatrix(X)
            return self.model.predict(dmatrix)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")


def find_latest_model(model_dir: Path = Path("memory/models"), horizon: int = 10,
                      target: str = "excess_return", model_type: str = "lightgbm") -> Optional[Path]:
    """按命名约定查找最新保存的模型。"""
    candidates = [
        model_dir / f"gbdt_h{horizon}_split_{target}.{model_type}.txt",
        model_dir / f"gbdt_h{horizon}_walkforward_{target}.{model_type}.txt",
        model_dir / f"gbdt_h{horizon}_latest.{model_type}.txt",
        model_dir / f"gbdt_latest_h{horizon}.{model_type}.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None
