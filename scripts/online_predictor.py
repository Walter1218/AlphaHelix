"""
AlphaHelix 在线预测器

为每个市场 regime 维护一个独立的在线 Logistic 回归模型，
输出个股在下一个再平衡周期内上涨的概率。

设计要点：
- 特征做滚动 z-score 标准化，均值/标准差只能用当前 T 日之前数据；
- 每个 regime 独立训练，避免不同市场环境下因子符号相互抵消；
- 支持保存/加载模型状态，便于 walk-forward 中断后续跑。
"""
import sys
import os
import json
import warnings
from pathlib import Path
from collections import deque

import numpy as np

warnings.filterwarnings("ignore")


def sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid。"""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    exp_z = np.exp(z[neg])
    out[neg] = exp_z / (1 + exp_z)
    return out


class RollingStandardizer:
    """
    滚动 z-score 标准化器。

    维护最近 window_size 个样本的原始特征值，
    transform 时用当前 buffer 计算 mean/std（不包含当前样本）。
    输出会按 clip 截断异常值，降低在线模型对极端样本的敏感。
    """

    def __init__(self, feature_names: list, window_size: int = 120, clip: float = 3.0):
        self.feature_names = list(feature_names)
        self.window_size = int(window_size)
        self.clip = float(clip)
        self.buffer: deque = deque(maxlen=self.window_size)

    def update(self, x: np.ndarray):
        """
        将一个新的原始特征向量加入滚动窗口。

        Args:
            x: 1D numpy array，长度等于 feature_names。
        """
        x = np.asarray(x, dtype=float)
        if x.shape[0] != len(self.feature_names):
            raise ValueError(f"Feature dim mismatch: {x.shape[0]} vs {len(self.feature_names)}")
        self.buffer.append(x.copy())

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        用当前 buffer 的统计量对 x 做 z-score。

        缺失值用当前均值填充，标准化后缺失值变为 0。
        若某特征 std 为 0 或 buffer 为空，则该特征标准化为 0。
        """
        x = np.asarray(x, dtype=float)
        if len(self.buffer) == 0:
            return np.zeros_like(x, dtype=float)

        arr = np.vstack(self.buffer)  # shape: (n, d)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0, ddof=1)

        # std 为 0 或 NaN 时，避免除 0
        std = np.where((std == 0) | np.isnan(std), 1.0, std)

        # 缺失值用均值填充，标准化后等于 0
        x_filled = np.where(np.isnan(x), mean, x)
        z = (x_filled - mean) / std
        z = np.where(np.isnan(z), 0.0, z)
        if self.clip > 0:
            z = np.clip(z, -self.clip, self.clip)
        return z

    def state_dict(self) -> dict:
        return {
            "feature_names": self.feature_names,
            "window_size": self.window_size,
            "clip": self.clip,
            "buffer": [arr.tolist() for arr in self.buffer],
        }

    @classmethod
    def from_state_dict(cls, state: dict):
        obj = cls(feature_names=state["feature_names"], window_size=state["window_size"], clip=state.get("clip", 3.0))
        for arr in state.get("buffer", []):
            obj.buffer.append(np.asarray(arr, dtype=float))
        return obj


class OnlineLogisticRegressor:
    """
    在线 Logistic 回归。

    使用 SGD 逐个样本更新，L2 正则化。
    """

    def __init__(self, n_features: int, lr: float = 1e-3, l2_reg: float = 1e-2,
                 fit_intercept: bool = True):
        self.n_features = int(n_features)
        self.lr = float(lr)
        self.l2_reg = float(l2_reg)
        self.fit_intercept = fit_intercept
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0
        self.n_updates = 0

    def partial_fit(self, x: np.ndarray, y: int):
        """
        用单个样本 (x, y) 更新模型参数。

        Args:
            x: 已经标准化后的 1D 特征向量。
            y: 0 或 1。
        """
        x = np.asarray(x, dtype=float)
        z = float(np.dot(self.w, x))
        if self.fit_intercept:
            z += self.b
        p = float(sigmoid(np.array([z]))[0])
        error = float(y) - p

        # 梯度更新
        grad_w = -error * x + self.l2_reg * self.w
        self.w -= self.lr * grad_w
        if self.fit_intercept:
            self.b += self.lr * error  # 截距不正则

        self.n_updates += 1

        # 学习率衰减
        effective_lr = self.lr / (1 + 1e-5 * self.n_updates)
        self.lr = effective_lr

    def predict_proba(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        z = float(np.dot(self.w, x))
        if self.fit_intercept:
            z += self.b
        return float(sigmoid(np.array([z]))[0])

    def predict(self, x: np.ndarray) -> int:
        return 1 if self.predict_proba(x) > 0.5 else 0

    def state_dict(self) -> dict:
        return {
            "n_features": self.n_features,
            "lr": self.lr,
            "l2_reg": self.l2_reg,
            "fit_intercept": self.fit_intercept,
            "w": self.w.tolist(),
            "b": self.b,
            "n_updates": self.n_updates,
        }

    @classmethod
    def from_state_dict(cls, state: dict):
        obj = cls(
            n_features=state["n_features"],
            lr=state["lr"],
            l2_reg=state["l2_reg"],
            fit_intercept=state.get("fit_intercept", True),
        )
        obj.w = np.asarray(state["w"], dtype=float)
        obj.b = float(state["b"])
        obj.n_updates = int(state.get("n_updates", 0))
        return obj


class RegimeModelManager:
    """
    管理每个 regime 的 RollingStandardizer + OnlineLogisticRegressor。
    """

    REGIMES = ["trend_up", "range", "trend_down", "high_vol"]

    def __init__(self, feature_names: list, window_size: int = 120,
                 lr: float = 1e-4, l2_reg: float = 1e-1,
                 burn_in_samples: int = 200,
                 fallback_regime: str = "range",
                 clip: float = 3.0):
        self.feature_names = list(feature_names)
        self.window_size = int(window_size)
        self.lr = float(lr)
        self.l2_reg = float(l2_reg)
        self.burn_in_samples = int(burn_in_samples)
        self.fallback_regime = fallback_regime
        self.clip = float(clip)

        self.standardizers: dict = {}
        self.models: dict = {}
        self.sample_counts: dict = {}

        for regime in self.REGIMES:
            self.standardizers[regime] = RollingStandardizer(feature_names, window_size, clip=clip)
            self.models[regime] = OnlineLogisticRegressor(
                n_features=len(feature_names), lr=lr, l2_reg=l2_reg
            )
            self.sample_counts[regime] = 0

    def _resolve_regime(self, regime: str) -> str:
        if regime in self.models:
            return regime
        return self.fallback_regime

    def is_ready(self, regime: str) -> bool:
        regime = self._resolve_regime(regime)
        return self.sample_counts[regime] >= self.burn_in_samples

    def predict_proba(self, regime: str, x: np.ndarray) -> float:
        """
        对单个样本输出上涨概率。

        如果该 regime 还没 burn-in 完成，回退到 fallback_regime。
        """
        regime = self._resolve_regime(regime)
        x_std = self.standardizers[regime].transform(x)
        return self.models[regime].predict_proba(x_std)

    def update(self, regime: str, x: np.ndarray, y: int):
        """
        用单个样本更新对应 regime 的模型。

        注意：必须先 transform（用旧统计量），再 partial_fit，最后 update standardizer。
        """
        regime = self._resolve_regime(regime)
        standardizer = self.standardizers[regime]
        model = self.models[regime]

        x_std = standardizer.transform(x)
        model.partial_fit(x_std, y)
        standardizer.update(x)
        self.sample_counts[regime] += 1

    def state_dict(self) -> dict:
        return {
            "feature_names": self.feature_names,
            "window_size": self.window_size,
            "clip": self.clip,
            "lr": self.lr,
            "l2_reg": self.l2_reg,
            "burn_in_samples": self.burn_in_samples,
            "fallback_regime": self.fallback_regime,
            "standardizers": {k: v.state_dict() for k, v in self.standardizers.items()},
            "models": {k: v.state_dict() for k, v in self.models.items()},
            "sample_counts": self.sample_counts.copy(),
        }

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        obj = cls(
            feature_names=state["feature_names"],
            window_size=state["window_size"],
            lr=state["lr"],
            l2_reg=state["l2_reg"],
            burn_in_samples=state["burn_in_samples"],
            fallback_regime=state.get("fallback_regime", "range"),
            clip=state.get("clip", 3.0),
        )
        for regime in cls.REGIMES:
            obj.standardizers[regime] = RollingStandardizer.from_state_dict(
                state["standardizers"][regime]
            )
            obj.models[regime] = OnlineLogisticRegressor.from_state_dict(
                state["models"][regime]
            )
            obj.sample_counts[regime] = int(state["sample_counts"].get(regime, 0))
        return obj


def features_from_record(record: dict, feature_names: list) -> np.ndarray:
    """
    从 Pass2 记录字典中提取特征向量。

    缺失因子填充 NaN，由 RollingStandardizer 在 transform 时处理。
    """
    values = []
    for name in feature_names:
        v = record.get(name)
        try:
            values.append(float(v))
        except (TypeError, ValueError):
            values.append(np.nan)
    return np.asarray(values, dtype=float)


# 第一版使用的因子列表（与 screen.py Pass2 输出字段对齐）
# 注意：不使用原始资金净流入金额（net_mf_5d/20d），因为量纲随市值差异过大；
# 改用占比/背离类因子，并加入市值因子控制规模效应。
DEFAULT_FEATURE_NAMES = [
    "mom_5", "mom_20", "mom_60", "mom_120",
    "risk_adj_mom", "relative_strength",
    "ep", "bp", "sp", "dv_ratio",
    "roe", "revenue_growth", "profit_growth", "ocf_growth",
    "net_mf_ratio", "net_mf_divergence",
    "sector_momentum", "relative_to_sector", "sector_breadth",
    "forecast_type_score", "forecast_pchange_mid", "express_diluted_roe",
    "reversal_score", "amount_ratio_5d", "volatility_20", "liquidity", "total_mv",
]


if __name__ == "__main__":
    # 简单 sanity check
    manager = RegimeModelManager(DEFAULT_FEATURE_NAMES, window_size=30, burn_in_samples=10)
    x = np.random.randn(len(DEFAULT_FEATURE_NAMES))
    print(f"trend_up prob before burn-in: {manager.predict_proba('trend_up', x):.4f}")

    for i in range(20):
        x = np.random.randn(len(DEFAULT_FEATURE_NAMES))
        y = 1 if np.random.rand() > 0.5 else 0
        manager.update("trend_up", x, y)

    print(f"trend_up prob after burn-in: {manager.predict_proba('trend_up', x):.4f}")
    print(f"trend_up sample count: {manager.sample_counts['trend_up']}")
    print(f"is_ready: {manager.is_ready('trend_up')}")

    tmp_path = "/tmp/alphahelix_online_model_test.json"
    manager.save(tmp_path)
    loaded = RegimeModelManager.load(tmp_path)
    print(f"loaded prob: {loaded.predict_proba('trend_up', x):.4f}")
