# 智能选股项目与技术调研

> 调研日期：2026-07-04  
> 来源：GitHub 高 Star 项目、量化金融经典文献、行业实践

---

## 1. 高 Star 开源项目速览

| 项目 | Stars | 定位 | 核心看点 |
|---|---|---|---|
| [microsoft/qlib](https://github.com/microsoft/qlib) | 45.6k | AI 量化投资平台 | 完整 ML 流水线：数据 → 特征(Alpha158/360) → 模型 → 回测 → 组合优化；内置 LightGBM/XGBoost/LSTM/Transformer/GATs/DoubleEnsemble；用 IC/IR、分层收益、换手率评估；官方示例 IR 可达 1.4~2.0。 |
| [wilsonfreitas/awesome-quant](https://github.com/wilsonfreitas/awesome-quant) | 27.4k | 量化资源索引 | 涵盖数据、技术指标、回测、组合优化、因子分析、另类数据等全链路工具。 |
| [stefan-jansen/machine-learning-for-trading](https://github.com/stefan-jansen/machine-learning-for-trading) | 19.5k | 《ML for Trading》第 3 版代码 | 强调"过程即优势"：从数据到生产；使用 Polars/LightGBM/Optuna/PyTorch；涵盖 XGBoost/LightGBM/CatBoost、PatchTST/iTransformer、TabPFN、因果推断、强化学习、MLOps。 |
| [AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL) | 15.6k | 金融强化学习框架 | 用 PPO/A2C/SAC/TD3/DDPG 做股票交易；train-test-trade 流水线；支持多数据源（Tushare、Yahoo、JoinQuant 等）。 |
| [huseinzol05/Stock-Prediction-Models](https://github.com/huseinzol05/Stock-Prediction-Models) | 9.4k | 股价预测模型集合 | LSTM/GRU/Attention/Seq2Seq/CNN-Seq2seq 等深度学习模型； stacking（XGB+RF+GB+Adaboost）和多种 RL agent。 |
| [hudson-and-thames/mlfinlab](https://github.com/hudson-and-thames/mlfinlab) | 4.9k | 金融机器学习工具箱 | 实现 Marcos Lopez de Prado《Advances in Financial ML》：元标签(meta-labeling)、triple-barrier 标签、分数差分、样本权重、purged CV、回测过拟合检测。 |

---

## 2. 智能选股的通用架构

几乎所有高 Star 项目都遵循以下 5 层流水线：

```
数据层
  ↓
特征工程（Alpha 因子 + 标签）
  ↓
预测模型（监督学习 / 强化学习）
  ↓
组合构建（仓位 + 风控）
  ↓
回测评估（IC、IR、成本、最大回撤）
```

### 2.1 数据层

- **价格/量价**：OHLCV、复权价、成交额、波动率。
- **基本面**：财报（point-in-time，用 ann_date 防穿越）、估值、盈利预期。
- **资金流**：主力净流入、北向资金、融资融券、龙虎榜。
- **另类数据**：新闻 sentiment、研报、行业政策、宏观指标。
- **关键原则**：必须 point-in-time，避免 survivorship bias、lookahead bias。

### 2.2 特征工程

- **Alpha 因子**：动量、反转、价值、质量、波动、流动性、情绪、事件等。
- **标准化**：
  - 截面 rank（0~1）或截面 z-score；
  - 行业/市值中性化（减去行业均值，去除共同驱动）。
- **标签设计**：
  - 未来 N 日收益率（回归）；
  - 未来 N 日收益率分位数（分类/排序）；
  - **triple-barrier**：结合止盈、止损、时间退出，比简单持有期更贴近交易；
  - **meta-labeling**：主模型决定方向，副模型决定仓位/是否下注。

### 2.3 预测模型

| 模型类型 | 适用场景 | 优点 | 缺点 |
|---|---|---|---|
| 线性/Ridge/LASSO | 基线、可解释 | 快、不易过拟合 | 捕捉非线性有限 |
| LightGBM/XGBoost/CatBoost | 主流 cross-sectional 选股 | 自动特征交互、对噪声鲁棒、速度快 | 需调参、易过拟合 |
| 神经网络（LSTM/Transformer/GNN） | 时序/图结构 | 表达能力强 | 数据量要求高、黑箱、易过拟合 |
| 强化学习 | 连续决策/执行 | 可直接优化目标（收益-风险） | 训练不稳定、sim-to-real 难 |

**行业共识**：
- 先用 **LightGBM/XGBoost** 跑通 cross-sectional 预测，作为强基线。
- 线性模型作为 sanity check，复杂模型必须显著跑赢基线才使用。
- 预测目标通常是**未来收益率的排序/分位数**，而不是二分类方向。

### 2.4 组合构建

- **从预测到持仓**：
  - 按预测得分分组（如 top 10% 做多，bottom 10% 做空）；
  - 或取 top-k 等权/按得分加权。
- **中性化约束**：
  - 行业中性、市值中性、beta 中性；
  - 控制换手率、单票上限、行业上限。
- **仓位/风控**：
  - Kelly 准则、波动率目标、CVaR 约束；
  - 止损、回撤控制、kill switch。

### 2.5 回测评估

- **预测质量**：IC（Pearson）、rank IC、ICIR、IC 衰减、分位数收益单调性。
- **组合绩效**：年化收益、超额收益、Sharpe、Information Ratio、最大回撤、Calmar。
- **交易成本**：佣金、印花税、滑点、市场冲击。
- **过拟合控制**：purged k-fold CV、combinatorial purged CV、deflated Sharpe ratio、White's Reality Check。

---

## 3. 对 AlphaHelix 的关键启示

### 3.1 我们当前做法的不足

| 我们 | 行业主流 | 差距 |
|---|---|---|
| 在线 Logistic 回归二分类 | LightGBM/XGBoost 回归/排序 + 元标签 | 模型表达力弱、标签信息损失大 |
| 5 日方向准确率 | IC/IR + 分位数收益 + 累计超额 | 方向准确率噪声大，不是最优优化目标 |
| 原始资金流量金额 | 资金流量占比/标准化/市值中性化 | 量纲不一致，模型被大市值主导 |
| 单模型 / 4 个 regime 小模型 | 滚动重训练 + 集成 | 样本量不足，模型不稳定 |
| 无行业/市值中性化 | 截面中性化是标配 | 暴露行业 beta，选股信号被稀释 |
| 无交易成本 | 回测必须扣成本 | 高换手策略会失效 |

### 3.2 建议的改进方向

1. **换模型**：用 **LightGBM/XGBoost** 做未来收益率回归或排序预测，替代在线 Logistic 回归。
2. **换标签**：预测未来 5/10/20 日**收益率**（或相对基准的超额收益），而不是二分类方向。
3. **特征处理**：
   - 截面 rank 标准化；
   - 行业/市值中性化；
   - 移除原始金额，使用 ratio/占比类因子；
   - 加入更多数据源：北向、融资融券、龙虎榜、披露日预告。
4. **评估升级**：
   - 每期计算预测得分与未来收益的 rank IC；
   - 看分位数收益是否单调（top 组 > bottom 组）；
   - 再进入组合回测。
5. **组合构建**：
   - 按预测得分分 5/10 组，先验证 group 单调性；
   - 取 top-k 时加入行业/市值约束和换手率惩罚；
   - 加入交易成本。
6. **验证纪律**：
   - 用 purged walk-forward 或严格 train/test 切分；
   - 调参只能在训练集，测试集只做一次最终评估；
   - 报告 deflated Sharpe / ICIR。

### 3.3 最小可行下一步

1. **离线批量 Ridge/LightGBM 信号验证**：用 2024 年训练，2025-2026 年测试，看 rank IC 是否显著 > 0。
2. **特征工程升级**：
   - 行业/市值中性化；
   - 资金流改为 ratio；
   - 加入北向/融资融券/龙虎榜。
3. **组合层升级**：
   - 分位数分组验证；
   - top-k 等权 + 行业上限；
   - 交易成本。
4. **如果离线模型有信号**，再考虑在线/滚动更新版本。

---

## 4. 参考资源

- [Qlib 官方文档](https://qlib.readthedocs.io/)
- [Advances in Financial Machine Learning](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089) — Marcos Lopez de Prado
- [Machine Learning for Trading](https://ml4trading.io/) — Stefan Jansen
- [FinRL 论文与教程](https://github.com/AI4Finance-Foundation/FinRL)
- [awesome-quant 资源列表](https://github.com/wilsonfreitas/awesome-quant)
