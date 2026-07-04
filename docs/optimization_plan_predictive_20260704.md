# AlphaHelix 预测型持仓优化方案

> 文档日期：2026-07-04  
> 状态：待评审  
> 关联讨论：方向准确率定义、在线学习训练方式、Pass2 预测化改造

---

## 1. 背景与问题

当前 `scripts/screen.py` 的 Pass2 本质上是**相对排序**：把因子做 rank 标准化后按固定权重加权，取分数最高的 N 只股票。这带来几个问题：

1. **没有绝对预测**：`total_score` 只表示“相对其他候选股更好”，不表示“未来会涨”。
2. **强制满仓**：`--top-n` 会强制选 N 只，即使模型认为所有候选股都没机会。
3. **方向准确率失真**：当前方向准确率 = Top-N 中有多少只正收益。但 Top-N 里可能根本没有被模型“看涨”的票。
4. **权重优化触顶**：在 30 个月样本上，无论在线学习、IC 校准、per-regime 权重还是防御仓位，都无法稳定跑赢默认静态权重，说明在排序层雕花已接近极限。

本方案的目标是把 Pass2 从“排序器”改造成**预测未来收益/方向的回归/分类模型**，并配合阈值机制生成真正的持仓列表。

---

## 2. 核心设计原则

| 原则 | 说明 |
|---|---|
| **预测先于决策** | 每只候选股必须先得到未来 H 天收益/方向的预测值，再决定是否入选。 |
| **阈值制持仓** | 只有预测值超过阈值的股票才进入 `holding_list`；否则空仓。 |
| **在线学习** | 每个 (股票, 周期) 样本都是一个训练样本；先预测、后观察、再更新。 |
| **无未来函数** | 特征只能用 T 日及之前数据；标签只能用 T+H 日及之前数据；模型更新只能在标签暴露后发生。 |
| **简单模型优先** | 先用在线线性/Ridge 回归，避免复杂模型在小样本下过拟合。 |
| **预测频率 = 再平衡频率 = 更新频率** | 消除重叠标签和复利评估歧义。 |

---

## 3. 目标变量定义

### 3.1 分类目标（第一版）

```python
label_i(t) = 1 if close_i(t+Δ) > close_i(t) else 0
```

模型输出“未来一个再平衡周期 Δ 内上涨”的概率。

### 3.2 回归目标（备选）

```python
label_i(t) = close_i(t+Δ) / close_i(t) - 1
```

若二分类效果不佳，可切换到预测未来收益率的回归目标。

### 3.3 超额收益目标（不推荐作为唯一目标）

虽然超额收益是最终目标，但把相对基准收益作为模型目标会引入额外噪声（基准本身也在波动）。建议先用绝对收益方向训练模型，再用阈值/仓位控制来处理系统性风险。

---

## 4. 模型设计

### 4.1 模型选择：分 Regime 在线二分类模型

每个 regime 维护一个独立的在线二分类器（在线 Logistic 回归或在线线性分类器）：

- `trend_up`、`range`、`trend_down`、`high_vol` 各一个模型；
- 每期用当前 regime 对应的模型做预测；
- 该期标签暴露后，只更新对应 regime 的模型。

第一版使用 **在线 Logistic 回归**：

- 输出上涨概率；
- 可用 sigmoid 交叉熵损失做 SGD；
- L2 正则化防止过拟合；
- 系数可解释。

备选升级路径：
- Passive-Aggressive 分类器；
- 在线梯度提升 / 贝叶斯逻辑回归；
- 若样本量不足，4 个 regime 模型可共享部分系数（multi-task online learning）。

### 4.2 特征输入

保留 Pass1 阶段计算的全部因子，并在 Pass2 阶段统一标准化：

- 动量类：`mom_5`, `mom_20`, `mom_60`, `mom_120`, `risk_adj_mom`, `relative_strength`
- 估值类：`ep`, `bp`, `sp`, `dividend_yield`
- 质量类：`roe`, `revenue_growth`, `profit_growth`, `ocf_growth`
- 资金类：`net_mf_5d`, `net_mf_20d`, `net_mf_ratio`, `net_mf_divergence`
- 行业类：`sector_momentum`, `relative_to_sector`, `sector_breadth`
- 事件类：`forecast_type_score`, `forecast_pchange_mid`, `express_diluted_roe`
- 反转/流动性：`reversal_score`, `amount_ratio_5d`, `volatility_20`, `liquidity`

### 4.3 滚动特征标准化

所有特征必须做**滚动 z-score**：

```python
mean_f(t) = mean(feature_f values up to t)
std_f(t)  = std(feature_f values up to t)
x_f(t)    = (feature_f(t) - mean_f(t)) / std_f(t)
```

- 滚动窗口建议 120 个交易日；
- `mean`/`std` 只能用 T 日之前的数据；
- 缺失值填充为 0（即该期因子等于历史均值）。

### 4.4 更新公式（在线 Logistic 回归 SGD）

对于每个训练样本 `(x, y)`，其中 `y ∈ {0, 1}`：

```python
z = w · x
p = sigmoid(z)          # 预测上涨概率
error = y - p
grad = -error * x + lambda_reg * w
w = w - lr * grad
```

- `lr`：学习率，建议初始 1e-4 ~ 1e-3，可随样本数衰减；
- `lambda_reg`：L2 正则系数，建议 1e-3 ~ 1e-2；
- 初始权重 `w` 可设为 0 或当前硬编码权重的标准化版本。

---

## 5. 持仓构建（滚动再平衡）

### 5.1 再平衡规则

每个再平衡日 `T`：

```python
# 当前持仓 + Pass1 候选股一起重新打分
scored = []
for stock in current_holdings + pass1_candidates:
    prob = regime_model.predict_proba(stock.features)
    if prob > threshold:
        scored.append((stock, prob))

# 按上涨概率排序，最多保留 max_positions 只
scored.sort(key=lambda x: x[1], reverse=True)
new_holdings = scored[:max_positions]
```

- 不在 `new_holdings` 中的原持仓股票，在 `T` 日卖出；
- 在 `new_holdings` 中的股票，在 `T` 日买入/继续持有；
- `threshold`：初始 0.5（预测上涨概率 > 50%）；
- `max_positions`：对应原 `--top-n`，只是上限，不是必须满仓。

### 5.2 空仓机制

如果当前持仓和候选股中没有任何票超过 `threshold`，则 `new_holdings` 为空，当期空仓，组合收益 = 0。

### 5.3 仓位分配（第一版等权）

- 每只股票等权；
- 后续可升级为按上涨概率加权，或按预测夏普加权。

### 5.4 持仓周期

- 不再有固定持有期；
- 一只股票会一直被持有，直到下一期再平衡时它的预测概率掉到 threshold 以下。

---

## 6. 训练与预测节奏

### 6.1 时间线

```
T 日收盘（再平衡日）
  ├─ 判断当前 regime
  ├─ 用对应 regime 的模型对所有持仓 + 候选股打分
  ├─ 生成 new_holdings
  └─ 按 new_holdings 调仓，持有到下一个再平衡日 T+Δ

T+Δ 日收盘
  ├─ 计算过去 Δ 天内每只被持仓股票的真实涨跌 label
  ├─ 用这些样本更新对应 regime 的模型
  └─ 更新后的模型用于下一次再平衡
```

### 6.2 再平衡周期 Δ

- 预测频率 = 再平衡频率 = 评估频率 = Δ；
- 建议第一版 `Δ = 5 个交易日`（周频），兼顾样本数量与交易摩擦；
- 若 Δ 过大，模型对市场变化反应慢；若 Δ 过小，换手率和成本上升。

### 6.3 Burn-in 期

每个 regime 的模型冷启动时参数不稳定，前 B 个该 regime 出现的周期只收集样本、更新模型，不计入绩效：

- 建议每个 regime 至少 B = 6~10 个样本后再开始交易；
- Burn-in 期间可以模拟交易用于观察，但不纳入最终评估指标。

---

## 7. 评估指标

| 指标 | 定义 | 目标 |
|---|---|---|
| **方向准确率** | 每个再平衡周期内，被持仓股票中实际上涨的比例 | > 55%（长期目标 70%） |
| **持仓平均收益** | 每个再平衡周期内，持仓组合的平均绝对收益 | > 0 |
| **持仓平均超额收益** | 持仓平均收益 - 同期基准收益 | > 0 |
| **累计组合收益** | 按再平衡周期复利计算的组合 NAV 收益 | > 0 |
| **累计超额收益** | 组合累计收益 - 基准累计收益 | > 0 |
| **持仓覆盖率** | 非空仓期数 / 总期数 | 避免过度空仓 |
| **平均持仓数量** | 每期 `holding_list` 长度 | 反映模型信心 |
| **换手率** | 相邻两期持仓变化市值占比 | 控制交易成本 |
| **系数稳定性** | 各 regime 模型滚动系数的标准差 | 不过度震荡 |
| **预测-真实相关性** | 预测上涨概率 与 实际涨跌 的相关系数 | > 0 |

---

## 8. 系统改造清单

### 8.1 新增模块

- `scripts/online_predictor.py`
  - `OnlineLogisticRegressor` 类：每个 regime 一个实例，支持 `partial_fit` 和 `predict_proba`；
  - `RollingStandardizer` 类：滚动 z-score；
  - `RegimeModelState` 管理：保存/加载 4 个 regime 的模型权重、均值、标准差；
  - 模型持久化到 `memory/models/online_regime_models.json`。

### 8.2 改造 `scripts/screen.py`

- Pass1 保留，继续作为候选池生成器；
- Pass2 不再调用 `compute_weighted_score`；
- Pass2 调用 `online_predictor` 为每只候选股/持仓股输出 `predicted_up_prob`；
- 按 `threshold` 和 `max_positions` 生成 `new_holdings`；
- 输出 JSON 新增字段：`predicted_up_prob`、`regime`；
- 移除对 `memory/weights/*_latest.json` 动态权重的依赖（或仅作为 fallback）。

### 8.3 改造 `scripts/walkforward.py`

- 每期循环：
  1. 判断当前 regime；
  2. 用对应 regime 的模型对所有持仓 + 候选股打分；
  3. 生成 `new_holdings`，执行再平衡；
  4. 持有到下一再平衡日，评估真实收益；
  5. 用本期完结样本更新对应 regime 的模型；
  6. 保存模型状态。
- 每个 regime 前 B 个样本为 burn-in，只更新不纳入绩效；
- 新增 CLI 参数：
  - `--predictor-model`：模型类型（默认 `online_logistic`）；
  - `--rebalance-days`：再平衡周期 Δ（默认 5 个交易日）；
  - `--threshold`：入选阈值（默认 0.5）；
  - `--max-positions`：最大持仓数（替代 `--top-n`）；
  - `--burn-in-samples`：每个 regime 的 burn-in 样本数（默认 200）；
  - `--learning-rate`：在线学习率；
  - `--l2-reg`：正则系数。
- 移除/废弃当前 `--online-update`、`--ic-calibrate`、`--min-score` 的实现。

### 8.4 改造 `scripts/evaluate.py`

- 支持评估空 `holding_list`：组合收益 = 0，方向准确率 = NaN；
- 按再平衡周期维护组合 NAV，计算累计收益；
- 输出中新增 `predicted_up_prob` 与 `actual_return` 的对比；
- 记录每次再平衡的买入/卖出列表，用于计算换手率。

### 8.5 新增诊断脚本

- `scripts/diagnose_predictor.py`
  - 绘制各 regime 因子系数热力图；
  - 绘制预测概率分布 vs 实际涨跌分布；
  - 计算分位数命中率（预测概率前 20% 的股票实际上涨比例）；
  - 输出按 regime 分组的性能统计与样本数量。

---

## 9. 实施路线图

| 阶段 | 任务 | 验收标准 |
|---|---|---|
| **Phase 1** | 实现 `scripts/online_predictor.py` + `RollingStandardizer` | 可独立训练/预测/保存/加载 |
| **Phase 2** | 改造 `screen.py` Pass2 为预测型 | 输出包含 `predicted_up_prob`，支持 threshold |
| **Phase 3** | 改造 `walkforward.py` 支持在线更新 | 30 个月 walk-forward 跑通，模型逐期更新 |
| **Phase 4** | 加入 burn-in、评估指标、诊断脚本 | 可输出系数热力图、预测分布、分位数命中率 |
| **Phase 5** | 超参数 walk-forward 调优 | 在 2024 训练、2025-2026 测试的 OOS 上验证 |
| **Phase 6** | 接入交易成本与仓位分配 | 回测扣除印花税+佣金+滑点 |

---

## 10. 反时间穿越纪律

- 特征滚动窗口只能用 T 日之前数据；
- 标签只能用 T+Δ 日及之前收盘价计算；
- 模型更新必须在 T+Δ 日之后；
- 不允许用全样本训练后再回测同一区间；
- 所有调参必须在独立的 OOS 区间上最终验证一次。

---

## 11. 风险与应对

| 风险 | 应对 |
|---|---|
| 底层因子无预测力 | Phase 4 先做单因子滚动 IC 诊断，剔除无效因子；若多数因子无效，需引入新数据源。 |
| 在线学习对噪声过拟合 | 使用 Ridge 正则 + 学习率衰减 + burn-in。 |
| 空仓过多错过反弹 | `threshold` 从 0 开始，逐步测试；监控持仓覆盖率。 |
| 特征间多重共线性 | Ridge 正则本身可缓解；必要时做 VIF 分析并剔除高相关因子。 |
| 横截面样本相关性 | 同一期股票收益存在市场因子相关，有效样本量小于名义样本量；用更长的滚动窗口和更保守的显著性标准。 |
| 模型系数漂移 | 诊断脚本监控系数稳定性；必要时切换到卡尔曼滤波。 |

---

## 12. 第一版建议参数

| 参数 | 建议值 |
|---|---|
| 再平衡周期 Δ | 5 个交易日 |
| 模型 | 分 Regime 在线 Logistic 回归 |
| 学习率 lr | 1e-4 ~ 1e-3 |
| L2 正则 lambda_reg | 1e-3 ~ 1e-2 |
| 特征滚动窗口 | 120 个交易日 |
| 入选 threshold | 0.5 |
| 最大持仓 max_positions | 10 |
| 每个 regime burn-in 样本数 | 200（约 1~2 年周频样本） |

---

## 13. 待补充与待决策事项

以下问题在第一版实现前需要明确或补齐到方案中：

### 13.1 缺失标签的处理

- 若某只股票在持有期内停牌、退市或无法获取 T+H 收盘价，则该样本没有有效 label。
- 处理方案：**不将该样本加入训练集**，也不计入该期评估。
- 如果一期中多数股票缺失，则该期跳过交易，仅更新模型（若有部分有效样本）。

### 13.2 标签稳健性

- 个股短期收益率分布肥尾，涨停/跌停会造成异常 label。
- 第一版先用原始收益率；若模型对异常值敏感，可切换到：
  - Winsorize label（如 1%/99% 分位数截断）；
  - Huber loss 替代 MSE；
  - 对数收益率 `(log(close_t+H) - log(close_t))`。

### 13.3 阈值选择方法

- 初始 `threshold = 0.0`，即只选预测正收益的股票。
- 后续需要在 walk-forward 框架下校准阈值：
  - 用 burn-in 期后的一个验证段（如 2024-07 至 2024-12）尝试 `threshold ∈ {-0.01, 0, 0.005, 0.01, 0.02}`；
  - 选择使“持仓方向准确率 × 持仓覆盖率”最大的阈值；
  - 最终阈值固定后，在 2025-2026 做唯一一次测试。

### 13.4 基准对照组

- 除了和沪深300 比超额，还需要保留一个**策略内部 baseline**：
  - 当前静态 Top-N 策略；
  - 等权持有 Pass1 候选池。
- 新模型必须显著跑赢这个内部 baseline，才能说明预测模型有价值。

### 13.5 随机性与可复现

- SGD 更新顺序会影响最终权重。
- 每期训练样本按 `ts_code` 字典序排序后更新，确保结果可复现。
- 初始权重固定为 0 或当前硬编码权重的标准化版本，不使用随机初始化。

### 13.6 分 Regime 模型的样本量

- 4 个 regime 模型需要足够样本才能稳定。
- 如果某些 regime 出现频率太低（如 `trend_down` 很少），该模型可能欠拟合。
- 应对方案：
  - 先统计 2024-2026 各 regime 出现次数；
  - 若某 regime 样本过少，可考虑将该 regime 与相似 regime 合并（如 `trend_down` + `high_vol`），或回退到统一模型。

### 13.7 交易成本纳入评估

- 虽然 Phase 6 才正式接入交易成本，但第一版评估就应记录换手率，并粗略估算：
  - 印花税 0.1% 单边（卖出）；
  - 佣金 0.02% 双边；
  - 滑点 0.1% 双边。
- 若 threshold 导致频繁空仓/建仓，换手率可能很高，侵蚀收益。

### 13.8 因子有效性前置诊断

- 在写模型前，先用 `scripts/factor_ic.py` 计算各因子在目标持有期 H=10 下的滚动 rank IC。
- 如果绝大多数因子 IC 接近 0 或 IR < 0.3，说明当前因子体系本身无信号，应先引入新数据源，而不是直接上在线模型。

---

## 14. 成功标准

在 2024-01 至 2026-06 的滚动再平衡回测中（扣除 burn-in 期）：

1. 每个再平衡周期内，被持仓股票的方向准确率 > 55%；
2. 持仓组合平均收益 > 0；
3. 累计超额收益 > 0；
4. 预测上涨概率与实际涨跌的相关系数 > 0；
5. 相比当前静态 Top-N baseline，方向准确率和累计超额同时提升。

若第一版未达到，需回到 **13.8 因子有效性前置诊断**，确认是否存在可被模型学习的信号。
