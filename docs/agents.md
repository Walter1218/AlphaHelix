# AlphaHelix 智能体设计

## 1. 设计原则

- **胜率优先**：选股优化的首要目标是**胜率**（excess return > 0 的比例），其次才是累计超额收益。高胜率意味着模型稳定可靠，适合实盘。
- **单一主控**：MVP 阶段使用一个 `alpha-analyst` agent 完成端到端选股，降低多 agent 协作失败风险。
- **工具驱动**：所有数据获取、因子计算、文件写入都通过 LLM 工具调用完成，保证过程可审计。
- **可审计**：每个工具调用的输入输出都进入 Trace。
- **可进化**：选股结果通过后续 `evaluate.py` 评估，反哺 prompt 与因子权重优化。

## 2. Agent 配置

### 2.1 文件位置

HelixAgent 扫描 `.opencode/agent/`（单数）目录加载 agent 定义：

```
.opencode/agent/alpha-analyst.md
```

> 早期放在 `.opencode/agents/` 会导致配置不被识别。

### 2.2 最小可运行配置示例

```yaml
---
name: alpha-analyst
mode: primary
model: kimi-for-coding/k2p7
tools:
  "*": false
  bash: true
  read: true
  write: true
  memory: true
  tushare_stock_basic: true
  tushare_daily: true
  tushare_daily_basic: true
  tushare_fina_indicator: true
  tushare_moneyflow: true
  tushare_index_daily: true
  tushare_trade_cal: true
  screen_candidates: true
  evaluate_picks: true
permission:
  bash: allow
  read: allow
  write: allow
  memory: allow
---
```

要点：
- `mode: primary`：让 alpha-analyst 成为主控 agent。
- `tools` 白名单：必须显式列出工具 ID，否则 LLM 不会调用。
- `permission`：声明允许的操作类型。

### 2.3 工具命名规则

`.opencode/tool/` 下单文件单工具，文件名即工具 ID：

| 文件 | 工具 ID |
|---|---|
| `tushare_stock_basic.ts` | `tushare_stock_basic` |
| `screen_candidates.ts` | `screen_candidates` |
| `evaluate_picks.ts` | `evaluate_picks` |

每个文件必须 `export default tool({...})`（来自 `@opencode-ai/plugin`）。

### 2.4 必须遵守的约束与纪律

完整约束清单见 [docs/risk.md](../risk.md) 第 0 节。agent 在执行选股流程时必须重点遵守以下纪律：

| 编号 | 纪律 | agent 侧要求 |
|---|---|---|
| C01 | 无未来函数 | 不得使用 `trade_date` 之后的价格、财报、新闻做任何推理 |
| C08 | LLM 不做数值计算 | 收益率、PE、动量等数值必须调用工具获取，禁止心算或推断 |
| C09 | 评估必须确定性 | 不要自行判断选股好坏，所有绩效指标由 `evaluate.py` 产出 |
| C21 | 必须包含止损价 | 每只股票输出必须带 `stop_loss` 字段 |
| C28 | CLI 用 headless 模式 | 外部调用时必须加 `--format json` |
| C31 | 双写产物 | 每次选股必须同时写入 `.md` 和 `.json` |
| C35 | 责任声明 | 最终报告必须说明「我们对研究方法和数据质量负责，但不承诺收益，市场存在不确定性」 |
| C38 | 任何策略决策必须 walk-forward / 样本外验证 | 凡影响选股决策的模型、因子、权重、阈值、参数、特征工程，只能用决策点之前的数据生成；禁止在全样本上调优后回测同一区间 |

违反上述纪律的选股结果不得输出。

### 2.5 回测与策略决策的红线

> 本项纪律是 C01「无未来函数」的**泛化版本**：不仅选股时不能看未来，**用来生成选股规则的所有决策过程**也不能看未来。

#### 核心原则

- **决策点原则**：一个决策在何时被使用，就只能使用在该时刻**已经公开可得**的数据。
- **训练/测试不重叠**：任何调参、选特征、定权重、定阈值的「训练集」结束时间，必须早于「测试集」开始时间。
- **全样本调参即穿越**：只要用到了测试区间内的数据来影响决策，就是时间穿越，无论最终选股公式本身是否只用到历史数据。

#### 禁止行为

1. **全样本优化后回测同一区间**：
   - 包括但不限于：全样本 IC 算权重、全样本网格搜索 `learning_rate`/`lookback`/`min_score`、全样本挑选最优因子、全样本决定 regime 映射。
   - 错误示例：用 2024-01 到 2026-06 全部数据计算因子 IC 生成权重，再回测 2024-01 到 2026-06。

2. **用测试集反馈迭代策略后重复测试同一区间**：
   - 错误示例：发现 2025 年某个因子有效，把它加入策略，再跑一遍 2024-2026 回测看效果。

3. **把样本内优化结果当作有效绩效**：
   - 任何基于全样本调参后的回测结果，只能作为「诊断/上限参考」，必须明确标注，且不能写入 `memory/eval/` 作为正式结果，不能用于生产。

#### 允许做法

1. **Walk-forward**：
   - 每期用该期之前的数据生成规则/参数，再用于当期。
   - 例如 `walkforward.py --ic-calibrate`、滚动窗口更新。

2. **样本外验证**：
   - 用 2024 年训练/调参，用 2025-2026 年测试；训练集与测试集不重叠，且测试集只做一次最终评估。

3. **诊断性分析**：
    - 可以计算全样本统计量来发现问题，但必须在代码、文档、输出中明确标注「样本内诊断，非有效回测/非生产用」。

### 2.6 数据处理防穿越规范

> 本项纪律是 C01「无未来函数」的**技术实现规范**：所有特征工程操作必须严格按日期分组，禁止使用跨时间信息。

#### 核心原则

- **截面操作原则**：任何特征工程操作（rank、winsorize、neutralize、discretize）必须按日期分组执行，只能使用当天数据。
- **时间序列操作原则**：滚动窗口（rolling/shift/ewm）必须使用历史数据，禁止使用未来数据。
- **全局统计禁止**：禁止使用全样本均值、中位数、分位数等统计量来处理单日数据。

#### 必须遵守的规范

| 操作 | 正确做法 | 错误做法 | 说明 |
|---|---|---|---|
| Rank 标准化 | `df.groupby("date")[col].rank(pct=True)` | `df[col].rank(pct=True)` | 全局 rank 会泄露未来数据 |
| Winsorize 截尾 | `df.groupby("date")[col].transform(clip_fn)` | `df[col].quantile()` | 全局分位数会泄露未来数据 |
| 中性化 | 按日期分组做回归 | 全截面回归 | 全截面回归会泄露未来数据 |
| 离散化 | `df.groupby("date")[col].transform(lambda x: pd.qcut(...))` | `pd.qcut(df[col])` | 全局分箱会泄露未来数据 |
| Regime 特征 | `rolling(60).median()` | `.median()` | 全局 median 会泄露未来数据 |
| 行业内 rank | `df.groupby(["date", "industry"])[col].rank()` | `df[col].rank()` | 全局 rank 会泄露未来数据 |

#### 代码审查清单

每次修改 `feature_engineering.py` 或 `add_features.py` 时，必须检查：

1. [ ] 所有 `rank()` 调用是否按日期分组？
2. [ ] 所有 `quantile()` 调用是否按日期分组？
3. [ ] 所有 `rolling()` 调用是否只使用历史窗口？
4. [ ] 所有 `median()`/`mean()` 调用是否按日期分组或使用滚动窗口？
5. [ ] 所有回归中性化是否按日期分组？
6. [ ] 新增特征是否可能引入未来数据？

#### 已知数据穿越问题（已修复）

| 问题 | 影响 | 修复方案 |
|---|---|---|
| `rank_features` 全局 rank | 虚假高胜率（61.9% → 真实57%） | 改为 `df.groupby("date")[col].rank()` |
| `winsorize_features` 全局分位 | 特征分布失真 | 改为按日期分组截尾 |
| `neutralize_features` 全截面回归 | 中性化系数泄露未来 | 改为按日期分组回归 |
| `vol_regime` 全局 median | Regime 判断泄露未来 | 改为 `rolling(60).median()` |
| 模型选择用全量数据 | 虚假高IC（0.082 → 真实0.047） | 改为 walk-forward 选择 |
| 数据集构建时机 | 旧数据集导致模型泄露 | 修复后必须重建数据集 |

### 2.7 数据泄露案例库

> 本节记录实际发生的数据泄露案例，供后续开发参考。

#### 案例1：全局 Rank 导致虚假高胜率

**现象**：修复前胜率 61.9%，修复后 57.1%

**原因**：`rank_features` 使用 `df[col].rank(pct=True)` 对所有日期一起 rank

**泄露机制**：
```
日期 T 的 rank = f(日期 T 的值, 日期 T+1 的值, 日期 T+2 的值, ...)
```
计算 T 日的 rank 时，使用了 T+1、T+2 等未来日期的数据。

**修复**：改为 `df.groupby("date")[col].rank(pct=True)`

**教训**：rank/winsorize/neutralize 等截面操作必须按日期分组。

#### 案例2：全局 Median 导致 Regime 判断泄露

**现象**：vol_regime 使用全局 median 判断市场状态

**泄露机制**：
```
vol_regime(T) = market_vol(T) > median(all market_vol)
```
使用了所有日期（包括未来）的 median。

**修复**：改为 `rolling(60).median()`，只用历史数据。

**教训**：任何统计量（median/mean/quantile）都必须用滚动窗口或按日期分组。

#### 案例3：模型选择用全量数据

**现象**：Regime 自适应模型 IC 从 0.082 降到 0.047

**原因**：`precompute_regime_performance` 用全量数据计算每个 regime 的最佳模型

**泄露机制**：
```
best_model(regime) = argmax(model IC on ALL data in regime)
```
选择了在全量数据上表现最好的模型，包括未来数据。

**修复**：改为 walk-forward 选择，只用历史数据。

**教训**：模型选择、超参数调优、特征选择都必须用历史数据。

#### 案例4：数据集构建时机

**现象**：original_30 模型在修复前的数据集上训练，导致泄露

**原因**：`features_h10_composite.parquet` 在修复前构建（2026-07-05），修复后未重建

**泄露机制**：使用旧数据集训练的模型，其特征已经是泄露的。

**修复**：修复特征工程后，必须重建所有相关数据集。

**教训**：修改特征工程后，必须检查所有依赖该特征工程的数据集是否需要重建。

### 2.8 数据泄露检查清单

> 每次进行实验前，必须完成以下检查：

1. [ ] **特征工程**：所有 rank/winsorize/neutralize 是否按日期分组？
2. [ ] **数据集构建**：数据集是否在特征工程修复后重建？
3. [ ] **模型选择**：是否使用 walk-forward 而非全量数据？
4. [ ] **超参数调优**：是否使用历史数据而非全量数据？
5. [ ] **特征选择**：是否使用历史数据而非全量数据？
6. [ ] **Regime 检测**：是否使用滚动窗口而非全局统计？
7. [ ] **回测结果**：是否与之前有显著差异？（差异过大可能意味着泄露）

## 3. alpha-analyst 职责与能力

### 3.1 数据获取

**职责**：从 Tushare 安全、高效地获取原始数据。

**调用工具**：
- `tushare_trade_cal`：确认最新交易日
- `tushare_index_daily`：获取沪深300等指数日线
- `tushare_stock_basic`：获取全市场股票基础信息
- `tushare_daily`：个股日线行情
- `tushare_daily_basic`：每日估值指标
- `tushare_fina_indicator`：季度财务指标
- `tushare_moneyflow`：个股资金流向

**约束**：
- 严禁使用未来数据
- 调用失败时记录原因，不阻塞主流程
- 高频数据做本地缓存

### 3.2 因子初筛

**职责**：基于本地 Python 脚本计算量化因子，输出候选池。

**调用工具**：
- `screen_candidates`（推荐）
- 或 `bash` 直接调用 `python scripts/screen.py <strategy> <date> <top_n>`

**核心因子**：

| 因子 | 计算方式 | 权重 |
|---|---|---|
| 20日动量 | (close_t / close_t-20) - 1 | 25% |
| 60日动量 | (close_t / close_t-60) - 1 | 15% |
| 估值 | 1/PE + 1/PB 综合排名 | 30% |
| 质量 | 总市值排名（规模因子） | 20% |
| 流动性 | 近20日成交额排名 | 10% |

**输出**：Top-N 候选股票列表，含因子原始值。

### 3.3 定性研究

**职责**：结合行业、资金流向做定性分析。

**调用工具**：
- `tushare_moneyflow`
- `memory`（检索历史相似环境，**当前因 HelixAgent `Unexpected server error` 暂时禁用**）
- `webfetch`（可选，抓取公开研报、行业新闻）

**分析维度**：
- 行业景气度与政策导向
- 主力资金动向
- 历史相似市场环境下该股/该行业表现

**输出**：每只股票的一段话定性评价。

### 3.4 选股决策

**职责**：综合因子打分与定性研究，输出最终投资组合。

**输出格式**：

```json
{
  "date": "20260703",
  "data_date": "20260702",
  "market_summary": "...",
  "picks": [
    {
      "ts_code": "600519.SH",
      "name": "贵州茅台",
      "score": 0.92,
      "rank": 1,
      "rationale": "...",
      "confidence": "high",
      "stop_loss": 1480.0
    }
  ],
  "risk_notes": ["..."]
}
```

**约束**：
- 必须说明推荐理由
- 必须给出置信度和止损价
- 必须给出风险提示

### 3.5 风控过滤

通过 agent 指令与本地脚本双重过滤：

- 剔除 ST/*ST/退市股（`screen.py` 已按历史名称过滤）
- 剔除日均成交额 < 5000 万的标的
- 单一行业集中度不超过 40%（agent 定性检查并提示）
- 必须包含止损价
- 避免高波动/高杠杆叙事

### 3.6 未来函数禁忌与数据防穿越

**原则**：T 日选股只能用 T 日及之前已公开的数据。任何使用 T 日之后信息的行为都会让回测失真，是 AlphaHelix 的红线。

**已实现措施**（`scripts/screen.py` + `_tushare_utils.py`）：

| 数据类型 | 防穿越规则 | 实现位置 |
|---|---|---|
| 价格数据 | 日线只取 `start_date` 到 `trade_date`；买入价用 `trade_date` 收盘价 | `screen.py`, `evaluate.py` |
| 财报数据 | `fina_indicator` 必须满足 `ann_date <= trade_date`；按最近已公告报告期取数 | `screen.py:fetch_fina_factors` |
| ST/*ST/退市 | 用 `namechange` 接口查历史名称，不用当前名字判断历史状态 | `_tushare_utils.py:is_st_historical` |
| 退市/停牌 | 通过 `daily` 数据判断 `trade_date` 当天是否有交易记录 | `screen.py:pass1_screen` |
| 估值/资金流 | `daily_basic` 和 `moneyflow` 只取截至 `trade_date` | `screen.py` |
| 行业分类 | 仅用于报告展示，不做基于当前行业的量化截断 | `screen.py:build_universe` |

**已知限制**：
- `stock_basic` 的 `industry` 字段为当前分类，历史回测中若股票行业发生过变更，报告中的行业分布可能与历史真实分布存在偏差。
- 行业集中度目前由 agent 在生成报告时定性提示，而非脚本自动截断，以确保严格回测不依赖可能过时的行业数据。

**禁止行为**：

- 用 T+1 及之后的价格评估 T 日选股
- 用未公告的财报做 T 日决策
- 用当前 ST/退市状态过滤历史股票池
- 用未来才能知道的宏观事件、政策、新闻做 T 日选股

**回测规范**：

- 入场价格必须是 `trade_date` 收盘价或开盘价（需在代码中明确）
- 出场价格必须是 `exit_date` 收盘价
- 每次因子/策略改动后，必须重新跑完整 walk-forward 回测
- 保留 out-of-sample 区间用于验证

> 完整约束清单（C01-C37）见 [docs/risk.md](../risk.md) 第 0 节。

### 3.7 离线评估

**职责**：选股后 1 个月评估实际表现。

**调用工具/脚本**：
- `evaluate_picks`（HelixAgent 工具）
- `scripts/evaluate.py`（确定性收益计算）
- `tushare_daily` / `tushare_index_daily`

**输出**：命中率、超额收益、最大回撤、置信度相关性，并将结果追加到 `memory/stock/YYYYMMDD.md`。

## 4. 执行流程

```
alpha-analyst 接收用户指令
    ↓
调用 tushare_trade_cal 确认最新交易日
    ↓
调用 tushare_index_daily 获取市场基准
    ↓
调用 tushare_stock_basic 获取股票池
    ↓
调用 screen_candidates 做因子初筛（Top 50）
    ↓
对 Top 候选调用 tushare_daily / tushare_daily_basic / tushare_fina_indicator / tushare_moneyflow
    ↓
LLM 综合打分并生成 Top-K 组合
    ↓
write 工具写入 memory/stock/YYYYMMDD.md + .json

> 注：当前 MVP 流程跳过 `memory_search`，待 HelixAgent 修复后再加入历史经验复用步骤。
    ↓
检查约束清单（C01-C37）：确认无未来函数、止损价完整、含责任声明
    ↓
20 交易日后 evaluate.py 自动评估
```

## 5. 与 HelixAgent Mode 的映射

| AlphaHelix 模式 | Helix Mode | 用途 |
|---|---|---|
| 单次选股 | `primary` | 当前 MVP 主流程 |
| 多策略对比 | `compose` / `max` | 未来生成多个候选策略并评分 |
| 数据查询 | `ask` | 仅查询数据，不做推荐 |

## 6. 注意事项

### 6.1 敏感信息

- **禁止在代码、文档或提交记录中硬编码 `TUSHARE_TOKEN`**。token 必须通过 `.env` 注入。
- 仓库中的 `.env.example` 仅含占位符，复制为 `.env` 后再填入真实 token。
- 不要提交 `memory/` 下的选股报告、回测结果、权重文件、日志等运行时数据；这些目录已通过 `.gitignore` 排除。

### 6.2 路径占位符

- 文档中涉及的本地路径（如 `/path/to/AlphaHelix`、`<path-to-HelixAgent>`）均为占位符，部署或运行时请替换为实际绝对路径。
- cron 示例中的 `/usr/local/bin/bun` 和 `/usr/local/bin/python3` 也可能因环境不同而变化，请用 `which bun` / `which python3` 确认。

### 6.3 `memory/` 目录

- `memory/stock/`、`memory/eval/`、`memory/weights/`、`memory/prompt_adaptations/` 等均为运行时产物。
- 首次克隆后这些目录为空，执行 `daily-screen.ts`、`walkforward.py` 或 `feedback_harness.py` 后会自动生成。
- 若需备份历史选股记录，请单独备份 `memory/`；仓库本身不保存这些文件。

### 6.4 远程仓库

- 代码已推送至 `https://github.com/Walter1218/AlphaHelix`。
- 后续提交前请再次检查：
  1. `git diff --check` 无空白错误；
  2. 无新增的 token、密码或个人本地路径；
  3. `memory/`、`.cache/`、`*.log` 等未出现在 `git status` 的待提交列表中。
