# AlphaHelix 智能体设计

## 1. 设计原则

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

违反上述纪律的选股结果不得输出。

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
write 工具写入 memory/stock/YYYY-MM-DD.md + .json

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
