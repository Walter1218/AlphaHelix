# AlphaHelix 架构设计

> 本文档描述总体架构与模块职责。更完整的 7 层系统蓝图参见 [system-design.md](system-design.md)。

## 1. 项目定位

AlphaHelix 是构建在 HelixAgent 之上的 A 股智能选股智能体。它由数据层、因子/策略层、风控层、Agent 执行层、评估层、Feedback Harness 层和自动化运维层组成，结合 Tushare 金融数据库与 LLM 推理能力，目标是持续提升未来一个月股价走势预测的准确率。

## 2. 核心设计原则

### 2.1 LLM 负责「推理」，Python 负责「计算」

- 因子计算、回测、数据清洗等数值密集型工作由 `scripts/screen.py` 完成
- LLM 负责解释因子含义、融合定性信息（新闻、行业、宏观）、生成投资逻辑
- 避免让 LLM 直接做大量数字运算，减少 token 消耗和数值幻觉

### 2.2 记忆即经验

每次选股结果写入 `memory/stock/YYYY-MM-DD.md`。后续选股时通过 HelixAgent 的 `memory_search` 检索相似市场环境下的历史记录，形成可积累的投资经验。

> 当前 `memory_search` 因 HelixAgent 环境问题暂时禁用，待修复后重新接入。

### 2.3 预测可验证

每只推荐股票必须包含：

- `score`：综合得分
- `rationale`：推荐理由
- `confidence`：置信度
- `stop_loss`：止损价

持有期结束后通过 `evaluate.py` 对比实际走势，生成命中率、超额收益、最大回撤等指标。

### 2.4 Harness 反馈驱动进化

在传统记忆-评估闭环之上，新增 **Feedback Harness 层**：

```
选股执行 → Memory 写入 → 持有期评估 → Feedback Harness
                                          ↓
                    ┌─────────────────────┼─────────────────────┐
                    ↓                     ↓                     ↓
              因子权重更新          策略配置权重          prompt 自适应提示
                    ↓                     ↓                     ↓
              screen.py 加载      strategy 选择参考      alpha-analyst 读取
                    └─────────────────────┴─────────────────────┘
                                          ↓
                                    下次执行
```

## 3. 模块职责

| 模块 | 文件 | 职责 |
|---|---|---|
| Agent 定义 | `.opencode/agent/alpha-analyst.md` | 定义选股 agent 的角色、工作流、输出格式、工具白名单 |
| 数据工具 | `.opencode/tool/tushare_*.ts`、`screen_candidates.ts`、`evaluate_picks.ts` | 单文件单工具，使用 `@opencode-ai/plugin/tool` 封装为 LLM 可调用的工具 |
| 领域知识 | `.opencode/skills/tushare-stock/SKILL.md` | 注入 tushare 接口文档、选股 SOP、评估标准 |
| 因子初筛 | `scripts/screen.py` | 本地计算动量、估值、质量、流动性、资金因子；支持多策略与动态权重 |
| 市场状态 | `scripts/market_regime.py` | 基于沪深300 判断 `trend_up/range/trend_down/high_vol` |
| 每日选股 | `scripts/daily-screen.ts` | 定时调用 HelixAgent 完成选股流程 |
| 回测评估 | `scripts/evaluate-picks.ts` + `scripts/evaluate.py` | 持有期后确定性计算选股结果的实际表现 |
| Walk-forward | `scripts/walkforward.py` | 多期自动选股 + 评估，输出月度/累计指标 |
| Feedback Harness | `scripts/factor_ic.py`、`scripts/strategy_tracker.py`、`scripts/weight_optimizer.py`、`scripts/feedback_harness.py` | 计算 factor IC、跟踪策略表现、优化权重、生成 prompt 自适应提示 |
| 记忆存储 | `memory/stock/*.md` + `memory/stock/*.json` | 存储选股报告与回测快照 |
| 权重存储 | `memory/weights/*_latest.json` | 动态因子权重 |
| Prompt 自适应 | `memory/prompt_adaptations/latest.md` | 基于近期表现的风险/风格提示 |

## 4. 数据流

```
┌─────────────────┐
│  cron / manual  │
│  daily-screen   │
└────────┬────────┘
         ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         HelixAgent Server                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐     │
│  │ alpha-      │  │ read        │  │ screen_                 │     │
│  │ analyst     │→ │ prompt_     │→ │ candidates              │     │
│  │ agent       │  │ adaptations │  │ (strategy=regime)       │     │
│  └──────┬──────┘  └─────────────┘  └───────────┬─────────────┘     │
│         │                                       │                   │
│         │         ┌─────────────┐  ┌─────────────┐                │
│         │         │ tushare_*   │  │ market_     │                │
│         │         │ tools       │  │ regime      │                │
│         │         └─────────────┘  └─────────────┘                │
│         │                                       │                   │
│         └───────────────────────────────────────┘                   │
│                              ↓                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐     │
│  │ memory_     │  │ LLM         │  │ Cardinal / prompt       │     │
│  │ search      │→ │ reasoning   │→ │ risk gate               │     │
│  │ (disabled)  │  │             │  │                         │     │
│  └─────────────┘  └──────┬──────┘  └─────────────────────────┘     │
│                          ↓                                          │
│                   JSON output + write to memory                     │
│              (.md report + .json snapshot)                          │
└─────────────────────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────┐
│ memory/stock/            │
│ YYYY-MM-DD.md + .json    │
└──────────────────────────┘
         ↓
   10/20 trading days later
         ↓
┌──────────────────────────┐     ┌──────────────────────────────┐
│ evaluate.py              │────▶│ Feedback Harness             │
│ deterministic metrics    │     │ factor IC / strategy tracker │
└──────────────────────────┘     │ / weight optimizer           │
                                 └──────────────┬───────────────┘
                                                ↓
                           ┌────────────────────┼────────────────────┐
                           ↓                    ↓                    ↓
              memory/weights/*_latest.json  memory/strategy_tracker  memory/prompt_adaptations/latest.md
                           ↓                    ↓                    ↓
              screen.py 加载动态权重      策略配置参考          alpha-analyst 读取
```

## 5. 关键工具说明

### 5.1 tushare_stock_basic

获取全市场上市股票基础信息，用于初筛剔除 ST、退市、次新股。

### 5.2 tushare_daily

获取个股日线行情，用于 LLM 分析价格走势、支撑压力。

### 5.3 tushare_daily_basic

获取每日估值指标（PE、PB、换手率、市值）。

### 5.4 tushare_fina_indicator

获取季度财务指标（ROE、毛利率、营收增速）。

### 5.5 tushare_moneyflow

获取个股资金流向，识别主力资金动向。

### 5.6 screen_candidates

由 `.opencode/tool/screen_candidates.ts` 提供，调用本地 `scripts/screen.py` 做因子初筛，返回 Top-N 候选池。

> 默认策略为 `regime`，会自动按市场状态在 `momentum_value_hybrid`、`quality_growth`、`contrarian` 之间切换。`screen.py` 会自动加载 `memory/weights/{strategy}_latest.json` 中的动态权重。

### 5.7 evaluate_picks / evaluate.py

由 `.opencode/tool/evaluate_picks.ts` 提供，调用本地 `scripts/evaluate.py`，读取 `memory/stock/YYYYMMDD.json` 快照，确定性计算实际收益、超额收益、最大回撤等指标。

### 5.8 feedback_harness

由 `scripts/feedback_harness.py` 提供，在 walk-forward 或每日评估后运行：

1. 计算每个因子的 rank IC
2. 跟踪各策略滚动收益与命中率
3. 优化因子权重并写入 `memory/weights/`
4. 生成 prompt 自适应提示并写入 `memory/prompt_adaptations/latest.md`

## 6. 风控设计

通过 Cardinal 机制与本地过滤扩展以下规则：

- `stock-st-block`：拦截推荐 ST/*ST/退市 股票的输出（基于 `namechange` 历史名称）
- `stock-liquidity-block`：拦截日均成交额 < 5000 万的标的
- `stock-concentration-warn`：单一行业入选数量超过 top_n 的 40% 时警告
- `stock-stop-loss-missing`：缺少止损价时暂停
- `stock-ann-date-check`：财报数据必须满足 `ann_date <= trade_date`
- `stock-drawdown-alert`：近期最大回撤过大时，prompt 自适应提示降低仓位

## 7. 评估指标

| 指标 | 说明 |
|---|---|
| 方向准确率 | 推荐股票中持有期收益为正的占比 |
| Top3 命中率 | Top3 股票中收益为正的占比 |
| 相对超额收益 | 组合平均收益 - 沪深300同期收益 |
| 最大回撤 | 持有期内组合最大回撤 |
| 置信度相关性 | LLM 置信度与实际收益的相关性 |
| 因子 IC | 因子排名与未来收益排名的 Spearman 相关系数 |
| 策略滚动收益 | 各策略在最近 N 期的加权超额收益 |

## 8. 运行方式

### 8.1 手动执行选股

通过 HelixAgent CLI 以 headless 模式启动：

```bash
cd /Users/onetwo/Documents/trae_projects/AlphaHelix
export MIMOCODE_HOME=/path/to/HelixAgent/.mimo
export TUSHARE_TOKEN=...
bun /path/to/HelixAgent/packages/opencode/src/index.ts run \
  --agent alpha-analyst \
  --format json \
  --title "AlphaHelix 20260702" \
  "执行今日 A 股选股流程"
```

> 必须加 `--format json`，否则 CLI 默认进入交互式 TUI，在无 TTY shell 中会无输出空转。

### 8.2 自动化调度

MVP 阶段使用 `cron` 调用 `scripts/daily-screen.ts`：

```bash
# 交易日 15:30 选股
30 15 * * 1-5 cd /Users/onetwo/Documents/trae_projects/AlphaHelix && bun run scripts/daily-screen.ts

# 每月第一个交易日 09:00 更新 feedback harness
0 9 1 * * cd /Users/onetwo/Documents/trae_projects/AlphaHelix && python scripts/feedback_harness.py --auto
```

> `--auto` 参数待实现：让 harness 自动发现最新可用的回测日期并增量更新。

### 8.3 回测与反馈

```bash
# 运行 walk-forward 回测
python scripts/walkforward.py \
  --start 20250101 --end 20250531 \
  --strategy regime --horizon 10 --top-n 10 --universe-size 200

# 基于回测结果更新权重与 prompt 提示
python scripts/feedback_harness.py \
  --dates 20250127,20250228,20250331,20250430,20250530 \
  --start 20250101 --end 20250531 --horizon 10 --strategy regime
```

## 9. 演进路线

### Phase 1：MVP（已完成）
- tushare 工具可用
- alpha-analyst agent 可执行选股
- 结果写入 memory
- 2026-07-03 首次跑通

### Phase 2：因子、策略与 Regime（已完成）
- `screen.py` 因子库完善（质量、成长、资金、估值）
- 多策略 ensemble：`momentum_value_hybrid`、`quality_growth`、`contrarian`
- `market_regime.py` 判断市场状态，`regime` 自动切换

### Phase 3：风控与记忆（部分完成）
- Cardinal 规则落地（ST/退市、流动性、行业集中度、止损、财报防穿越）
- Memory RAG 启用（待 HelixAgent `memory_search` 修复）

### Phase 4：评估与回测（已完成）
- `evaluate.py` 确定性评估
- `walkforward.py` 多期回测

### Phase 5：自动化调度（下一步）
- cron 定时选股
- cron 定时运行 Feedback Harness
- 日志与告警

### Phase 6：Feedback Harness 在线进化（进行中）
- factor IC 计算
- strategy tracker
- weight optimizer
- prompt 自适应提示
- 在线学习（自动增量更新，待实现）

### Phase 7：高级数据与模型实验（未来）
- 新闻 sentiment、机构持仓、北向资金等数据源
- DPO 数据集导出与模型微调
- 机器学习因子组合

详见 [docs/roadmap.md](docs/roadmap.md) 与 [docs/improvement-plan.md](docs/improvement-plan.md)。

## 10. 当前限制

| 限制 | 说明 | 计划解决阶段 |
|---|---|---|
| `memory_search` 不可用 | HelixAgent 调用该工具触发 `Unexpected server error` | Phase 3（待 HelixAgent 修复） |
| 行业市值权重控制未落地 | 当前仅做数量控制，未按市值权重截断 | Phase 3 |
| 自动化调度未配置 | daily-screen 与 feedback harness 仍需手动触发 | Phase 5 |
| 在线学习未实现 | feedback harness 需手动传 `--dates` | Phase 6 |
| `quality_growth` 偏弱 | 回测中表现落后于 momentum | Phase 6 |
| 缺少交易成本 | 回测未扣除印花税、佣金、滑点 | Phase 4/6 |

## 11. 责任声明

AlphaHelix 对研究方法、数据来源、回测过程与因子逻辑的严谨性负责，并通过 walk-forward 与 Feedback Harness 持续优化模型。但证券市场受宏观环境、政策变化、市场情绪等不可控因素影响，模型输出不代表对未来收益的保证。使用者应结合自身判断审慎决策，过往表现不代表未来收益。
