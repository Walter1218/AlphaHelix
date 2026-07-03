# AlphaHelix

> One helix, infinite alpha.

基于 HelixAgent 底层 AI 能力的 A 股智能选股智能体。通过 LLM 推理 + Tushare 数据 + 记忆进化，持续优化未来一个月股价走势预测准确率。

> **状态更新（2026-07-03）**：MVP 端到端已稳定跑通；`screen.py` 从 5 个因子扩展到 18+ 因子（含事件、反转、行业相对强度）；已实现 `momentum_value_hybrid`、`quality_growth`、`contrarian`、`event_driven` 四策略及基于沪深300 的 `regime` 自动切换；Feedback Harness v1 落地，可自动更新因子权重与 prompt 自适应提示；已完成 8 个月 walk-forward 回测（2025-01~05、2026-04~06），`regime` 策略累计超额 +13.82%，方向准确率 60%+。`memory_search` 因 HelixAgent 环境问题暂时禁用。

## 核心目标

- **输入**：市场量价、财务、资金、行业等多维度信息
- **输出**：未来一个月预期表现最优的 Top-K 股票组合
- **优化目标**：组合相对沪深 300 的超额收益（Alpha）与方向命中率
- **进化机制**：每次预测写入记忆，定期通过 Feedback Harness 评估并更新因子权重与 prompt 风格

## 架构

```
User / Cron
    ↓
HelixAgent (agent=alpha-analyst, mode=primary)
    ↓
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ tushare_*       │  │ screen.py       │  │ evaluate.py     │
│ 数据工具        │  │ 因子初筛        │  │ 收益评估        │
└─────────────────┘  └─────────────────┘  └─────────────────┘
    ↓
LLM 综合推理 → JSON 输出
    ↓
write → memory/stock/YYYYMMDD.md + .json
    ↓
Feedback Harness（factor_ic / strategy_tracker / weight_optimizer）
    ↓
memory/weights/*_latest.json + memory/prompt_adaptations/latest.md
    ↓
下次选股自动加载新权重与提示
```

## 目录结构

```
AlphaHelix/
├── .opencode/
│   ├── agent/alpha-analyst.md        # 选股 agent 定义（HelixAgent 扫描单数目录）
│   ├── tool/tushare_*.ts             # Tushare API 单文件单工具
│   ├── tool/screen_candidates.ts     # 本地因子初筛工具
│   ├── tool/evaluate_picks.ts        # 本地收益评估工具
│   └── skills/tushare-stock/SKILL.md # 选股 SOP 与接口文档
├── scripts/
│   ├── daily-screen.ts              # 每日选股调度入口
│   ├── evaluate-picks.ts            # 历史评估入口
│   ├── screen.py                    # 本地因子计算脚本（支持多策略与动态权重）
│   ├── evaluate.py                  # 确定性收益评估脚本
│   ├── walkforward.py               # 多期 walk-forward 回测
│   ├── market_regime.py             # 基于沪深300 的市场状态分类
│   ├── _tushare_utils.py            # Tushare 共享工具（缓存、限流、交易日历、历史 ST 判断）
│   ├── factor_ic.py                 # 因子秩相关系数（IC）计算
│   ├── strategy_tracker.py          # 策略滚动绩效跟踪
│   ├── weight_optimizer.py          # 因子权重优化
│   └── feedback_harness.py          # Feedback Harness 编排器
├── docs/
│   ├── system-design.md             # 7 层系统蓝图（推荐先看）
│   ├── architecture.md              # 总体架构与模块职责
│   ├── research.md                  # HelixAgent 能力调研与方案对比
│   ├── agents.md                    # agent 设计与工具白名单
│   ├── evolution.md                 # 进化闭环设计
│   ├── roadmap.md                   # Phase 1-7 落地路线图
│   ├── risk.md                      # 风险、边界与约束目录（C01-C37）
│   ├── decisions.md                 # 关键决策记录（ADR）
│   ├── improvement-plan.md          # 数据、因子、策略、风控改进路线图
│   └── operations.md                # 运维手册（cron、日志、排错）
├── memory/
│   ├── stock/                       # 选股报告与快照
│   ├── eval/                        # 评估与回测结果
│   ├── weights/                     # 动态因子权重
│   ├── prompt_adaptations/          # prompt 自适应提示
│   └── log/                         # 运行日志
├── .env.example                     # 环境变量模板
├── package.json
└── README.md
```

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/system-design.md](docs/system-design.md) | 7 层系统蓝图，当前最权威的总体设计 |
| [docs/architecture.md](docs/architecture.md) | 总体架构与模块职责 |
| [docs/research.md](docs/research.md) | HelixAgent 能力调研与方案对比 |
| [docs/agents.md](docs/agents.md) | 多智能体分工与协作流程 |
| [docs/evolution.md](docs/evolution.md) | 以准确率为目标的进化闭环 |
| [docs/roadmap.md](docs/roadmap.md) | Phase 1-7 落地路线图 |
| [docs/risk.md](docs/risk.md) | 数据/模型/策略/合规风险与约束目录（C01-C37） |
| [docs/decisions.md](docs/decisions.md) | 关键决策记录（ADR） |
| [docs/improvement-plan.md](docs/improvement-plan.md) | 数据、因子、策略、风控改进路线图 |
| [docs/operations.md](docs/operations.md) | 运维手册：cron、日志、排错、维护 |

## 快速开始

### 1. 安装依赖

```bash
cd /Users/onetwo/Documents/trae_projects/AlphaHelix
bun install
pip install tushare pandas numpy
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 TUSHARE_TOKEN
```

`.env` 中支持的变量：

| 变量 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `TUSHARE_TOKEN` | 是 | - | Tushare Pro API token |
| `ALPHAHELIX_SERVER_URL` | 否 | `http://127.0.0.1:3096` | HelixAgent HTTP API 地址（当前未使用） |
| `OPENCODE_SERVER_PASSWORD` | 否 | - | HelixAgent server password（未设置则留空） |

### 3. 运行选股（推荐 `daily-screen.ts`）

```bash
# 自动加载 .env，调用 HelixAgent CLI headless 模式，输出写入 memory/stock/
bun run scripts/daily-screen.ts

# 或直接使用 package.json 中的别名
bun run screen
```

> `daily-screen.ts` 会将 HelixAgent 子进程的 stdout/stderr 重定向到 `memory/log/daily-screen-*.log`，避免 pipe 模式下的偶发 `Unexpected server error`。

### 4. 手动运行因子初筛

```bash
# 使用 regime 策略（默认），基于沪深300 自动选择底层策略
python scripts/screen.py regime 20260702 10

# 使用指定策略
python scripts/screen.py momentum_value_hybrid 20260702 10
python scripts/screen.py quality_growth 20260702 10
python scripts/screen.py contrarian 20260702 10
python scripts/screen.py event_driven 20260702 10
```

### 5. 评估历史选股

```bash
# 评估 10 个交易日前的选股
python scripts/evaluate.py 20260602 10

# 多期 walk-forward 回测
python scripts/walkforward.py \
  --start 20250101 --end 20250630 \
  --strategy regime --horizon 10 --top-n 10 --universe-size 200
```

### 6. 运行 Feedback Harness

```bash
# 基于一组历史选股日更新权重与 prompt 自适应提示
python scripts/feedback_harness.py \
  --dates 20250127,20250228,20250331,20250430,20250530,20260430,20260529,20260615 \
  --start 20250101 --end 20260615 --horizon 10 --strategy regime

# 输出：
# memory/weights/momentum_value_hybrid_latest.json
# memory/weights/quality_growth_latest.json
# memory/weights/contrarian_latest.json
# memory/prompt_adaptations/latest.md
```

## 关键设计原则

1. **因子计算本地化**：数值计算走 `scripts/screen.py` 和 `scripts/evaluate.py`，避免 LLM 数值幻觉
2. **严格防时间穿越**：T 日选股只能用 T 日及之前已公开数据；财报校验 `ann_date`、ST 状态查历史名称、退市判断查当天交易记录
3. **推理结构化**：LLM 输出必须包含 JSON：`ts_code`, `score`, `rank`, `rationale`, `confidence`, `stop_loss`
4. **双文件持久化**：每次选股同时写入 `memory/stock/YYYYMMDD.md`（报告）和 `YYYYMMDD.json`（回测快照）
5. **记忆可进化**：Feedback Harness 基于历史结果自动更新因子权重与 prompt 风格
6. **风险可拦截**：Cardinal 规则拦截 ST、退市、低流动性、财报未公告等高风险标的

## 当前状态

- **阶段**：Phase 1~4 已完成，Phase 5（自动化调度）与 Phase 6（Feedback Harness 在线进化）并行推进中。
- **已验证**：
  - 端到端工具链可用，能生成报告与快照。
  - `daily-screen.ts` 可无人值守执行，输出稳定。
  - `scripts/evaluate.py` 可完成历史持有期收益评估。
  - `screen.py` 支持 `momentum_value_hybrid`、`quality_growth`、`contrarian`、`event_driven` 四策略与 `regime` 自动切换。
  - `screen.py` 已覆盖动量、估值、质量、资金、事件、反转、行业相对强度七类因子（18+）。
  - `market_regime.py` 可基于沪深300 判断市场状态。
  - `walkforward.py` 已完成 8 个月回测。
  - Feedback Harness v1 可产出动态权重与 prompt 自适应提示。
- **回测结果（10 日持有期，regime 策略）**：
  - 2025-01~05：月均超额 +1.67%，方向准确率 66%，累计超额 +8.86%（优于单一 momentum 的 +4.92%）。
  - 2026-04~06：累计超额 +4.96%，方向准确率稳定 60%+。
  - 两段样本合计超额约 +13.82%（为分段超额简单相加，非复利）。
  - 详细结果见 `memory/eval/walkforward_regime_*.json`。
- **待完成**：
  - cron 自动化调度（Phase 5）。
  - Feedback Harness `--auto` 在线学习模式（Phase 6）。
  - 分行业命中率反馈与置信度校准。
  - `quality_growth` 策略调优。
- **已知问题**：
  - `memory_search` 在当前 HelixAgent 环境下会触发 `Unexpected server error`，已暂时从选股流程中移除。
  - 行业集中度目前为数量控制，市值权重控制尚未完全实现。

详见 [docs/roadmap.md](docs/roadmap.md)。

## 责任声明

AlphaHelix 对研究方法、数据来源、回测过程与因子逻辑的严谨性负责，并通过 walk-forward 与 Feedback Harness 持续优化模型。但证券市场受宏观环境、政策变化、市场情绪等不可控因素影响，模型输出不代表对未来收益的保证。使用者应结合自身判断审慎决策，过往表现不代表未来收益。
