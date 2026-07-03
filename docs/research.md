# AlphaHelix 调研报告

> 调研目标：基于 HelixAgent 底层 AI 能力，结合 Tushare 数据库，实现智能选股智能体。

## 1. HelixAgent 能力调研

### 1.1 项目结构

调研仓库（本地路径，请替换为实际路径）：
- `<path-to-HelixAgent>`
- `<path-to-Helix>`
- `<path-to-bubu>/`（现有 tushare 脚本参考）

### 1.2 可复用的核心能力

| 能力 | 对应源码 | 在 AlphaHelix 中的用途 |
|---|---|---|
| **多 Agent 模式** | `packages/opencode/src/session/mode-registry.ts` | 定义 `alpha-analyst` agent，mode 用 `compose`/`max` |
| **工具注册表** | `packages/opencode/src/tool/registry.ts` | 注册 `tushare_*`、`screen_candidates`、`evaluate_picks` 工具 |
| **目录扫描工具** | `packages/opencode/src/tool/registry.ts:212` | 工具放 `.opencode/tool/` 自动加载，无需改核心 |
| **Plugin 机制** | `packages/plugin/src/tool.ts` | 未来可拆分为独立 plugin 复用 |
| **Skill 注入** | `packages/opencode/src/skill/index.ts` | 写 `SKILL.md` 把 tushare 接口文档、选股 SOP 注入系统 prompt |
| **BM25 + Vector RAG** | `packages/core/src/memory/service.ts:70-116` | 存储历史选股与结果，FTS 权重 0.6 + Vector 权重 0.4 |
| **Trace** | `packages/opencode/src/trace/trace.ts` | 记录选股决策链，供 DPO 与审计 |
| **Judge** | `packages/opencode/src/session/goal-judge.ts` | 评估选股任务完整性与相关性 |
| **Cardinal** | `packages/opencode/src/session/cardinal.ts` | 实时风控拦截，四级 block/pause/stop/warn |
| **Effect HTTP Client** | `packages/opencode/src/tool/webfetch.ts` | tushare REST 调用走 `effect/unstable/http` |
| **工作流/调度** | `packages/opencode/src/tool/workflow.ts` | 异步执行；定时选股外挂 cron |
| **Memory 工具** | `packages/opencode/src/tool/memory.ts` | 当前仅支持 `search`，写入需用 `write` 工具 |

### 1.3 关键发现

- HelixAgent 无原生 cron/recurrence 引擎，定时选股建议用 cron + HTTP client 调用。
- 工具定义支持目录扫描（`.opencode/tool/*.ts`），是最快落地方式；目录工具建议用原生 JSON Schema 描述参数，避免外部依赖。
- Skill 是 markdown 文件（`SKILL.md`），可注入领域知识与 SOP。
- Memory 混合检索（BM25 + Vector）适合积累选股经验；但内置 `memory` 工具目前仅支持 `search`，写入需用 `write` 工具。
- Max mode 支持 `candidates: 5`，可生成多候选并评分，天然适合 ensemble 选股。

## 2. Tushare 数据维度

### 2.1 可用接口（与选股相关）

| 维度 | 接口 | 关键字段 |
|---|---|---|
| 股票基础 | `stock_basic` | ts_code, name, list_date, list_status |
| 日线行情 | `daily` | trade_date, open, high, low, close, vol, amount |
| 每日指标 | `daily_basic` | pe, pb, turnover_rate, total_mv, circ_mv |
| 财务指标 | `fina_indicator` | roe, grossprofit_margin, revenue_yoy, profit_yoy |
| 资金流向 | `moneyflow` | buy_sm_vol, sell_sm_vol, net_mf_vol |
| 指数行情 | `index_daily` | close, pct_chg（沪深300、上证指数） |
| 行业/概念 | `industry_classified`, `concept_detail`, `index_member` | 板块归属 |
| 新闻/公告 | `news`, `major_news` | 标题、内容、时间 |
| 宏观 | `shibor`, `macro_cn` | 利率、GDP、CPI |

### 2.2 限制

- 高级接口需要积分，免费用户权限有限。
- 有调用频次限制，全量扫描需要本地缓存与限流。
- 数据更新时效：日线 T-1，财务数据按公告日。

### 2.3 现有脚本参考

`<path-to-bubu>/fund_screener.py`（本地参考脚本）已实现：
- DataAgent：拉取 ETF 列表与日线
- AlphaAgent：计算动量、流动性、波动率因子
- RiskAgent：按成交额与波动率过滤

AlphaHelix 在此基础上扩展为股票池，并加入 LLM 推理层。

## 3. 方案对比

### 3.1 方案 A：纯 Python 量化框架

**优点**：数值计算精确、可回测、速度快。  
**缺点**：
- 无法处理定性信息（新闻、行业叙事、宏观事件）
- 因子权重依赖人工或传统机器学习
- 没有经验积累机制

### 3.2 方案 B：裸 LLM + tushare 脚本

**优点**：有推理能力，能理解自然语言。  
**缺点**：
- 没有持久记忆，每次冷启动
- 没有决策质量评估与风险拦截
- 没有多候选比较，易受随机性影响

### 3.3 方案 C：HelixAgent（本方案）

**核心优势**：
- **Memory RAG**：历史选股经验可检索复用
- **Max mode**：多候选 ensemble，降低单次推理方差
- **Cardinal**：实时风控拦截
- **Trace**：完整决策审计链
- **Evolution Flywheel**：DPO 数据驱动持续优化
- **Skill**：结构化领域知识注入
- **Plugin 架构**：无需改动 HelixAgent 核心

## 4. 关键决策

| 决策 | 选项 | 选择 | 理由 |
|---|---|---|---|
| 集成方式 | 改核心 / 插件 / 目录工具 | **目录工具 + Skill** | 最快落地，无需 rebuild HelixAgent |
| 因子计算 | LLM 计算 / Python 计算 | **Python 计算** | 避免 token 爆炸与数值幻觉 |
| 定时调度 | Helix workflow / cron | **cron + HTTP client** | Helix 当前无原生 cron |
| 记忆格式 | 数据库 / markdown | **markdown** | 与 Helix Memory 机制一致，便于 RAG |
| 评估周期 | 1周 / 1月 / 1季 | **1个月（20交易日）** | 与目标一致，减少噪音 |

## 5. 参考资料

- HelixAgent README: `<path-to-HelixAgent>/README.md`
- Helix README: `<path-to-Helix>/README.md`
- Tushare Pro: https://tushare.pro
- Tushare Skills: https://github.com/waditu-tushare/skills
