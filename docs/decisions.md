# AlphaHelix 关键决策记录（ADR）

## ADR-001：项目命名

**日期**：2026-07-02  
**决策**：项目命名为 **AlphaHelix**  
**选项**：AlphaHelix / Finch / HelixSeer / 螺旋金眼  
**理由**：
- Alpha = 超额收益，Helix = 螺旋进化
- α-螺旋是蛋白质基本结构，隐喻「金融世界的基本构建块」
- 与 HelixAgent 品牌一致，保留 Helix 命名空间

---

## ADR-002：集成方式

**日期**：2026-07-02  
**决策**：使用 HelixAgent 的「目录扫描工具 + Skill」方式集成  
**选项**：
1. 修改 HelixAgent 核心代码
2. 开发独立 plugin 包
3. 目录扫描工具 + SKILL.md

**选择**：选项 3  
**理由**：
- 无需 rebuild HelixAgent 核心
- 最快落地 MVP
- 未来可平滑迁移为独立 plugin

---

## ADR-003：因子计算位置

**日期**：2026-07-02  
**决策**：因子计算由本地 Python 脚本完成，LLM 负责推理  
**选项**：
1. LLM 直接计算因子
2. Python 脚本计算因子
3. 混合：简单因子 LLM，复杂因子 Python

**选择**：选项 2  
**理由**：
- 避免 LLM 数值幻觉
- 减少 token 消耗
- 可复用现有量化库（pandas/numpy）

---

## ADR-004：定时调度方案

**日期**：2026-07-02  
**决策**：使用 cron + HTTP client 调用 HelixAgent API  
**选项**：
1. HelixAgent 内置 workflow 工具
2. cron + standalone script
3. 自研调度服务

**选择**：选项 2  
**理由**：
- HelixAgent 当前 workflow 无原生 cron/recurrence
- cron 简单可靠，与系统已有机制兼容

---

## ADR-005：记忆格式

**日期**：2026-07-02  
**决策**：使用 Markdown 文件存储选股记录  
**选项**：
1. SQLite 数据库
2. Markdown 文件
3. 向量数据库

**选择**：选项 2  
**理由**：
- 与 HelixAgent Memory RAG 机制一致
- 便于人工审阅与版本控制
- BM25 + Vector 混合检索已支持

---

## ADR-006：评估周期

**日期**：2026-07-02  
**决策**：以 20 个交易日（约 1 个月）为持有评估周期  
**选项**：1周 / 1月 / 1季度  
**选择**：1月  
**理由**：
- 与项目目标「未来一个月价格走势准确率」一致
- 减少短期噪音
- 样本量适中，便于迭代

---

## ADR-007：目标模型

**日期**：2026-07-02  
**决策**：默认使用 `kimi-for-coding/k2p7` 作为 alpha-analyst 模型  
**选项**：
1. xiaomi/mimo-v2.5-pro
2. kimi-for-coding/k2p7
3. openai/gpt-4o

**选择**：选项 2  
**理由**：
- Kimi K2.7 支持 tool_call 与 reasoning
- 上下文窗口 262k，适合多只股票分析
- 当前 HelixAgent 已配置 kimi provider

---

## ADR-008：初筛策略

**日期**：2026-07-02  
**决策**：MVP 采用 `momentum_value_hybrid` 策略  
**选项**：
1. 纯动量
2. 纯价值
3. 动量+价值+质量混合

**选择**：选项 3  
**理由**：
- 动量捕捉趋势
- 价值提供安全边际
- 质量过滤基本面恶化标的

---

## ADR-009：风险控制层级

**日期**：2026-07-02  
**决策**：采用 Cardinal 四级拦截 + RiskAgent 双重检查  
**选项**：
1. 仅输出后人工检查
2. 工具层过滤（screen.py）
3. Cardinal + RiskAgent

**选择**：选项 3  
**理由**：
- 工具层过滤只能处理硬规则
- Cardinal 可拦截 LLM 输出中的风险内容
- RiskAgent 做组合层面风险检查

---

## ADR-010：数据来源范围

**日期**：2026-07-02  
**决策**：MVP 仅使用 Tushare 公开数据，不接入付费新闻/另类数据  
**选项**：
1. 仅 Tushare
2. Tushare + 付费新闻
3. Tushare + 新闻 + 另类数据

**选择**：选项 1  
**理由**：
- 降低 MVP 复杂度
- 控制成本
- 验证核心闭环后再扩展

---

## ADR-011：目录工具格式

**日期**：2026-07-03  
**决策**：`.opencode/tool/*.ts` 使用 `@opencode-ai/plugin/tool` 定义工具，单文件单工具  
**选项**：
1. 使用 `@opencode-ai/plugin/tool` helper + zod schema
2. 使用原生 JSON Schema 对象 + 多命名 export
3. 纯原生实现

**选择**：选项 1  
**理由**：
- HelixAgent 目录工具扫描器要求每个 `.ts` 文件 `export default tool({...})`。
- 一个文件内多个命名导出（如 `export const daily`, `export const stock_basic`）不会被识别为多个工具。
- `@opencode-ai/plugin` 已在 AlphaHelix `.opencode/node_modules` 中可用，无需额外安装。

---

## ADR-012：Agent 目录与配置

**日期**：2026-07-03  
**决策**：Agent 定义放在 `.opencode/agent/`（单数），并显式声明 `tools` 白名单  
**选项**：
1. `.opencode/agents/`（复数）
2. `.opencode/agent/`（单数）

**选择**：选项 2  
**理由**：
- HelixAgent 实际扫描的是 `.opencode/agent/`。
- 放在 `.opencode/agents/` 不会报错，但配置不会被加载，导致 agent 行为不可控。
- 必须配置 `tools:` 白名单，否则 LLM 看不到任何工具。

---

## ADR-013：CLI 运行模式

**日期**：2026-07-03  
**决策**：非交互式执行必须使用 `--format json`  
**选项**：
1. 默认 TUI 模式
2. `--format json` headless 模式

**选择**：选项 2  
**理由**：
- 默认 TUI 模式在无真实 TTY 的 shell 中会进入空转/无输出状态。
- `--format json` 让 HelixAgent 直接输出事件流，可重定向、可后台运行。
- 已验证可用 `screen -dmS` 长期后台运行。

---

## ADR-014：Python 脚本参数约定

**日期**：2026-07-03  
**决策**：`screen.py` 与 `evaluate.py` 使用位置参数，工具层负责转换  
**选项**：
1. Python 脚本使用 `--strategy`、`--top-k` 等 flag
2. Python 脚本使用位置参数

**选择**：选项 2  
**理由**：
- `screen.py` 最初设计为位置参数；工具层若传 flag 会导致解析失败。
- 位置参数对 agent/bash 调用更直观，减少歧义。

---

## ADR-015：评估方式

**日期**：2026-07-02  
**决策**：收益指标由本地 Python 脚本 `scripts/evaluate.py` 确定性计算，不由 LLM 计算  
**选项**：
1. LLM 读取 memory 后计算
2. Python 脚本确定性计算

**选择**：选项 2  
**理由**：
- 避免 LLM 数值幻觉
- 可复现、可审计
- 更快、更便宜

---

## ADR-016：子进程 stdout 重定向到文件

**日期**：2026-07-03  
**决策**：`daily-screen.ts` 通过 `stdio: ['ignore', logFd, logFd]` 将 HelixAgent 子进程输出重定向到日志文件  
**选项**：
1. 使用 `stdio: ['ignore', 'pipe', 'pipe']` 实时捕获 stdout/stderr
2. 使用文件描述符重定向到 `memory/log/daily-screen-*.log`

**选择**：选项 2  
**理由**：
- `pipe` 模式下 HelixAgent CLI 偶发 `Unexpected server error`，重定向到文件后可稳定运行。
- 日志文件便于事后排查，且不会阻塞父进程事件循环。

---

## ADR-017：暂时禁用 `memory_search`

**日期**：2026-07-03  
**决策**：MVP 阶段从选股 prompt 中移除 `memory_search` 调用  
**选项**：
1. 在 prompt 中保留 `memory_search` 调用
2. 暂时移除，待 HelixAgent 修复后再启用

**选择**：选项 2  
**理由**：
- 任何包含 `memory_search` 的 prompt 都会立即触发 `Unexpected server error`。
- 该问题在 HelixAgent 侧，AlphaHelix 无法通过配置修复。
- 移除后不影响核心选股流程，仅损失历史经验复用能力。

---

## ADR-018：指数数据使用 `index_daily`

**日期**：2026-07-03  
**决策**：`scripts/evaluate.py` 对指数代码使用 `index_daily` API，个股使用 `daily` API  
**选项**：
1. 所有代码统一使用 `daily`
2. 指数代码使用 `index_daily`，个股使用 `daily`

**选择**：选项 2  
**理由**：
- Tushare 中指数行情与个股行情分属不同接口。
- `000300.SH` 等指数通过 `daily` 调用返回空数据，导致回测失败。

---

## ADR-019：先扩展因子，再补齐数据接口

**日期**：2026-07-03  
**决策**：优先使用现有 `daily_basic`/`fina_indicator`/`moneyflow` 扩展因子，而非先新增 Tushare 工具  
**选项**：
1. 先实现 `tushare_industry`/`margin`/`northbound`/`top_list` 等工具，再扩展因子
2. 先用现有工具扩展因子，验证有效后再补齐数据接口

**选择**：选项 2  
**理由**：
- `daily_basic`/`fina_indicator`/`moneyflow` 已封装为工具，立即可用。
- 新增工具接口会增加开发时间和测试复杂度，而现有数据已能覆盖质量、成长、资金等核心维度。
- 因子扩展后首次回测效果显著提升（方向准确率 20% → 70%，超额 -3.20% → +7.70%），证明优先扩因子是正确的。

---

## ADR-020：两轮筛选降低 API 调用量

**日期**：2026-07-03  
**决策**：`screen.py` 采用两轮筛选：第一轮 400 只快速初筛，第二轮对 top 80 深度计算财务/资金因子  
**选项**：
1. 对全部 400 只股票获取日线、财务、资金流向
2. 先初筛出 top 80，再对 top 80 获取财务/资金流向

**选择**：选项 2  
**理由**：
- `fina_indicator` 和 `moneyflow` 需按股票代码逐个调用，全部 400 只会使单次运行超过 20 分钟。
- 两轮筛选将额外 API 调用从约 800 次降到约 160 次，首次运行控制在 5 分钟内。
- 基本面和资金面对初筛尾部股票影响有限，不会显著降低结果质量。

---

## ADR-021：严格禁止时间穿越

**日期**：2026-07-03  
**决策**：将防未来函数作为 AlphaHelix 核心约束，T 日选股只能用 T 日及之前已公开数据  
**已落地规则**：
1. `fina_indicator` 必须满足 `ann_date <= trade_date`。
2. ST/*ST/退市 过滤使用 `namechange` 历史名称，禁止用当前名字判断历史状态。
3. 退市/停牌判断使用 `trade_date` 当天是否有 `daily` 交易记录。
4. `daily_basic`、`moneyflow` 只取截至 `trade_date` 的数据。
5. `evaluate.py` 入场价用 `trade_date` 收盘价，出场价用 `exit_date` 收盘价。

**写入文档**：`docs/agents.md` 3.6 节「未来函数禁忌与数据防穿越」。

---

## ADR-022：财报因子取最近已披露期

**日期**：2026-07-03  
**决策**：不预设定财报期，查询股票全部历史财报后按 `ann_date <= trade_date` 取最新已披露期  
**选项**：
1. 固定按 `0331/1231/0931` 选最近报告期
2. 动态查询全部财报，按公告日过滤

**选择**：选项 2  
**理由**：
- 固定报告期在 4 月初等敏感期可能用到尚未公告的一季报。
- 动态查询能自动适配任意交易日期，且能处理财报提前/延迟披露的情况。

---

## ADR-023：行业分类仅用于报告展示

**日期**：2026-07-03  
**决策**：`stock_basic` 的 `industry` 字段仅用于报告展示和 agent 定性提示，不做量化截断  
**选项**：
1. 用当前行业做自动行业集中度截断
2. 当前行业仅作展示，集中度由 agent 定性控制

**选择**：选项 2  
**理由**：
- `stock_basic` 的 `industry` 是当前分类，历史回测中可能因行业变更而错配。
- 行业集中度属于风险控制而非 alpha 来源，定性提示已能满足 MVP 需求。
- 待获取历史行业数据后再启用量化截断。

---

## ADR-026：引入 market_regime 与多策略 ensemble

**日期**：2026-07-03
**决策**：实现 `market_regime.py` 基于沪深300 判断市场状态，并扩展 `screen.py` 支持 `momentum_value_hybrid`、`quality_growth`、`contrarian` 三策略，通过 `regime` 参数自动切换
**市场状态映射**：
- `trend_up` → `momentum_value_hybrid`
- `range` → `momentum_value_hybrid`（回测显示 momentum 在震荡市仍优于 quality_growth）
- `trend_down` → `contrarian`
- `high_vol` → `quality_growth`

**回测结果**：
- 2025-01 至 2025-05：`regime` 累计超额 +8.86%，优于 `momentum_value_hybrid` 的 +4.92%。
- 2026-04 至 2026-06：`regime` 与 `momentum_value_hybrid` 均为 +4.96%（因 classifier 将该区间全部判为 trend_up/range）。

**实现位置**：`scripts/market_regime.py`、`scripts/screen.py`、`.opencode/tool/screen_candidates.ts`。

---

## ADR-027：screen_candidates 默认使用 regime 策略

**日期**：2026-07-03
**决策**：将 `screen_candidates` 工具的默认策略从 `momentum_value_hybrid` 改为 `regime`，让 agent 无需关心具体策略即可获得 regime-aware 的股票池
**理由**：
- 降低 agent prompt 复杂度。
- 保证每日选股自动适配市场状态。
- 向后兼容：agent 仍可显式指定 `momentum_value_hybrid` 等策略。

---

## ADR-028：Feedback Harness 层：用时效性结果反哺权重与 prompt

**日期**：2026-07-03
**决策**：构建 Feedback Harness，把 walk-forward/每日选股的时效性结果转化为因子权重调整、策略配置权重和 prompt 自适应提示
**核心组件**：
- `scripts/factor_ic.py`：基于选股快照中的因子值与后续收益，计算 rank IC。
- `scripts/strategy_tracker.py`：滚动跟踪各策略超额收益与命中率，softmax 输出配置权重。
- `scripts/weight_optimizer.py`：按 `new_weight = old_weight * (1 + lr * IC)` 更新因子权重，保持正负权重和归一化。
- `scripts/feedback_harness.py`：一键编排上述流程，并生成 `memory/prompt_adaptations/latest.md`。

**关键发现（基于 8 个月回测）**：
- 近期有效因子：`ocf_growth`（IC=+0.142）、`roe`（IC=+0.074）、`mom_20`（IC=+0.053）。
- 近期失效因子：`net_mf_5d`（IC=-0.189）、`net_mf_ratio`（IC=-0.162）、`net_mf_20d`（IC=-0.139）。
- 策略配置权重：`regime` > `momentum_value_hybrid` > `contrarian` > `quality_growth`。

**集成方式**：
- `screen.py` 自动加载 `memory/weights/{strategy}_latest.json` 覆盖硬编码权重。
- `alpha-analyst` prompt 要求先读取 `memory/prompt_adaptations/latest.md`。
- `daily-screen.ts` prompt 使用 `strategy=regime` 并引用 feedback 文件。

---

## 运行结果

- **2026-07-03**：首次端到端跑通，alpha-analyst 在 headless 模式下完成选股，产出 6 只组合。
  - 输出文件：`memory/stock/20260703.md`、`memory/stock/20260703.json`
- **2026-07-03 晚间**：`daily-screen.ts` 完成 3 次稳定运行；首次历史回测跑通。
  - 回测：`2026-06-15` 入选组合持有 10 个交易日，组合收益 -1.41%，相对沪深300 超额 -3.20%，方向准确率 20%。
  - 输出文件：`memory/eval/20260615_h10.json`
- **2026-07-03 因子扩展后**：`screen.py` 新增质量、资金、估值因子，同一日期回测显著提升。
  - 旧版（5 因子）：组合收益 -1.41%，超额 -3.20%，方向准确率 20%。
  - 新版（12+ 因子）：组合收益 +9.49%，超额 +7.70%，方向准确率 70%，Top3 命中率 66.67%。
  - 输出文件：`memory/eval/20260615_h10_v3.json`
- **2026-04-10 敏感期验证**：Q1 财报未全部披露，动态财报期 + 防穿越规则下回测：组合收益 +7.68%，超额 +4.81%，方向准确率 80%，Top3 命中率 100%。
  - 输出文件：`memory/eval/20260410_h10.json`
- 后续系统改进计划见 [docs/improvement-plan.md](docs/improvement-plan.md)。

---

## ADR-024：历史 ST 检查不可关闭

**日期**：2026-07-03
**决策**：生产环境和正式回测必须开启历史 ST/*ST/退市 过滤，仅允许通过 `AH_SKIP_ST_CHECK=1` 做快速实验
**验证结果**：
- 2026-04-30，关闭 ST 检查：组合收益 -7.60%，超额 -8.55%。
- 2026-04-30，开启 ST 检查：组合收益 +2.21%，超额 +1.26%。
- 差距超过 9 个百分点，证明 distressed 股票会严重拖累组合。

**实现位置**：`_tushare_utils.py:is_st_historical` 查 `namechange`；`screen.py` 通过 `AH_SKIP_ST_CHECK` 环境变量控制是否跳过。

---

## ADR-025：walkforward.py 增加可配置参数与断点续跑

**日期**：2026-07-03
**决策**：回测脚本必须支持参数化、断点续跑和进度监控，否则 6 个月以上回测无法高效完成
**已落地**：
- `--universe-size`：调整股票池大小，200 只在保证质量的同时将单期运行时间降到约 40 秒。
- `--skip-st-check`：仅用于快速实验，明确标记不用于正式评估。
- `--no-resume`：默认启用断点续跑，避免重复消耗 API。
- `--progress-file`：定期写出 JSON 进度文件。
- 自动保存每期 `memory/eval/{date}_h{horizon}.json`。

---

## 运行结果

- **2026-07-03**：首次端到端跑通，alpha-analyst 在 headless 模式下完成选股，产出 6 只组合。
- **2026-07-03 因子扩展后**：`screen.py` 新增质量、资金、估值因子，2026-06-15 回测：组合收益 +13.43%，超额 +11.64%，方向准确率 80%，Top3 命中率 100%。
- **2026-04-10 敏感期验证**：Q1 财报未全部披露，动态财报期 + 防穿越规则下回测：组合收益 +7.68%，超额 +4.81%，方向准确率 80%，Top3 命中率 100%。
- **2026-07-03 Walk-forward 回测**：完成 9 个月回测（2025-01 至 2025-05，2026-04 至 2026-06）。
  - 平均组合收益：+2.76%
  - 平均超额收益：+1.74%
  - 平均方向准确率：58.0%
  - 关键发现：历史 ST 过滤对结果影响巨大；策略在 2026-05 大跌月失效，急需 regime 判断和多策略 ensemble。

---

## ADR-031：多目标权重优化框架

**日期**：2026-07-03
**决策**：Feedback Harness 的权重优化应从「IC 驱动」升级为「方向准确率约束 + 超额收益最大化」的多目标优化
**问题**：
- 当前 `weight_optimizer.py` 基于因子 rank IC 更新权重，间接希望提升收益，但没有显式约束方向准确率。
- 用户认为方向准确率是核心约束，收益应在方向准确率达标后最大化。

**优化形式**：

```
maximize  avg_excess_return
subject to avg_direction_accuracy >= threshold（threshold >= 70%）
```

**实现路径**：
1. **离线版（当前先做）**：复用 `memory/stock/` 和 `memory/eval/` 产物，对 pass2 权重做网格/随机搜索，筛选方向准确率 ≥ 70% 的组合，取收益最高者。
2. **在线版（未来）**：在 `walkforward.py` 中每期结束后用历史已实现结果更新权重，下一期使用新权重，实现滚动优化。
3. ** regime 条件版（未来）**：不同市场状态下分别维护一套优化权重。

**注意事项**：
- 需警惕过拟合，必须保留 out-of-sample 验证。
- 70% 方向准确率阈值较高，可能在小样本（8 个月）下无满足条件的组合，此时可适当放宽至 60% 作为过渡。

**首次运行结果（2026-07-03）**：
- 已落地 `scripts/multi_objective_optimizer.py` 与完整 pass2 snapshot（`memory/stock/{date}_{strategy}_full.json`）。
- 在 8 个月样本上，对 `event_driven` 和 `contrarian` 各随机搜索 10,000 组 pass2 权重，方向准确率均无法突破 55%。
- `event_driven` baseline 已接近 pass2 权重空间内的收益最优；`contrarian` baseline 同样无法通过 pass2 权重调整扭亏。
- **判断**：70% 阈值在当前样本和策略结构下不可行，必须升级到 pass1 优化、regime 条件优化或引入新因子。

---

## ADR-030：event_driven 策略回测验证结果

**日期**：2026-07-03
**决策**：基于 walk-forward 结果，`event_driven` 是当前样本内表现最强的单一策略；`contrarian` 新公式表现弱于旧公式，需谨慎接入 regime 映射
**回测参数**：持有期 10 个交易日，`top_n=10`，`universe_size=200`，开启历史 ST 检查
**结果**：

| 区间 | 策略 | 平均超额 | 方向准确率 | 累计超额 |
|---|---|---|---|---|
| 2025-01~05 | momentum_value_hybrid | +0.95% | 58.0% | +4.92% |
| 2025-01~05 | contrarian（新公式） | +0.67% | 76.0% | +3.65% |
| 2025-01~05 | event_driven | **+2.14%** | 56.0% | **+11.60%** |
| 2025-01~05 | regime | +1.67% | 66.0% | +8.86% |
| 2026-04~06 | momentum_value_hybrid | +2.03% | 50.0% | +4.96% |
| 2026-04~06 | contrarian（新公式） | -3.47% | 33.3% | -10.35% |
| 2026-04~06 | event_driven | **+3.38%** | 43.3% | **+8.99%** |
| 2026-04~06 | regime | +2.03% | 50.0% | +4.96% |

**结论**：
1. `event_driven` 在两个区间均显著跑赢，应纳入 regime 映射或作为默认候选策略。
2. `contrarian` 新公式未能复制旧公式在 2025 年的强势，且 2026 Q2 继续亏损，说明纯反转因子对 regime 判断依赖极高。
3. 下一步：调整 `market_regime.py` 的映射，让 `event_driven` 在 trend_up/range 等状态下参与竞争；继续观察 `contrarian` 在趋势下跌月的独立表现。

---

## ADR-029：首次推送到远程仓库前的敏感信息与文档清理

**日期**：2026-07-03
**决策**：在推送到 `https://github.com/Walter1218/AlphaHelix` 前，统一清理个人本地路径、将运行时数据目录排除在版本控制外，并更新文档标记已完成项
**处理内容**：
1. 将所有文档和 README 中的 `/Users/onetwo/Documents/trae_projects/AlphaHelix` 替换为 `/path/to/AlphaHelix`。
2. 将 `memory/`（选股报告、回测结果、权重、prompt 自适应提示、日志）整体加入 `.gitignore`，仅保留 `.gitkeep`；已提交的历史运行时数据通过 `git rm --cached` 取消跟踪。
3. 确认 `TUSHARE_TOKEN` 未硬编码在任何源码或文档中，仅通过 `.env` 注入。
4. 更新 `docs/roadmap.md`、`docs/improvement-plan.md` 等文档，标记事件/反转/行业相对强度因子已完成，并修正 `reversal_score` 公式描述。

---

## ADR-032：Trace 覆盖与持久化现状

**日期**：2026-07-03
**问题**：AlphaHelix 当前是否有全局 Trace 覆盖与持久化？
**现状**：
- **已实现基础版**。新增 `scripts/_trace.py`，在 `screen.py`、`evaluate.py`、`feedback_harness.py`、`multi_objective_optimizer.py`、`walkforward.py` 关键节点写入 `memory/trace/YYYYMMDD.jsonl`。
- 每条 trace 包含 `timestamp`、`run_id`、`step`、`date`、`strategy` 和 `payload`（inputs/outputs/metadata）。
- 尚未与 HelixAgent 内部 trace 打通；也尚未按命中率自动标记 chosen/rejected。
- `docs/agents.md` 与 `docs/research.md` 中提到的 Trace 是指 HelixAgent 内部能力（`packages/opencode/src/trace/trace.ts`），但 AlphaHelix 并未调用或导出相关数据。
- 当前可审计的仅有：
  - `memory/log/daily-screen-*.log`：HelixAgent 子进程 stdout/stderr，非结构化。
  - `memory/stock/YYYYMMDD.json`：最终选股结果与因子值，缺少中间推理链。
  - `memory/eval/YYYYMMDD_*.json`：评估结果，不含决策过程。

**影响**：
- 无法做 DPO（Direct Preference Optimization）训练，因为缺少 chosen/rejected traces。
- 无法回放某次选股的完整决策链（LLM 想了什么、调用了哪些工具、为什么调整权重）。
- 无法系统性分析高/低命中率案例的差异。

**实现结果**：
- 已选择并实施方案 2 + 方案 3 的混合：
  - Python 脚本层：`scripts/_trace.py` 写入结构化 JSONL trace。
  - Agent 层：新增 `.opencode/tool/append_trace.ts`，`alpha-analyst` 在每个关键步骤调用该工具记录 reasoning。
- `memory/trace/YYYYMMDD.jsonl` 现在同时包含脚本执行事件和 agent reasoning 事件，便于端到端 case 分析。
- **尚未完成**：按命中率自动标记 chosen/rejected 并导出 DPO 数据集。

---

---

## 待决策事项

- [ ] 是否接入实盘交易（当前明确否）
- [ ] 是否引入机器学习模型做因子组合（Phase 6 评估）
- [ ] 是否开源（保留选项）
