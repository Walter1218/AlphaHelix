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
**结果（旧 regime 映射）**：

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

---

## ADR-033：更新 regime 映射，trend_up/range 优先使用 event_driven

**日期**：2026-07-04
**决策**：修改 `scripts/market_regime.py` 的映射表，让 `regime` 策略在 trend_up/range 时使用 `event_driven`，trend_down 保留 `contrarian`，high_vol 保留 `quality_growth`
**改动文件**：`scripts/market_regime.py:99-104`
**结果（新 regime 映射）**：

| 区间 | 策略 | 平均超额 | 方向准确率 | 累计超额 |
|---|---|---|---|---|
| 2025-01~05 | regime（新映射） | **+2.66%** | **64.0%** | **+14.58%** |
| 2026-04~06 | regime（新映射） | **+3.38%** | 43.3% | **+8.99%** |

**结论**：
1. 2025 年 regime 新映射同时跑赢旧 regime（+2.66% vs +1.67%）和单一 event_driven（+2.14%），方向准确率 64.0% 也优于 event_driven 单策略的 56.0%。
2. 2026 Q2 与 event_driven 单策略一致，因为 classifier 将该区间全部判为 trend_up/range。
3. `regime` 重新成为整体最优策略；`contrarian` 仅在 2025-04 的 trend_down 月被触发，当月方向准确率 100%。

---

## ADR-034：批量数据获取的上下文隔离原则（待实现）

**日期**：2026-07-04
**状态**：待实现
**问题**：当前 `screen.py` 按股票逐个调用 `daily` API，回测大量历史月份时极慢。未来改为按 `trade_date` 批量获取全市场日线可提升 50-100 倍速度，但必须防止数据泄漏（look-ahead bias）。

**设计原则**：
1. **Harness 层控制时间窗口**：任何批量数据请求必须由 `screen.py` / `evaluate.py` 等 harness 脚本发起，并显式传入 `start_date` / `end_date`。
2. **工具层不暴露全历史**：`.opencode/tool/tushare_daily.ts` 等 agent 工具禁止返回超出请求时间窗口的数据。
3. **缓存切片隔离**：批量缓存按 `trade_date` 或 `date_range` 切片存储，禁止缓存某只股票从上市至今的完整序列。
4. **Agent 不直接访问缓存**：agent 只能通过工具调用获取数据，不能直接 `read` `.cache/tushare/` 文件。
5. **回测脚本自我约束**：`walkforward.py` 调用 `screen.py` 时，确保每个 `trade_date` 只使用 ≤ `trade_date` 的数据；`evaluate.py` 只使用 `trade_date` 到 `exit_date` 的价格。

**实现方向**：
- 新增 `_tushare_utils.py` 批量接口 `tushare_call_batch(api_name, params_list)`，内部并行/限速调用。
- 新增 `fetch_daily_all(date)` 一次性获取某交易日全市场日线；`screen.py` 构建 universe 后按日期批量拉取。
- 保留现有单股票接口供 agent 工具使用。

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

## ADR-035：Walk-forward 在线学习与 regime 条件权重规划

**日期**：2026-07-04
**状态**：已实现
**目标**：建立真正的 feedback loop——每期结束后用历史已发生数据更新权重，且不同 regime 维护不同权重，使策略能随时间自适应。

### 当前 loop 的不足

1. **权重是全样本静态的**：`feedback_harness.py` 用所有历史日期算一个 pooled IC，生成一套 `*_latest.json`，未来所有日期都用同一套权重。
2. **没有按 regime 区分**：同一套权重同时用于 trend_up/range/trend_down/high_vol，无法适应不同市场环境。
3. **不是在线学习**：必须手动跑 harness，不能每期自动增量更新。
4. **策略映射固定**：`market_regime.py` 的 regime→strategy 映射是硬编码，没有根据 rolling 绩效动态调整。

### 设计方案

#### 1. 数据流

```
walkforward.py --online-update
    ↓
对每个 trade_date 按时间顺序处理：
    1. 根据当前 regime 加载该 regime 的滚动权重
    2. 用该权重跑 screen.py 选股
    3. evaluate.py 评估
    4. 将该期因子值与收益追加到 regime-specific 滚动窗口
    5. 重新计算该 regime 的 IC，更新权重
    6. 保存 memory/weights/{strategy}_{regime}_rolling.json
```

#### 2. 权重更新规则

- 对每个 regime 维护最近 N 期（如 N=6）的选股快照与评估结果。
- 在该窗口内计算每个因子的 rank IC。
- 用 `weight_optimizer.py` 的 IC 规则更新权重：`new_weight = old_weight * (1 + lr * IC)`。
- 更新后的权重只用于下一期及以后，**绝不用于当前期**（防未来函数）。

#### 3. regime 条件权重加载

- `screen.py` 在 `regime` 模式下，确定 `actual_strategy` 和 regime 后：
  - 优先加载 `memory/weights/{actual_strategy}_{regime}_rolling.json`
  - fallback 到 `memory/weights/{actual_strategy}_latest.json`
  - 再 fallback 到硬编码权重
- 非 regime 模式仍使用策略级权重。

#### 4. 动态策略映射（可选第二阶段）

- `strategy_tracker.py` 已能计算各策略滚动超额收益。
- 扩展 `market_regime.py`：根据 regime 在最近 N 期各策略的表现，动态选择该 regime 下表现最好的策略。
- 例如：若最近 6 个月 range 市场下 `event_driven` 明显优于 `momentum_value_hybrid`，则 range→event_driven 可进一步强化。

#### 5. `--auto` 在线 harness（可选第三阶段）

- `feedback_harness.py --auto` 自动扫描 `memory/eval/`，识别新增日期。
- 对新增日期按 regime 增量更新滚动权重，不重新计算全部历史。

### 防未来函数约束

- T 期权重只能使用 < T 期的数据。
- 滚动窗口用 `max_lookback` 限制，避免用太远历史。
- 评估在线学习效果时，必须与静态权重 walk-forward 同参数对比。

### 验收标准

- 在线学习版本的方向准确率 ≥ 静态版本。
- 不同 regime 的权重有明显差异（如 trend_down 中反转因子权重更高）。
- 无未来函数：用 T 期权重回测 T-1 期结果不变。

---

## ADR-036：30 个月全样本回测结果与样本选择偏差

**日期**：2026-07-04
**状态**：已完成分析
**回测参数**：`regime` 策略（新映射），持有期 10 个交易日，`top_n=10`，`universe_size=200`，2024-01 至 2026-06 共 30 期。

**结果**：

| 年份 | 期数 | 平均组合收益 | 平均超额 | 方向准确率 | 累计超额 |
|---|---|---|---|---|---|
| 2024 | 12 | -0.36% | -0.37% | 48.3% | -4.42% |
| 2025 | 12 | +2.57% | +1.48% | 58.3% | +17.76% |
| 2026 H1 | 6 | -1.50% | -2.14% | 33.5% | -12.81% |
| **合计** | **30** | **+0.58%** | **+0.02%** | **49.4%** | **-4.44%** |

**关键发现**：
1. **样本选择偏差**：早期 8 个月样本（2025-01~05、2026-04~06）恰好避开了 2026 年 1-2 月的大跌，导致对策略真实能力过度乐观。
2. **2024 年整体疲弱**：全年 12 期平均超额为负，方向准确率仅 48.3%，说明策略对 2024 年市场风格（红利、低波、大盘蓝筹）不适应。
3. **2026 H1 暴跌集中**：2026-01（-8.89%）、2026-02（-13.76%）两期严重拖累，event_driven 在急跌环境中未能及时切换防御。
4. **2025 年是主要贡献者**：12 期平均超额 +1.48%，方向准确率 58.3%，累计超额 +17.76%。

**启示**：
- 不能依赖单一年份或精心挑选的样本评估策略。
- 必须引入熊市/急跌防御机制：动态仓位、空仓选项、趋势过滤、或 regime 条件权重。
- 在线学习（ADR-035）应尽快实现，让策略在 2024 年失利后自适应调整。

---

## ADR-037：在线学习 walk-forward 验证结果

**日期**：2026-07-04
**状态**：已跑测，首版 IC 驱动在线学习未显著优于静态权重
**回测参数**：`regime` 策略，持有期 10 个交易日，`top_n=10`，`universe_size=200`，`skip-st-check`，2024-01 至 2026-06 共 30 期。

**实现**：
- `walkforward.py` 新增 `--online-update`、`--online-lookback`、`--online-lr` 参数。
- 每期按 regime 加载 `memory/weights/{strategy}_{regime}_rolling.json`；用该权重选股评估后，将该期纳入该 regime 的最近 N 期窗口并更新权重；更新后的权重仅用于下一期。
- `screen.py` 支持 `pass1_weights_override` / `pass2_weights_override`，并在未覆盖时自动加载 regime 滚动权重。

**结果对比**（同参数）：

| 版本 | 频率 | 平均超额 | 方向准确率 | Top3 命中率 | 累计超额 |
|---|---|---|---|---|---|
| 静态权重 | 月度 | +0.02% | 49.4% | 51.1% | -4.44% |
| 在线学习 v1 | 月度（lr=0.5, lookback=6） | +0.02% | 51.5% | 47.2% | -6.13% |
| 静态权重 | 周度 | +0.60% | 49.6% | 50.7% | +236.94% |
| 在线学习 v1 | 周度（lr=0.3, lookback=12） | +0.01% | 46.9% | 46.2% | -41.59% |

> 注：周度累计超额因 127 期复利而数值较大，平均超额更可比。

**关键发现**：
1. **在线学习 consistently 劣于静态权重**：月度和周度的 Top3 命中率、方向准确率、累计超额均下降。
2. 周度静态反而比月度静态平均超额更高（+0.60% vs +0.02%），说明更频繁的再平衡在当前因子体系下有利；但在线学习把这一优势完全抹去。
3. 简单的 rank-IC 乘性更新对噪声敏感，在 30 期（月度）或 127 期（周度）样本下容易过拟合近期，导致权重漂移。
4. 在 2026-01/02 急跌期间，在线学习未能阻止大额亏损，反而因前几期权重漂移加剧了损失。

**决策**：
- 保留 `--online-update` 框架和 regime 滚动权重文件机制，作为后续实验入口。
- 当前默认 **不启用** 在线学习，直到找到更稳定的更新规则。
- 下一步尝试：
  1. 缩小学习率（0.05）或引入动量平滑；
  2. 仅更新 pass2 排序权重，保持 pass1 过滤稳定；
  3. 用滚动超额收益/命中率而非纯 IC 指导权重；
  4. 在线学习 + 熊市防御（空仓/减仓）联动。

---

## ADR-038：Tushare 并发预取与数据上下文隔离

**日期**：2026-07-04
**状态**：已实现
**目标**：解决 walk-forward 慢、以及智能体/脚本可能访问窗口外数据的问题。

### 实现
1. **`_tushare_utils.py` 线程安全与并发**：
   - 用 `threading.Lock` 保护全局限流时间戳，支持多线程。
   - 新增 `concurrent_map()`，默认 `ALPHAHELIX_MAX_WORKERS=4`。
   - 新增 `ALPHAHELIX_DATA_WINDOW_START/END` 环境变量；数据接口（daily/daily_basic/moneyflow/fina/forecast/express/index_daily 等）请求超出窗口时直接抛异常。
2. **`scripts/prefetch_data.py`**：
   - 按 `--start/--end/--lookback` 批量并发拉取窗口内所需数据。
   - 预取交易日历、沪深300 日线、stock_basic、每个交易日的 daily/daily_basic/moneyflow，以及指定数量股票的 fina/forecast/express。
   - 预取时自动设置 `ALPHAHELIX_DATA_WINDOW`，确保拉取范围即允许范围。
3. **`screen.py` 按日期截面加载**：
   - `load_daily_window()` 和 `load_moneyflow_window()` 按 `trade_date` 批量加载截面数据，避免对每只股票调用 `daily(ts_code, start, end)` 和 `moneyflow(ts_code, start, end)` 造成的缓存不命中和慢请求。
   - pass1/pass2 循环已并行化。

### 效果
- 周度 127 期 walk-forward 从「单期 600+s、预计无法完成」降到约 **98 分钟**（静态）。
- 预取一次后，后续选股主要读本地缓存，Tushare API 调用被限制在预取窗口内。
- 数据上下文隔离让回测脚本/智能体在设置了窗口后无法请求窗口外数据，降低未来函数和偷看风险。

### 待完善
- TypeScript 工具（`.opencode/tool/tushare_*.ts`）尚未接入窗口隔离；目前靠 Python 层限制。下一步可把 `ALPHAHELIX_TRADE_DATE` 上限检查加入 TS 工具。
- `memory/` 中的运行时缓存/结果仍不提交 git；`.gitignore` 已排除。

---

## ADR-039：选股分数阈值与空仓机制

**日期**：2026-07-04
**状态**：已实验，`walkforward.py` 已支持 `--min-score`
**目标**：通过「分数未达标就空仓」来过滤低置信预测，提升风险调整后收益或命中率。

### 实现
- `walkforward.py` 新增 `--min-score` 参数。
- 若当期最高分 `< min_score`，则该期视为空仓（组合收益=0，超额=-基准收益）。
- 空仓期的方向判断规则：基准跌视为正确，基准涨视为错误。

### 月度样本实验结果（`regime`，2024-01 至 2026-06，10 日持有期，`top_n=10`，`universe_size=200`）

| min_score | 空仓期数 | 平均超额 | 方向准确率 | Top3 命中率 | 累计超额 |
|---|---|---|---|---|---|
| 无阈值 | 0 | +0.02% | 49.4% | 51.1% | -4.44% |
| 0.65 | 3 | **+0.11%** | 47.4% | 46.7% | **-0.94%** |
| 0.70 | 9 | -0.25% | 38.4% | 32.2% | -10.95% |
| 0.75 | 23 | -0.31% | 49.0% | 10.0% | -9.59% |
| 0.80 | 29 | -0.41% | 42.0% | 1.1% | -12.55% |

### 结论
- **轻度阈值（0.65）能改善累计超额**（从 -4.44% 提升到 -0.94%），但代价是方向准确率和 Top3 命中率下降。
- **阈值过高会频繁空仓**，错过反弹，反而拉低收益。
- 当前 score 是加权秩次和，阈值对策略/因子权重敏感，不宜硬编码。

### 下一步
- 把阈值从全局固定值改为 **regime 自适应**（不同市场状态下使用不同 min_score）。
- 用滚动超额收益/最大回撤做阈值网格搜索，而不是手工设定。
- 探索「分数差距（top - median）」阈值，可能比绝对分数更稳定。

---

## ADR-040：基于历史 rank IC 的 pass2 权重校准

**日期**：2026-07-04
**状态**：已实验，`walkforward.py` 已支持 `--ic-calibrate`

### 问题
月度回测显示 score 对入选股没有排序能力：rank1~rank10 命中率都在 48~53% 之间，基本随机。说明 pass2 权重与真实预测方向不一致。

### 因子 IC 诊断（月度 30 期样本）

| 因子 | rank IC | 原权重方向 |
|---|---|---|
| forecast_pchange_mid | **-0.10** | event_driven 正权重 → 反向 |
| amount_ratio_5d | **-0.09** | 部分策略正权重 → 反向 |
| mom_20 | **-0.06** | 动量策略正权重 → 反向 |
| roe | **+0.08** | 权重偏低 |
| total_mv | **+0.07** | 有一定权重 |
| ocf_growth | **+0.07** | 权重偏低 |
| reversal_score | **+0.07** | 权重偏低 |

### 实现
- 新增 `scripts/calibrate_weights_from_ic.py`：读取历史选股快照与收益，计算各因子 rank IC，负 IC 置 0，正 IC 归一化生成 pass2 权重。
- `walkforward.py` 新增 `--pass2-weights` 用于加载固定 IC 权重文件。
- `walkforward.py` 新增 `--ic-calibrate`：每期用**之前所有期**的历史数据重新计算 pass2 权重，确保无未来函数。

### 结果对比（月度 30 期，`universe_size=200`）

| 版本 | 平均超额 | 方向准确率 | Top3 命中率 | 累计超额 |
|---|---|---|---|---|
| 静态默认权重 | +0.02% | 49.4% | 51.1% | -4.44% |
| IC 权重（样本内） | **+0.84%** | **54.4%** | **58.3%** | **+28.64%** |
| IC 校准（样本外 walk-forward） | +0.12% | 50.1% | 51.7% | -0.33% |

> ⚠️ 「IC 权重（样本内）」属于**诊断性/违规上界**，因为它用全样本 IC 生成权重后又回测同一区间，违反 AGENTS.md C38。只有「IC 校准（样本外）」是合规回测。

### 关键发现
1. 样本内 IC 权重大幅提升，但属于用同一批数据训练又测试，存在过拟合。
2. 真正的 walk-forward IC 校准（每期用历史数据）仍有提升：平均超额 +0.12% vs +0.02%，累计超额 -0.33% vs -4.44%，win rate 60.0% vs 43.3%。
3. rank 命中率仍未呈现严格单调性（rank1 不一定最高），说明仅靠 pass2 权重调整还不够，需要继续优化 pass1 过滤和因子本身。

### 决策
- `--ic-calibrate` 作为可选实验模式保留，**不设为默认**（样本外收益还不稳定）。
- 下一步：
  1. 对 pass1 权重也做 IC 校准；
  2. 引入更多 stable 的反向/质量/流动性因子；
  3. 按 regime 分别校准权重，而非全样本统一 IC。

---

## 待决策事项

- [ ] 是否接入实盘交易（当前明确否）
- [ ] 是否引入机器学习模型做因子组合（Phase 6 评估）
- [ ] 是否开源（保留选项）

