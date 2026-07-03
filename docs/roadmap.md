# AlphaHelix 落地路线图

> 状态：Phase 1~4 已完成，事件/反转/行业相对强度因子已接入，Phase 5（自动化调度）与 Phase 6/7（进化优化/harness 深化）为当前重点。

## Phase 1：MVP（已完成）

### 目标
跑通一次端到端选股流程，验证 HelixAgent + Tushare 工具链可用。

### 任务
- [x] 创建项目骨架与文档
- [x] 安装依赖（bun、python、tushare）
- [x] 验证 `.opencode/tool/tushare_*.ts` 能被 HelixAgent 加载
- [x] 验证 `tushare_stock_basic`、`tushare_daily` 可正常返回数据
- [x] 验证 `alpha-analyst` agent 能执行选股指令
- [x] 验证选股结果能写入 `memory/stock/YYYYMMDD.md`
- [x] 修复 `daily-screen.ts`：切换到 CLI headless 模式并稳定运行

### 验收标准
```bash
bun /path/to/HelixAgent/packages/opencode/src/index.ts run \
  --agent alpha-analyst --format json --title "AlphaHelix test" \
  "执行今日 A 股选股流程"
# 期望：输出 JSON 格式的股票列表，并写入 memory
```

---

## Phase 2：因子、策略与 Regime（已完成）

### 目标
实现本地因子计算、候选池生成、多策略 ensemble 与市场 regime 判断。

### 任务
- [x] 实现 `momentum_value_hybrid` 因子初筛（动量、估值、规模、流动性）
- [x] 加入 ROE、营收增速、净利润增速、经营现金流增速等质量因子
- [x] 加入资金流因子（5 日/20 日主力净流入、净流入占比）
- [x] 扩展估值因子（EP、BP、SP、股息率）
- [x] 实现多策略：`quality_growth`、`contrarian`
- [x] 实现 `market_regime.py`：基于沪深300 判断市场状态
- [x] 实现 `regime` 自动切换策略
- [x] 接入本地 JSON 缓存，避免重复调用 tushare
- [x] 实现 `screen_candidates` 工具的异常处理与限流
- [x] 加入黑名单机制（ST、退市、次新股，基于历史名称）

### 验收标准
```bash
python scripts/screen.py regime 20260702 50
# 期望：输出 50 只候选股票的 JSON，自动按 regime 选择底层策略
```

---

## Phase 3：Memory 与风控（部分完成）

### 目标
让系统具备经验积累能力和基础风险拦截能力。

### 已完成任务
- [x] 规范 `memory/stock/*.md` 格式，确保可被 RAG 检索
- [x] 扩展 Cardinal 规则：
  - 拦截 ST/退市股推荐（基于 `namechange` 历史名称，非当前名字）
  - 拦截低流动性标的（日均成交额 < 5000 万）
  - 行业集中度数量控制（单一行业 ≤ top_n 的 40%）
  - 缺少止损价暂停（agent prompt 要求）
  - 财报 `ann_date` 防穿越校验

### 待完成任务
- [ ] 实现 `memory_search` 在选股流程中的调用（当前 HelixAgent 调用该工具会触发 `Unexpected server error`，待修复后启用）
- [ ] 扩展 Cardinal 规则：
  - 行业市值权重控制（单一行业 ≤ 40% 市值权重）
  - 高波动/高杠杆叙事拦截
  - 业绩预亏/暴雷拦截
- [ ] 实现 `RiskAgent` 的提示词与输出格式

### 验收标准
```bash
mimo run "alpha-analyst: 查询 2026-05 月半导体板块的选股记录并总结教训"
# 期望：能检索到历史记录并给出分析（待 memory_search 修复）
```

---

## Phase 4：评估与回测（已完成）

### 目标
实现历史持有期自动评估，形成可验证的进化基础。

### 任务
- [x] 实现 `scripts/evaluate.py` 确定性收益计算
- [x] 计算方向准确率、Top3 命中率、超额收益、最大回撤、置信度相关性
- [x] 完成首次历史回测（2026-06-15，10 日持有期）
- [x] 实现历史 walk-forward 回测（已覆盖 2025-01 至 2025-05、2026-04 至 2026-06，共 8 个月）
- [x] 生成评估报告（JSON），默认写入本地 `memory/eval/` 与 `memory/eval/walkforward_*.json`（运行后生成，不提交 git）

### 验收标准
```bash
python scripts/walkforward.py \
  --start 20250101 --end 20250531 \
  --strategy regime --horizon 10 --top-n 10 --universe-size 200
# 期望：输出月度与累计回测指标
```

---

## Phase 5：自动化调度（下一步）

### 目标
实现无人值守的每日选股与定期反馈更新。

### 任务
- [ ] 配置 cron：
  - 交易日 15:30 执行 `daily-screen.ts`
  - 每月第一个交易日 09:00 执行 `feedback_harness.py`（更新权重与 prompt 自适应提示）
- [ ] 添加日志轮转与通知（可选：飞书/邮件）
- [ ] 添加失败重试与告警
- [ ] 将评估结果自动追加写入 `memory/stock/{date}.md`

### 验收标准
```bash
crontab -l | grep AlphaHelix
# 期望：看到 daily-screen 与 feedback_harness 定时任务
```

---

## Phase 6：Feedback Harness 与在线进化（进行中）

### 目标
基于时效性数据持续优化因子权重、策略配置与 prompt 风格。

### 任务
- [x] 实现 factor IC 计算（`scripts/factor_ic.py`）
- [x] 实现 strategy tracker（`scripts/strategy_tracker.py`）
- [x] 实现 weight optimizer（`scripts/weight_optimizer.py`）
- [x] 实现 feedback harness orchestrator（`scripts/feedback_harness.py`）
- [x] 让 `screen.py` 自动加载 `memory/weights/*_latest.json`
- [x] 让 agent prompt 读取 `memory/prompt_adaptations/latest.md`
- [ ] 实现**在线学习**：每新增一期 walk-forward 结果，自动增量更新权重，无需手动指定日期
- [ ] 引入**分行业命中率反馈**：识别模型在哪些行业有效/失效
- [ ] 引入**置信度校准反馈**：根据 `high/medium/low` 命中率调整 agent 置信度阈值
- [ ] 引入**参数网格反馈**：对 `learning_rate`、`temperature`、`lookback` 做回测内网格搜索

### 验收标准
```bash
python scripts/feedback_harness.py \
  --dates 20250127,20250228,20250331,20250430,20250530,20260430,20260529,20260615 \
  --start 20250101 --end 20260615 --horizon 10 --strategy regime
# 期望：生成 memory/weights/*_latest.json 与 memory/prompt_adaptations/latest.md
```

---

## Phase 7：高级数据与模型实验（未来）

### 目标
引入更多数据源和模型能力，进一步提升 alpha。

### 任务
- [x] 建立 AlphaHelix Trace 持久化（`memory/trace/` JSONL）
- [ ] 把 agent LLM 思考过程纳入 trace
- [ ] 定期导出 DPO 数据集（chosen/rejected traces）
- [ ] 引入更多数据源：新闻 sentiment、机构持仓、北向资金、融资融券、龙虎榜
- [ ] 尝试不同模型（kimi/gpt/claude）的选股效果对比
- [ ] 引入机器学习模型做因子组合（在样本外验证稳健后再合并）
- [ ] 探索行业轮动与宏观 regime 的深度融合

### 验收标准
- 连续 3 个月方向准确率 > 55%
- 连续 3 个月相对沪深300超额收益 > 3%

---

## 多轨并行视图

除按时间划分的 Phase 外，也可按系统层级并行推进：

| 轨道 | 当前状态 | 下一步 | 负责人/模块 |
|---|---|---|---|
| **数据轨道** | 量价、估值、财务、资金、业绩预告/快报数据已接入 | 披露日期预告、融资融券、北向资金、龙虎榜 | `.opencode/tool/tushare_*.ts` |
| **因子/策略轨道** | 18+ 因子、四策略 + regime 切换已落地；event_driven 回测验证为当前最强单一策略 | 将 event_driven 接入 regime 映射；range 市场提升 contrarian 权重；调优 quality_growth | `scripts/screen.py`, `scripts/market_regime.py` |
| **风控轨道** | ST/退市、流动性、财报防穿越已落地 | 业绩预亏/暴雷拦截、行业市值权重控制、高波动叙事拦截 | `scripts/screen.py`, Cardinal |
| **Agent/执行轨道** | alpha-analyst 可执行选股 | 置信度校准、memory_search 修复后接入、失败重试 | `.opencode/agent/alpha-analyst.md`, `daily-screen.ts` |
| **评估轨道** | evaluate.py + walkforward.py 已落地 | 交易成本、更长样本、分行业命中率 | `scripts/evaluate.py`, `scripts/walkforward.py` |
| **Feedback Harness 轨道** | v1 已落地 | 在线学习、事件因子 IC 反馈、分行业反馈、置信度校准、参数网格 | `scripts/feedback_harness.py` |
| **自动化/运维轨道** | 手动运行 | cron、日志轮转、告警 | `scripts/daily-screen.ts`, cron |

各轨道独立推进，关键节点通过 `memory/` 目录中的权重、prompt 自适应提示、评估报告进行联动。

## 优先级与时间线

```
Week 1-2:   Phase 1  MVP（已完成）
Week 3-5:   Phase 2  因子与策略（已完成）
Week 6-8:   Phase 3  风控与 Memory（部分完成，memory_search 阻塞）
Week 9-11:  Phase 4  评估与回测（已完成）
Week 12-13: Phase 2/6 事件因子与反转因子接入（下一步，基于东山精密案例）
Week 14:    Phase 5  自动化调度
Week 15-17: Phase 6  Feedback Harness 在线学习深化
Week 18+:   Phase 7  高级数据与模型实验（未来）
```

---

## 当前状态

- [x] 项目骨架创建完成
- [x] 调研报告落盘
- [x] 架构设计文档完成
- [x] Phase 1 完成：2026-07-03 首次端到端跑通
- [x] Phase 2 完成：因子扩展、多策略、`regime` 切换
- [x] Phase 4 完成：walk-forward 回测 8 个月
- [x] Phase 6 v1 完成：Feedback Harness 落地，可生成动态权重与 prompt 自适应提示

### 阶段评估（2026-07-03）

AlphaHelix 处于 **Phase 4 完成、Phase 5/6 并行推进** 的阶段：

- 已验证：
  - HelixAgent 能加载 `.opencode/agent/alpha-analyst.md`，LLM 能调用 Tushare 工具与 `screen.py`，最终写入 `memory/stock/`。
  - `daily-screen.ts` 可无人值守执行，stdout 重定向到 `memory/log/`。
  - `scripts/evaluate.py` 可计算历史持有期收益。
  - `screen.py` 支持 `momentum_value_hybrid`、`quality_growth`、`contrarian` 三策略与 `regime` 自动切换。
  - `market_regime.py` 可基于沪深300 判断市场状态。
  - `walkforward.py` 已完成 8 个月回测，`regime` 策略在 2025 年累计超额 +8.86%，优于单一 momentum 的 +4.92%。
  - Feedback Harness 已产出动态权重与 prompt 自适应提示。

- 已知问题：
  - `memory_search` 工具在当前 HelixAgent 环境下会触发 `Unexpected server error`，已暂时从选股流程中移除。
  - 行业集中度控制目前为数量控制，市值权重控制尚未完全实现。
  - Feedback Harness 仍为手动运行，需接入 cron 实现在线学习。
  - `quality_growth` 策略在回测中表现偏弱，需继续调优。
  - 当前因子体系对**预期/主题驱动型行情**覆盖不足，2026-01-30 的东山精密案例即因静态财务/动量因子滞后而错失后续 40%+ 涨幅。

### 东山精密案例启示（2026-07-03）

对 东山精密（002384.SZ）以 2026-01-30 为数据截止点进行回测外验证，发现：

- 2026-01-30 时，模型看到 `mom_20=-8.62%`、`PE=130`、`营收增速=2.28%`、`20日资金净流出`，按 `range`→`momentum_value_hybrid` 逻辑给出**谨慎/中性**判断。
- 实际 2026-02~04 股价大涨 **+40.91%**（10 日持有超额 +1.78%，20 日 +28.92%，40 日 +46.56%）。
- 事后发现：2026-04-08 才公告 Q1 业绩预告（预增 119%-152%）和 2025 年报快报，行情明显**早于公告启动**，市场在交易 AI 算力/光模块/索尔思主题预期。

**结论**：当前因子体系擅长基于已披露基本面和动量选股，但对**预期驱动、主题驱动、事件驱动**的机会存在结构性盲区。已据此新增 `forecast`/`express` 事件因子、短期反转因子和行业相对强度因子；下一步应继续完善事件前置布局、在线学习与自动化调度。

### 下一步优先级（已按案例启示重排）

1. **range 市场下提升 contrarian 权重**：通过 `strategy_tracker` 动态调整，让 range 市场不只依赖 `momentum_value_hybrid`。
2. **资金流动量因子优化**：用 `net_mf_ratio` 替代绝对金额，并区分 5日/20日背离信号。
3. **接入披露日期预告**：基于 `disclosure_date` 做事件驱动布局。
4. **在线学习**：让 `feedback_harness.py` 自动发现新增回测结果并增量更新权重。
5. **自动化调度**：把 `daily-screen.ts` 与 `feedback_harness.py` 接入 cron，实现无人值守。
6. **分行业命中率反馈**：计算每个行业在最近 N 期的命中率，指导 agent 在行业配置上倾斜/回避。
7. **置信度校准**：统计 `high/medium/low` 置信度对应的实际收益。
8. **参数网格搜索**：对 `learning_rate`、`temperature`、`lookback` 等 harness 参数做回测内搜索。

---

## 完整开发任务清单

> 以下按 Phase / 轨道汇总，便于一次性查看全部待办。

### Phase 5：自动化调度

- [ ] 配置 cron：交易日 15:30 执行 `daily-screen.ts`
- [ ] 配置 cron：每月第一个交易日 09:00 执行 `feedback_harness.py`
- [ ] 添加日志轮转与通知（可选：飞书/邮件）
- [ ] 添加失败重试与告警
- [ ] 将评估结果自动追加写入 `memory/stock/{date}.md`

### Phase 2/6：因子与 Feedback Harness 联动（当前最高优先级）

> 基于东山精密案例，当前因子体系对预期/主题驱动行情覆盖不足，应优先补齐以下因子并通过 Harness 迭代权重。

- [x] **接入业绩预告/快报因子**：基于 Tushare `forecast`/`express` 构建业绩超预期事件因子，新增 `event_driven` 策略；walk-forward 验证为当前最强单一策略
- [ ] **接入披露日期预告**：基于 `disclosure_date` 做事件驱动布局
- [x] **加入短期反转/超跌因子**：已新增 `mom_5`、`amount_ratio_5d`、`reversal_score`，并强化 `contrarian` 策略；新公式回测 2025 平均超额 +0.67%，2026 Q2 -3.47%
- [ ] **资金流动量因子优化**：用 `net_mf_ratio` 替代绝对金额，捕捉 5日/20日背离
- [x] **构建行业相对强度因子**：已新增 `sector_momentum`、`relative_to_sector`、`sector_mom5`、`sector_amount_ratio`，应用于 `contrarian` 与 `event_driven`
- [ ] **将 event_driven 接入 regime 映射**：当前 regime 在 2026 Q2 实际等同于 momentum，错失 event_driven 超额收益
- [ ] **多目标离线权重优化**：pass2 权重调整无法实现 ≥ 55% 方向准确率；下一步尝试 (a) pass1 权重优化以扩大候选池，(b) regime 条件优化，(c) 引入新因子后再以 70% 为约束
- [ ] **range 市场下 contrarian 权重动态提升**：通过 `strategy_tracker` 根据滚动绩效调整策略配比
- [ ] `quality_growth` 策略调优
- [ ] 行业轮动与宏观 regime 深度融合
- [ ] 基于总市值的 sector weight cap

### Phase 6：Feedback Harness 在线进化

- [ ] 实现 `feedback_harness.py --auto` 在线学习（自动发现新增回测结果并增量更新）
- [ ] 引入分行业命中率反馈（`scripts/sector_tracker.py`）
- [ ] 引入事件因子 IC 反馈（`forecast`/`express` 等）
- [ ] 引入置信度校准反馈（统计 `high/medium/low` 命中率）
- [ ] 引入参数网格反馈（对 `learning_rate`、`temperature`、`lookback` 等做回测内搜索）

### Phase 3：风控深化

- [ ] 业绩预亏/暴雷拦截（利用 `forecast` type）
- [ ] 行业市值权重控制（单一行业 ≤ 40% 市值权重）
- [ ] 高波动/高杠杆叙事拦截
- [ ] 接入 `memory_search`（待 HelixAgent 修复）

### Phase 4：评估层增强

- [ ] `evaluate.py` 加入交易成本
- [ ] 扩展 walk-forward 到 12+ 个月
- [ ] 分行业命中率统计

### Phase 5：自动化调度

- [ ] 配置 cron：交易日 15:30 执行 `daily-screen.ts`
- [ ] 配置 cron：每月第一个交易日 09:00 执行 `feedback_harness.py`
- [ ] 添加日志轮转与通知（可选：飞书/邮件）
- [ ] 添加失败重试与告警
- [ ] 将评估结果自动追加写入 `memory/stock/{date}.md`

### Phase 7：高级数据与模型实验

- [ ] 新闻 sentiment、机构持仓、北向资金、融资融券、龙虎榜数据接入
- [ ] 分析师一致预期数据接入
- [ ] 多模型选股效果对比（kimi/gpt/claude）
- [ ] 机器学习因子组合模型（样本外验证后再合并）
- [x] 建立 AlphaHelix Trace 持久化（`memory/trace/` JSONL）
- [ ] 把 agent LLM 思考过程纳入 trace
- [ ] 定期导出 DPO 数据集（chosen/rejected traces）
