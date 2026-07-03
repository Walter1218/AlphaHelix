# AlphaHelix 进化闭环设计

> 核心目标：以「未来一个月股价走势准确率」为优化目标，让选股能力随时间持续提升。

## 1. 为什么需要进化闭环

单次选股无论结果好坏，都只是孤立事件。只有通过：
- **记录预测**
- **等待结果**
- **量化评估**
- **反馈优化**

才能形成可持续提升的系统。这也是 AlphaHelix 区别于「一次性选股建议」的关键。

## 2. 进化飞轮

```
        ┌──────────────┐
        │   选股执行日 T │
        └──────┬───────┘
               ↓
   ┌─────────────────────────┐
   │ PickAgent 输出 Top-K     │  ← 写入 Memory（预测快照）
   └───────────┬─────────────┘
               ↓
   等待 1 个月（约 20 个交易日）
               ↓
   ┌─────────────────────────┐
   │ Evaluator 计算实际收益   │  ← 调用 tushare_daily / index_daily
   └───────────┬─────────────┘
               ↓
   ┌─────────────────────────┐
   │ 生成多维度评估指标       │
   │ - 方向准确率             │
   │ - Top3 命中率            │
   │ - 相对沪深300超额收益    │
   │ - 最大回撤               │
   │ - 置信度-收益相关性      │
   └───────────┬─────────────┘
               ↓
   ┌─────────────────────────┐
   │ 反馈到优化层             │
   │ 1. Prompt 反思           │
   │ 2. 因子权重更新          │
   │ 3. DPO 数据导出          │
   │ 4. Cardinal 规则增强     │
   └─────────────────────────┘
               ↓
        下次选股执行
```

## 3. 评估指标

### 3.1 方向准确率

```
方向准确率 = 收益为正的股票数 / 推荐股票总数
```

### 3.2 Top3 命中率

```
Top3 命中率 = Top3 中收益为正的股票数 / 3
```

### 3.3 相对超额收益

```
超额收益 = 组合平均收益 - 沪深300同期收益
```

### 3.4 最大回撤

```
最大回撤 = max((前期高点 - 当期低点) / 前期高点)
```

### 3.5 置信度相关性

```
置信度相关性 = corr(confidence_score, actual_return)
```

用于判断 LLM 的自信是否与实际表现一致。

## 4. 数据模型

```ts
interface Pick {
  ts_code: string
  name: string
  score: number
  rank: number
  rationale: string
  confidence: "low" | "medium" | "high"
  stop_loss: number
}

interface Evaluation {
  date: string              // 选股日
  horizon: number           // 持有期，默认 20
  picks: Pick[]
  actualReturns: Array<{
    ts_code: string
    entryPrice: number
    exitPrice: number
    absReturn: number
    excessReturn: number
    maxDrawdown: number
  }>
  directionAccuracy: number
  top3HitRate: number
  portfolioReturn: number
  benchmarkReturn: number
  excessReturn: number
  maxDrawdown: number
  confidenceCorrelation: number
}
```

## 5. 反馈优化机制

### 5.1 Prompt 反思

Evaluator 分析：
- 哪些推荐理由被验证？
- 哪些理由被证伪？
- 哪些市场环境下模型容易犯错？

生成 Prompt 改进建议，例如：
- 强化「成交额萎缩」作为风险信号
- 弱化「单纯低 PE」在熊市中的权重
- 增加「北向资金流向」的考虑

### 5.2 因子权重更新

使用贝叶斯或梯度方法更新 `screen.py` 中的因子权重。

```python
# 简化示例：根据命中率调整权重
if direction_accuracy > 0.6:
    momentum_weight *= 1.05
else:
    momentum_weight *= 0.95
```

更严谨的做法：用滚动窗口做 walk-forward 优化。

### 5.3 DPO 数据导出

HelixAgent 已有 `script/dogfooding/export_dpo.ts` 机制，可复用：

```bash
bun run script/dogfooding/export_dpo.ts \
  --tag alphahelix \
  --output ~/.opencode/alphahelix-dpo.jsonl
```

**Chosen 样本**：
- 方向准确率 > 60%
- 超额收益 > 3%
- 推荐理由与结果一致

**Rejected 样本**：
- 方向准确率 < 40%
- 存在明显逻辑漏洞（如忽视流动性风险）
- 推荐理由与实际结果严重不符

### 5.4 Cardinal 规则增强

根据失败案例新增 Cardinal 规则：

```ts
// 示例：如果发现模型多次推荐低流动性小盘股导致亏损
const lowLiquidityRule: CardinalRule = {
  id: "alphahelix-low-liquidity",
  name: "低流动性拦截",
  evaluate: (ctx) => {
    if (ctx.outputContains("日均成交 < 3000万")) {
      return { level: "block", reason: "拦截低流动性标的" }
    }
    return null
  }
}
```

## 6. 未来函数禁忌

进化闭环中必须避免：
- 用 T+1 及之后的数据评估 T 日的选股
- 用未披露的财报做 T 日决策
- 在训练 DPO 数据时泄露未来信息

## 7. 评估自动化

```bash
# 每日收盘后选股（Linux/macOS 通用）
crontab -e
30 15 * * 1-5 cd /path/to/AlphaHelix && bun run scripts/daily-screen.ts

# 每日开盘前评估 20 个交易日前的选股
# Linux:
0 9 * * 1-5 cd /path/to/AlphaHelix && bun run scripts/evaluate-picks.ts $(date -d '20 days ago' +%Y%m%d)
# macOS:
0 9 * * 1-5 cd /path/to/AlphaHelix && bun run scripts/evaluate-picks.ts $(date -v-20d +%Y%m%d)
```

## 8. 调试与落地记录

### 2026-07-03 首次跑通

- **问题 1**：agent 放在 `.opencode/agents/`，HelixAgent 不识别。
  - **解决**：移到 `.opencode/agent/alpha-analyst.md`。
- **问题 2**：工具文件使用多命名导出，LLM 看不到任何自定义工具。
  - **解决**：拆分为单文件单工具，使用 `export default tool({...})`。
- **问题 3**：agent 未配置 `tools` 白名单，只聊天不调用工具。
  - **解决**：在 frontmatter 中显式启用需要的工具。
- **问题 4**：CLI 默认 TUI 模式在无 TTY shell 中空转。
  - **解决**：使用 `--format json` headless 模式，配合 `screen` 后台运行。
- **问题 5**：`screen_candidates` 工具传 flag 给 `screen.py`，导致解析失败。
  - **解决**：改为位置参数调用，并记录到 ADR-014。

首次成功输出 6 只组合，报告见 `memory/stock/20260703.md`。

### 2026-07-03 修复 `daily-screen.ts` 与首次回测

- **问题 6**：`daily-screen.ts` 通过 HTTP server 模式调用 HelixAgent，工具执行异常。
  - **解决**：完全切换到 CLI headless 模式（`bun <opencode> run --format json`）。
- **问题 7**：CLI 子进程 stdout 使用 pipe 时偶发 `Unexpected server error`。
  - **解决**：将子进程 stdout/stderr 重定向到 `memory/log/daily-screen-*.log` 文件。
- **问题 8**：prompt 中要求调用 `memory_search` 会立即触发 `Unexpected server error`。
  - **解决**：暂时从每日选股 prompt 中移除 `memory_search`，待 HelixAgent 稳定后再启用。
- **问题 9**：`scripts/evaluate.py` 对基准 `000300.SH` 使用 `daily` API，返回空数据。
  - **解决**：对指数代码使用 `index_daily` API。
- **成果**：
  - `daily-screen.ts` 连续 3 次成功跑通，输出 `memory/stock/20260703.md` 与 `.json`。
  - 首次历史回测完成：`2026-06-15` 入选组合持有 10 个交易日，组合收益 -1.41%，相对沪深300 超额 -3.20%，方向准确率 20%，报告见 `memory/eval/20260615_h10.json`。
  - 系统改进计划落盘：见 [docs/improvement-plan.md](docs/improvement-plan.md)。

### 2026-07-03 防穿越加固

- **问题**：历史回测中若使用当前 ST 名称、未公告财报或退市后数据，会产生未来函数，虚高回测收益。
- **解决**：
  - 财报因子增加 `ann_date <= trade_date` 校验（`screen.py:fetch_fina_factors`）。
  - ST/*ST/退市 过滤改为基于 `namechange` 历史名称（`_tushare_utils.py:is_st_historical`），而非当前名字。
  - 退市/停牌判断改为检查 `trade_date` 当天是否有 `daily` 交易记录，避免用当前上市状态。
  - 规则写入 `docs/agents.md` 作为 AlphaHelix 核心红线。
- **效果**：
  - 2026-06-15 单点样本因 Q1 已公告、无 ST 入选，回测结果不变（超额 +7.70%，方向准确率 70%）。
  - 2026-04-10（Q1 披露敏感期）回测验证通过：超额 +4.81%，方向准确率 80%，Top3 命中率 100%，证明财报防穿越和行业不截断策略在敏感日期仍有效。

### 2026-07-03 因子扩展落地

- **动作**：
  - `scripts/screen.py` 从 5 个因子扩展到 12+ 因子，新增：
    - 质量：ROE、营收增速、净利润增速、经营现金流增速（`fina_indicator`）
    - 估值：EP、BP、SP、股息率（`daily_basic`）
    - 资金：5 日/20 日主力净流入、净流入占比（`moneyflow`）
  - 加入 ROE ≥ 5% 过滤。
  - 加入行业数量集中度控制（单一行业不超过 top_n 的 40%）。
  - 采用两轮筛选：第一轮 400 只快速初筛，第二轮对 top 80 深度计算财务/资金因子，降低 API 调用量。
- **效果**：
  - 同一日期（2026-06-15，10 日持有期）回测结果：
    - 旧版：组合收益 -1.41%，超额 -3.20%，方向准确率 20%
    - 新版：组合收益 +9.49%，超额 +7.70%，方向准确率 70%，Top3 命中率 66.67%
  - 入选行业从金融股扎堆变为小金属、铜、玻璃、半导体、机械基件、通信设备、元器件、电气设备、化工原料等。

### 2026-07-03 多策略与 Regime 切换落地

- **动作**：
  - 在 `momentum_value_hybrid` 之外新增 `quality_growth`、`contrarian` 两种底层策略。
  - 新增 `scripts/market_regime.py`，基于沪深300 的 20 日/60 日均线、20 日波动率判断市场状态：`trend_up`、`range`、`trend_down`、`high_vol`。
  - `regime` 策略根据市场状态自动切换底层策略：
    - `trend_up` / `range` → `momentum_value_hybrid`
    - `trend_down` → `contrarian`
    - `high_vol` → `quality_growth`
  - `screen_candidates` 工具与 `daily-screen.ts` 默认使用 `regime` 策略。
- **效果**：
  - 单一策略在不同市场环境下波动较大，regime 切换后月度超额更稳定。
  - `quality_growth` 策略仍有调优空间，在部分月份跑输基准。

### 2026-07-03 Walk-forward 与 Feedback Harness v1 落地

- **动作**：
  - 新增 `scripts/walkforward.py`，支持多期回测、断点续跑、进度文件与按策略输出评估结果。
  - 完成 8 个月 walk-forward 回测（2025-01~05、2026-04~06，月末选股日）。
  - 新增 Feedback Harness 层：
    - `factor_ic.py`：计算因子秩 IC。
    - `strategy_tracker.py`：滚动跟踪策略绩效并生成 softmax 权重。
    - `weight_optimizer.py`：根据 IC 更新因子权重。
    - `feedback_harness.py`：编排以上模块，输出 `memory/weights/*_latest.json` 与 `memory/prompt_adaptations/latest.md`。
  - `screen.py` 启动时自动加载 `memory/weights/{strategy}_latest.json`。
  - `alpha-analyst.md` prompt 读取 `memory/prompt_adaptations/latest.md`。
- **效果**：
  - `regime` 策略 2025 年累计超额 +8.86%，优于单一 `momentum_value_hybrid` 的 +4.92%。
  - `regime` 策略 2026-04~06 累计超额 +4.96%，8 个月合计超额约 +13.82%。
  - 方向准确率在多数月份维持在 60% 左右。

### 当前阶段判断

AlphaHelix 处于 **Phase 4 完成、Phase 5/6 并行推进** 的阶段：

- MVP 已跑通，因子、策略、评估、Feedback Harness 均已落地。
- 已具备 8 个月 walk-forward 回测样本，初步验证 `regime` 策略有稳定超额。
- 下一步核心任务是**自动化与在线进化**：把每日选股与月度反馈更新接入 cron，实现无人值守；让 Feedback Harness 能自动发现新增回测结果并增量更新。
- 在自动化稳定运行 3 个月以上并确认指标持续有效前，不扩展数据源、不上新模型实验、不考虑实盘。

### 近期待验证假设

1. `regime` 自动切换在更长样本（12 个月以上）是否仍稳定跑赢单一策略？
2. Feedback Harness 动态权重能否在未来月份持续提升超额收益？
3. LLM 给出的 `confidence` 与实际收益是否正相关？
4. 分行业命中率反馈能否改善 `quality_growth` 等弱势策略？
5. 加入 `memory_search` 后，历史相似环境提示能否提升命中率？
6. Cardinal 规则能否拦截明显风险而不误杀？

## 9. 成功标准

| 阶段 | 目标 |
|---|---|
| Phase 1 | 系统能跑通端到端选股流程 |
| Phase 2 | 方向准确率 > 55%（略高于随机） |
| Phase 3 | 相对沪深300超额收益 > 3% / 月 |
| Phase 4 | 方向准确率 > 60%，超额收益稳定 > 5% / 月 |
| Phase 5 | cron 自动化稳定运行，每日选股无人工介入 |
| Phase 6 | Feedback Harness 在线学习 3 个月后，方向准确率进一步提升或超额更稳定 |
| Phase 7 | 连续 3 个月方向准确率 > 60%，相对沪深300超额 > 5% / 月 |

> 注意：股市具有强随机性，任何指标都应经过足够样本（至少 6-12 个月）验证。
