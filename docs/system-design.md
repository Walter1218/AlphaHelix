# AlphaHelix 全局系统方案

> 本文档是 AlphaHelix 的 master 设计蓝图，覆盖数据、因子、策略、风控、Agent、评估、Feedback Harness、自动化 7 层，明确各层职责、接口、数据流与演进路线。Feedback Harness 只是其中的 L6 一层。

---

## 1. 系统愿景与目标

### 1.1 愿景

打造一套**数据驱动、可验证、可进化**的 A 股选股智能体：

- 用 Python 处理所有数值计算，保证确定性与可审计性。
- 用 LLM 做定性推理与报告生成，不直接参与打分。
- 用 walk-forward 与 Feedback Harness 持续优化，而非依赖固定参数。
- 严格禁止未来函数，所有回测可被复现。

### 1.2 核心目标

| 指标 | 当前 | 3 个月目标 | 6 个月目标 |
|---|---|---|---|
| 月度方向准确率 | 56% | ≥ 58% | ≥ 60% |
| 月度超额收益（相对沪深300） | +1.34% | > 1.5% | > 2% |
| 单期最大回撤 | -9.16% | < -7% | < -6% |
| 回测样本覆盖 | 8 个月 | ≥ 12 个月 | ≥ 18 个月 |
| 人工干预频率 | 每次选股/回测 | 每周 1 次 review | 每月 1 次 review |

### 1.3 设计原则

1. **分层解耦**：每层只依赖相邻层的稳定接口。
2. **可回测**：所有策略、权重、参数变更必须经 walk-forward 验证。
3. **防穿越**：T 日决策只能用 T 日及之前已公开数据。
4. **可审计**：每次选股、每次权重更新、每次评估都有可追溯的 JSON/Markdown 产物。
5. **渐进进化**：不追求一次性完美，而是通过 Feedback Harness 持续迭代。

---

## 2. 总体架构

### 2.1 七层架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        L7 自动化/运维层                          │
│   cron / daily-screen.ts / 日志 / 告警 / 监控面板               │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                     L6 Feedback Harness 层                       │
│   factor IC / strategy tracker / weight optimizer / prompt adapter │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                        L5 评估层                                 │
│   evaluate.py / walkforward.py / 指标计算 / 交易成本模拟        │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                       L4 Agent 执行层                            │
│   alpha-analyst.md / HelixAgent / 工具调用 / 报告生成           │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                        L3 风控层                                 │
│   ST/退市过滤 / 流动性过滤 / 行业集中度 / 止损 / 业绩雷拦截     │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                      L2 因子/策略层                              │
│   screen.py / 多策略 / market_regime.py / 动态权重加载          │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                        L1 数据层                                 │
│   Tushare 工具 / JSON 缓存 / 数据校验 / 防穿越                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 层间接口

| 上游 | 下游 | 接口形式 | 关键产物 |
|---|---|---|---|
| L1 数据层 | L2 因子/策略层 | Python 函数 + DataFrame | 原始行情、财务、资金数据 |
| L2 因子/策略层 | L3 风控层 | DataFrame + 候选列表 | 带因子值的候选池 |
| L3 风控层 | L4 Agent 执行层 | JSON snapshot | `memory/stock/{date}.json` |
| L4 Agent 执行层 | L5 评估层 | JSON snapshot + 持仓记录 | `memory/stock/{date}.json` |
| L5 评估层 | L6 Feedback Harness 层 | eval JSON | `memory/eval/{date}_*_h{h}.json` |
| L6 Feedback Harness 层 | L2 因子/策略层 | 权重 JSON | `memory/weights/{strategy}_latest.json` |
| L6 Feedback Harness 层 | L4 Agent 执行层 | prompt 自适应提示 | `memory/prompt_adaptations/latest.md` |
| L7 自动化/运维层 | 所有层 | cron / 脚本调度 | 日志、告警、状态报告 |

---

## 3. 核心子系统设计

### 3.1 L1 数据采集与治理子系统

#### 3.1.1 职责

- 提供稳定、低延迟、可缓存的 A 股数据接口。
- 保证数据时序正确，严防未来函数。
- 对缺失、异常数据做降级处理。

#### 3.1.2 组件

| 组件 | 文件 | 说明 |
|---|---|---|
| Tushare 工具集 | `.opencode/tool/tushare_*.ts` | LLM 可调用的单文件单工具 |
| 数据访问层 | `scripts/_tushare_utils.py` | 统一封装 Tushare API、缓存、限流 |
| 缓存管理 | `.cache/tushare/` | 按接口+参数做 JSON 缓存 |

#### 3.1.3 数据分类与更新频率

| 数据类别 | 接口 | 缓存周期 | 用途 |
|---|---|---|---|
| 指数行情 | `index_daily` | 1 天 | regime 判断、基准收益 |
| 个股日线 | `daily` | 1 天 | 价格因子、评估 |
| 个股估值 | `daily_basic` | 1 天 | 估值因子 |
| 财务指标 | `fina_indicator` | 90 天 | 质量/成长因子 |
| 资金流向 | `moneyflow` | 1 天 | 资金因子 |
| 融资融券 | `margin`（待接入） | 1 天 | 杠杆情绪 |
| 北向资金 | `moneyflow_hsgt`（待接入） | 1 天 | 外资情绪 |
| 龙虎榜 | `top_list` / `top_inst`（待接入） | 1 天 | 游资/机构异动 |
| 行业分类 | `stock_basic` / `stock_company`（待接入） | 30 天 | 行业分散、轮动 |
| 财报披露 | `disclosure_date`（待接入） | 7 天 | 防业绩雷 |

#### 3.1.4 防穿越规则

- 财报：只使用 `ann_date <= trade_date` 的已披露报告。
- ST/退市：用 `namechange` 历史名称判断，不用当前名字。
- 行业：当前 `industry` 仅用于报告展示，不做历史量化截断。
- 价格：T 日收盘价或开盘价作为买入价，T+H 日收盘价作为卖出价。

### 3.2 L2 因子工厂与策略引擎

#### 3.2.1 因子工厂

所有因子在 `scripts/screen.py` 中本地计算，分为 5 大类：

| 类别 | 因子示例 | 数据源 |
|---|---|---|
| 动量/技术 | `mom_20`, `mom_60`, `volatility_20` | `daily` |
| 估值 | `ep`, `bp`, `sp`, `dividend_yield` | `daily_basic` |
| 质量/成长 | `roe`, `profit_growth`, `revenue_growth`, `ocf_growth` | `fina_indicator` |
| 资金 | `net_mf_5d`, `net_mf_20d`, `net_mf_ratio` | `moneyflow` |
| 流动性/规模 | `avg_amount_20`, `total_mv` | `daily` + `daily_basic` |

#### 3.2.2 策略引擎

当前支持 3 个策略 + 1 个 regime 调度器：

| 策略 | 适用环境 | 权重特点 |
|---|---|---|
| `momentum_value_hybrid` | 趋势向上或震荡市 | 动量 35%，估值 30%，质量 20%，资金 15% |
| `quality_growth` | 财报季、震荡市 | 质量 45%，成长 25%，估值 15%，资金 10% |
| `contrarian` | 大盘急跌后 | 估值 40%，质量 25%，反向动量 -15%，资金 10% |
| `regime` | 自动选择 | 根据 `market_regime.py` 输出切换 |

#### 3.2.3 动态权重

- 硬编码权重作为 baseline。
- `screen.py` 启动时检测 `memory/weights/{strategy}_latest.json`，存在则覆盖。
- 权重更新公式：`new_weight = old_weight * (1 + lr * IC)`，保持正负权重和归一化。

### 3.3 L3 风控中心

#### 3.3.1 已落地规则

| 规则 | 实现 | 动作 |
|---|---|---|
| ST/*ST/退市 | `is_st_historical` + `namechange` | 过滤 |
| 次新股 | `list_date` < trade_date - 120 天 | 过滤 |
| 低流动性 | `avg_amount_20 < 5000 万` | 过滤 |
| 高波动 | `volatility_20 > 7%` | 过滤（momentum 策略） |
| 财报穿越 | `ann_date <= trade_date` | 过滤 |
| 行业集中度 | 单一行业入选数量 ≤ top_n * 40% | 截断 |

#### 3.3.2 待落地规则

| 规则 | 实现位置 | 动作 |
|---|---|---|
| 行业市值权重控制 | `screen.py` | 单一行业市值权重 ≤ 40% |
| 业绩预亏/暴雷 | `disclosure_date` + `fina_indicator` | 过滤或标记高风险 |
| 高杠杆/高波动叙事 | Cardinal / agent prompt | 拦截或提示 |
| 大盘急跌空仓 | `market_regime.py` + agent | 减仓/空仓建议 |

### 3.4 L4 Agent 执行框架

#### 3.4.1 角色

`alpha-analyst` agent：

- 读取 `memory/prompt_adaptations/latest.md` 获取最新风格/风险提示。
- 调用 `screen_candidates`（默认 `strategy=regime`）获取候选池。
- 对 Top 候选调用 Tushare 工具做定性分析。
- 生成 Top-K 组合，包含 `score`、`rationale`、`confidence`、`stop_loss`。
- 写入 `memory/stock/{date}.md` 与 `memory/stock/{date}.json`。

#### 3.4.2 Prompt 工程

Prompt 分三部分：

1. **静态角色与输出格式**：始终不变。
2. **动态 feedback 提示**：从 `memory/prompt_adaptations/latest.md` 加载。
3. **当日上下文**：`trade_date`、最新市场 summary、候选股列表。

#### 3.4.3 置信度校准

- 统计 `high/medium/low` 三组命中率。
- 若 `high` 命中率 < 60%，收紧 `high` 标准。
- 若 `low` 命中率 > 50%，放宽筛选或检查因子方向。

### 3.5 L5 评估与回测平台

#### 3.5.1 单次评估

`scripts/evaluate.py` 读取 `memory/stock/{date}.json`，计算：

- 组合收益、超额收益、方向准确率、Top3 命中率
- 最大回撤、置信度相关性

#### 3.5.2 Walk-forward 回测

`scripts/walkforward.py`：

- 月频复调，覆盖多个月份。
- 支持 `--strategy`、`--universe-size`、`--skip-st-check`、`--progress-file`。
- 输出 `memory/eval/walkforward_*.json`。

#### 3.5.3 待增强

- 加入交易成本：0.1% 单边印花税 + 0.02% 双边佣金 + 滑点。
- 扩展样本至 12+ 个月。
- 分行业命中率报告。

### 3.6 L6 Feedback Harness 层

#### 3.6.1 职责

把 L5 的评估结果转化为 L2 的权重调整和 L4 的 prompt 自适应提示。

#### 3.6.2 组件

| 组件 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `factor_ic.py` | snapshot + eval | factor IC | 因子排名与收益排名的 Spearman 相关 |
| `strategy_tracker.py` | walkforward summaries | 策略配置权重 | softmax 滚动收益+命中率 |
| `weight_optimizer.py` | IC 序列 | 新权重 JSON | `new_weight = old_weight * (1 + lr * IC)` |
| `feedback_harness.py` | 回测日期区间 | 权重 + prompt 提示 | 一键编排 |

#### 3.6.3 Prompt 自适应示例

- 近期 `ocf_growth` IC 高 → rationale 中可侧重现金流改善。
- 近期 `net_mf_5d` IC 为负 → 降低对主力资金流入的依赖。
- 近期最大回撤 > 8% → 建议降低仓位、收紧止损。

### 3.7 L7 自动化/运维层

#### 3.7.1 定时任务

```bash
# 交易日 15:30 选股
30 15 * * 1-5 cd /Users/onetwo/Documents/trae_projects/AlphaHelix && bun run scripts/daily-screen.ts

# 每月第一个交易日 09:00 更新 feedback harness
0 9 1 * * cd /Users/onetwo/Documents/trae_projects/AlphaHelix && python scripts/feedback_harness.py --auto

# 每周一 08:00 生成周报
0 8 * * 1 cd /Users/onetwo/Documents/trae_projects/AlphaHelix && python scripts/generate_weekly_report.py
```

#### 3.7.2 监控指标

| 指标 | 采集方式 | 告警条件 |
|---|---|---|
| daily-screen 是否成功 | 检查 `memory/stock/{date}.json` 是否存在 | 超过 1 小时未生成 |
| 最近一期超额收益 | 读取 `memory/eval/` | 连续 2 期为负 |
| 最近一期最大回撤 | 读取 `memory/eval/` | 单期 < -10% |
| API 调用失败率 | 日志统计 | 超过 5% |
| 缓存命中率 | 日志统计 | 低于 80% |

---

## 4. 数据流与状态管理

### 4.1 日常选股数据流

```
cron (15:30)
  ↓
daily-screen.ts
  ↓
HelixAgent CLI → alpha-analyst agent
  ↓
read memory/prompt_adaptations/latest.md
  ↓
screen_candidates(strategy=regime, trade_date=today, top_n=10)
  ↓
screen.py 加载 memory/weights/{strategy}_latest.json
  ↓
market_regime.py 判断 regime → 选择具体策略
  ↓
风控过滤（ST/流动性/行业集中度）
  ↓
Top-N 候选池返回给 agent
  ↓
agent 做定性分析、生成报告
  ↓
write memory/stock/{date}.md + .json
```

### 4.2 反馈进化数据流

```
持有期结束（如 10 交易日后）
  ↓
evaluate.py 计算收益与风险指标
  ↓
memory/eval/{date}_*_h{h}.json
  ↓
feedback_harness.py --auto 扫描新增 eval
  ↓
factor_ic.py → memory/factor_ic/
strategy_tracker.py → memory/strategy_tracker/
weight_optimizer.py → memory/weights/*_latest.json
prompt adapter → memory/prompt_adaptations/latest.md
  ↓
下一次选股使用新权重与新 prompt 提示
```

### 4.3 状态持久化

| 状态 | 存储位置 | 更新频率 |
|---|---|---|
| 选股快照 | `memory/stock/{date}.json` | 每日 |
| 选股报告 | `memory/stock/{date}.md` | 每日 |
| 评估结果 | `memory/eval/{date}_*_h{h}.json` | 持有期结束 |
| 因子 IC | `memory/factor_ic/` | 每期回测后 |
| 策略权重 | `memory/strategy_tracker/` | 每期回测后 |
| 因子权重 | `memory/weights/*_latest.json` | 每期回测后 |
| prompt 自适应 | `memory/prompt_adaptations/latest.md` | 每期回测后 |
| 数据缓存 | `.cache/tushare/` | 按需 |

---

## 5. 接口契约

### 5.1 `screen.py` 接口

```bash
python scripts/screen.py <strategy> <trade_date> <top_n>
```

- `strategy`: `momentum_value_hybrid` | `quality_growth` | `contrarian` | `regime`
- `trade_date`: YYYYMMDD
- `top_n`: 返回候选数

输出：JSON 数组，每个元素包含 `ts_code`, `name`, `industry`, `total_score`, 各因子值。

### 5.2 `evaluate.py` 接口

```bash
python scripts/evaluate.py <date> <horizon>
```

输入：`memory/stock/{date}.json`
输出：JSON 对象，包含组合收益、超额收益、方向准确率、Top3 命中率、最大回撤、置信度相关性、明细。

### 5.3 `walkforward.py` 接口

```bash
python scripts/walkforward.py \
  --start <YYYYMMDD> \
  --end <YYYYMMDD> \
  --strategy <strategy> \
  --horizon <days> \
  --top-n <n> \
  [--universe-size <n>] \
  [--skip-st-check] \
  [--no-resume] \
  [--progress-file <path>]
```

输出：`memory/eval/walkforward_{start}_{end}_{strategy}_h{horizon}.json`

### 5.4 `feedback_harness.py` 接口

```bash
python scripts/feedback_harness.py \
  --dates <date1,date2,...> \
  --start <YYYYMMDD> \
  --end <YYYYMMDD> \
  --horizon <days> \
  --strategy <strategy> \
  [--lr <float>]
```

输出：
- `memory/factor_ic/{date}_pooled_h{h}.json`
- `memory/strategy_tracker/weights_{start}_{end}_h{h}.json`
- `memory/weights/{strategy}_latest.json`
- `memory/prompt_adaptations/latest.md`

---

## 6. 部署与运行模式

### 6.1 开发模式

手动运行单个脚本，快速验证：

```bash
python scripts/screen.py regime 20260702 10
python scripts/evaluate.py 20260615 10
python scripts/walkforward.py --start 20250101 --end 20250531 --strategy regime --horizon 10 --top-n 10
```

### 6.2 生产模式

通过 cron 自动运行，人工只需定期 review：

```bash
# 选股
crontab -e
30 15 * * 1-5 cd /path/to/AlphaHelix && bun run scripts/daily-screen.ts

# 评估与反馈
0 9 1 * * cd /path/to/AlphaHelix && python scripts/feedback_harness.py --auto
```

### 6.3 复盘模式

每月初运行一次完整 walk-forward，生成报告：

```bash
python scripts/walkforward.py --start 20240101 --end 20260630 --strategy regime --horizon 10 --top-n 10 --universe-size 200
python scripts/feedback_harness.py --auto
```

---

## 7. 演进路线

### Phase 1：MVP（已完成）

- HelixAgent + Tushare 工具链可用
- 单次选股可写入 memory

### Phase 2：因子、策略与 Regime（已完成）

- 12+ 因子
- 三策略 + regime 切换
- `market_regime.py`

### Phase 3：风控与 Memory（部分完成）

- ST/退市、流动性、财报防穿越已落地
- 行业集中度数量控制已落地
- `memory_search` 待 HelixAgent 修复

### Phase 4：评估与回测（已完成）

- `evaluate.py`
- `walkforward.py`
- 8 个月回测

### Phase 5：自动化调度（1-2 周）

- cron 配置
- 日志轮转
- 失败告警

### Phase 6：Feedback Harness 在线化（2-3 周）

- `--auto` 模式
- 分行业命中率反馈
- 置信度校准
- 参数网格搜索

### Phase 7：数据补齐与高级策略（3-4 周）

- 融资融券、北向资金、龙虎榜、行业分类工具
- 行业轮动
- 宏观 regime 指标

### Phase 8：模型与记忆实验（未来）

- DPO 数据集导出
- 模型微调
- memory_search 修复后深度接入

---

## 8. 风险与兜底

### 8.1 技术风险

| 风险 | 影响 | 兜底方案 |
|---|---|---|
| Tushare 免费接口限流 | 选股超时 | 缓存 + 夜间预取 + universe-size 200 |
| HelixAgent `memory_search` 不可用 | 损失历史经验复用 | 用 Feedback Harness 替代部分记忆功能；等待官方修复 |
| 数据质量问题 | 回测失真 | 多数据源交叉校验 + 异常值过滤 |
| 模型幻觉 | 错误推荐 | 所有数值计算由 Python 完成，LLM 只生成 rationale |

### 8.2 业务风险

| 风险 | 影响 | 兜底方案 |
|---|---|---|
| 过拟合 | 历史表现好，未来失效 | 样本外验证、参数扰动、交易成本扣除 |
| 风格切换 | 某个月大幅亏损 | 多策略 regime 切换 + 风控减仓 |
| 黑天鹅事件 | 系统性亏损 | 明确责任声明：我们对研究方法负责，但不承诺收益 |

---

## 9. 验收指标

### 9.1 单元验收

| 组件 | 验收命令 | 通过标准 |
|---|---|---|
| `screen.py` | `python scripts/screen.py regime 20260702 10` | 输出 10 只候选，无报错 |
| `evaluate.py` | `python scripts/evaluate.py 20260615 10` | 输出 JSON 评估报告 |
| `walkforward.py` | `python scripts/walkforward.py --start 20250101 --end 20250531 --strategy regime --horizon 10 --top-n 10` | 输出 5 期汇总 |
| `feedback_harness.py` | `python scripts/feedback_harness.py --auto` | 更新 weights 与 prompt 自适应文件 |

### 9.2 系统验收

- 连续 1 个月每日选股无失败。
- 最近 3 个月平均方向准确率 ≥ 55%。
- 最近 3 个月累计超额收益 > 0%。
- 单期最大回撤 < -10%。

---

## 10. 相关文档

- [architecture.md](architecture.md)：模块级架构与数据流
- [roadmap.md](roadmap.md)：Phase 与多轨道路线图
- [improvement-plan.md](improvement-plan.md)：详细改进计划与立即可执行动作
- [agents.md](agents.md)：alpha-analyst agent 设计与约束
- [decisions.md](decisions.md)：架构决策记录（ADR）
