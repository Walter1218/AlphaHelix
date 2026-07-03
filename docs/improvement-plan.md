# AlphaHelix 改进计划

> 基于 Tushare 数据扩展、因子增强、多策略 ensemble、风控落地与 Feedback Harness 反馈驱动进化的系统性改进路线图。

## 0. AlphaHelix 全局分层视图

Feedback Harness 只是 AlphaHelix 进化体系中的一个层级。全局上，系统由以下 7 层构成，每层独立演进、相互支撑：

| 层级 | 核心职责 | 已落地 | 下一步重点 |
|---|---|---|---|
| **L1 数据层** | 稳定、高效、低延迟地获取 Tushare 数据；缓存与防穿越 | `tushare_*.ts` 工具集、`_tushare_utils.py` JSON 缓存、`ann_date` 校验 | 补齐行业、融资融券、北向资金、龙虎榜等数据源 |
| **L2 因子/策略层** | 本地计算多维度因子；多策略 ensemble；regime 切换 | `screen.py` 18+ 因子；`momentum_value_hybrid` / `quality_growth` / `contrarian` / `event_driven`；`market_regime.py`；`regime` 自动切换 | 继续调优 `quality_growth`；引入行业轮动与宏观 regime 指标 |
| **L3 风控/ Cardinal 层** | 拦截不合格标的；控制行业集中度；止损与仓位建议 | 历史 ST/退市过滤、流动性过滤、`ann_date` 防穿越、数量级行业集中度 | 市值权重控制、业绩预亏/暴雷拦截、高波动/高杠杆叙事拦截 |
| **L4 执行/Agent 层** | 调用 LLM 做定性分析、生成报告、写入 memory | `alpha-analyst.md`、`.opencode/tool/`、`daily-screen.ts` | prompt 工程、置信度校准、失败重试 |
| **L5 评估层** | 持有期后确定性计算收益与风险指标 | `evaluate.py`、`walkforward.py`、8 个月回测 | 加入交易成本、分行业命中率、更长样本 |
| **L6 Feedback Harness 层** | 用时效性结果反哺权重、策略配置与 prompt | `factor_ic.py`、`strategy_tracker.py`、`weight_optimizer.py`、`feedback_harness.py`、动态权重加载 | 在线学习、分行业反馈、置信度校准、参数网格搜索 |
| **L7 自动化/运维层** | 无人值守调度、监控、告警、日志轮转 | `daily-screen.ts` 可手动运行 | cron 配置、失败通知、自动评估触发 |

**改进原则**：每层先独立验证，再通过接口与相邻层联动；不把所有优化都压在 Feedback Harness 上，也不让某一层长期阻塞整体进度。

## 1. 背景与当前状态

AlphaHelix 已从「能跑通单次选股」的 MVP，演进为具备多策略 regime 切换和 Feedback Harness 反馈层的选股系统。当前在 L1~L6 均有落地，L7 自动化尚未配置。

### 1.1 已验证

- 端到端选股 pipeline 已稳定：`daily-screen.ts` 可无人值守执行。
- `scripts/screen.py` 支持 `momentum_value_hybrid`、`quality_growth`、`contrarian` 三策略，以及 `regime` 自动切换。
- `scripts/market_regime.py` 可基于沪深300 判断市场状态。
- `scripts/evaluate.py` 可完成历史持有期收益评估，已修复指数 `000300.SH` 使用 `index_daily` 的问题。
- `scripts/walkforward.py` 已完成 8 个月历史回测（2025-01~05、2026-04~06）。
- **Feedback Harness v1 已落地**：
  - `factor_ic.py` 计算因子 rank IC。
  - `strategy_tracker.py` 滚动跟踪策略表现。
  - `weight_optimizer.py` 基于 IC 调整因子权重。
  - `feedback_harness.py` 一键编排并生成 prompt 自适应提示。
  - `screen.py` 自动加载 `memory/weights/*_latest.json`。
  - `alpha-analyst` prompt 读取 `memory/prompt_adaptations/latest.md`。

### 1.2 主要回测结果

> 统一参数：持有期 10 个交易日，`top_n=10`，`UNIVERSE_SAMPLE=200`，开启历史 ST 检查。

#### 1.2.1 单策略 baseline（`momentum_value_hybrid`）

| 选股日 | 组合收益 | 超额收益 | 方向准确率 | Top3 命中率 |
|---|---|---|---|---|
| 2025-01-27 | +1.87% | -0.64% | 60% | 33% |
| 2025-02-28 | +4.25% | +1.25% | 80% | 67% |
| 2025-03-31 | -3.37% | -0.13% | 10% | 33% |
| 2025-04-30 | +1.74% | -1.09% | 60% | 33% |
| 2025-05-30 | +6.21% | +5.33% | 80% | 100% |
| 2026-04-30 | +2.21% | +1.26% | 50% | 33% |
| 2026-05-29 | -9.16% | -6.82% | 20% | 0% |
| 2026-06-15 | +13.43% | +11.64% | 80% | 100% |

**8 期汇总**：平均组合收益 +2.66%，平均超额收益 +1.34%，方向准确率 56.3%。

#### 1.2.2 多策略对比（2025-01 至 2025-05）

| 策略 | 平均超额收益 | 方向准确率 | 累计超额收益 |
|---|---|---|---|
| momentum_value_hybrid | +0.95% | 58.0% | +4.92% |
| quality_growth | +0.10% | 58.0% | +0.17% |
| contrarian（新公式） | +0.67% | 76.0% | +3.65% |
| event_driven | **+2.14%** | 56.0% | **+11.60%** |
| regime（range→momentum, trend_down→contrarian） | +1.67% | 66.0% | +8.86% |

#### 1.2.3 多策略对比（2026-04 至 2026-06）

| 策略 | 平均超额收益 | 方向准确率 | 累计超额收益 |
|---|---|---|---|
| momentum_value_hybrid | +2.03% | 50.0% | +4.96% |
| quality_growth | +0.09% | 36.7% | +0.27% |
| contrarian（新公式） | -3.47% | 33.3% | -10.35% |
| event_driven | **+3.38%** | 43.3% | **+8.99%** |
| regime（实际全部使用 momentum） | +2.03% | 50.0% | +4.96% |

**结论**：
- `event_driven` 在两个区间均显著跑赢其他单一策略，累计超额 2025 年 +11.60%、2026 Q2 +8.99%，成为当前最强 baseline。
- `contrarian` 新公式在 2025 年表现弱于旧公式（+0.67% vs +1.17%），2026 Q2 仍显著亏损，说明纯反转因子在当前样本下适应性有限。
- `regime` 切换 2026 Q2 因 classifier 将市场判为 trend_up/range，实际等同于 momentum，未享受到 event_driven 的收益。

**2025-01 至 2025-05 汇总（5 期）**：
- 平均组合收益：+2.14%
- 平均超额收益：+0.95%
- 平均方向准确率：58.0%
- 累计组合收益：+10.88%
- 累计超额收益：+4.92%

### 1.3 各层主要问题与下一步方向

按全局分层梳理：

**L1 数据层**
- 缺少融资融券、北向资金、龙虎榜、行业分类等工具，限制了情绪与杠杆维度。
- 下一步：补齐 `tushare_margin`、`tushare_northbound`、`tushare_top_list`、`tushare_industry`。

**L2 因子/策略层**
- `quality_growth` 策略在回测中持续弱于 `momentum_value_hybrid`。
- `regime` 分类对市场急跌反应滞后（2026-05 被归为 range，实际大跌）。
- 下一步：调优 `quality_growth` 权重与触发条件；引入 VIX 近似、融资余额、行业动量扩散等更灵敏的 regime 指标。

**L3 风控层**
- 行业集中度仅做数量控制，未按市值权重截断。
- 缺少业绩预亏/暴雷拦截、高波动/高杠杆叙事拦截。
- 历史 ST 过滤已验证有效，需保持。

**L4 执行/Agent 层**
- `memory_search` 因 HelixAgent 环境问题暂时禁用，损失历史经验复用。
- 置信度（high/medium/low）与实际收益的关系尚未校准。
- 下一步：修复/绕过 memory 问题；统计置信度命中率并校准 agent 标准。

**L5 评估层**
- 回测样本仅 8 个月，未覆盖熊市、高波动、结构性行情。
- 未扣除交易成本，真实收益会略低。
- 下一步：扩展回测区间至 12+ 个月；加入印花税、佣金、滑点。

**L6 Feedback Harness 层**
- 当前需手动运行并传入 `--dates`，未实现在线增量更新。
- 缺少分行业命中率反馈、置信度校准、参数网格搜索。
- 下一步：实现 `--auto` 模式；补充分行业与置信度反馈。

**L7 自动化/运维层**
- `daily-screen.ts` 与 `feedback_harness.py` 均未接入 cron。
- 无失败通知与日志轮转。
- 下一步：配置 cron、添加告警、日志轮转。

---

## 2. 数据层扩展

### 2.1 目标

把当前仅依赖 `daily` + `daily_basic` 的数据层，扩展到覆盖量价、估值、财务、资金、情绪、行业六大类数据。

### 2.2 新增 Tushare 接口

| 数据类别 | Tushare API | 字段示例 | 用途 |
|---|---|---|---|
| 指数行情 | `index_daily` | `close`, `pct_chg`, `amount` | 市场 regime、基准收益 |
| 个股估值 | `daily_basic` | `pe`, `pb`, `ps`, `dv_ratio`, `total_mv`, `turnover_rate` | 估值、规模、流动性 |
| 财务指标 | `fina_indicator` | `roe`, `roa`, `grossprofit_margin`, `netprofit_yoy`, `tr_yoy`, `ocf_yoy` | 质量、成长 |
| 资金流向 | `moneyflow` | `net_mf_amount`, `buy_lg_amount`, `sell_lg_amount` | 主力动向 |
| 龙虎榜 | `top_list` | `pct_change`, `amount`, `reason` | 游资/机构异动 |
| 龙虎榜机构 | `top_inst` | `net_buy`, `trans_type` | 机构净买入 |
| 北向资金 | `moneyflow_hsgt` | `ggt_ss`, `ggt_sz`, `hgt` | 外资流向 |
| 融资融券 | `margin` | `rzye`, `rqye`, `rzrqye` | 杠杆情绪 |
| 涨跌停 | `limit_list` | `close`, `pct_chg`, `fd_amount` | 市场情绪 |
| 行业分类 | `stock_basic` / `stock_company` | `industry`, `fullname` | 行业分散、行业轮动 |
| 指数成分 | `index_weight` | `con_code`, `weight` | 基准权重参考 |
| 财报披露 | `disclosure_date` | `end_date`, `ann_date` | 避免业绩雷 |

### 2.3 工具层实现

新增/扩展以下 `.opencode/tool/` 文件：

- `tushare_industry.ts`：获取个股所属行业。
- `tushare_margin.ts`：融资融券余额。
- `tushare_northbound.ts`：北向资金流向。
- `tushare_top_list.ts`：龙虎榜数据。
- `tushare_fina_indicator.ts`：已有，扩展为支持季度滚动查询。
- `tushare_moneyflow.ts`：已有，扩展为支持多日汇总。

### 2.4 缓存策略

`_tushare_utils.py` 已支持 JSON 缓存。财务/行业类低频数据可长期缓存；资金/情绪类日频数据缓存 1-7 天；量价类数据按日期缓存。

新增数据接口后，预计缓存文件数量会快速增长。建议：

- 定期清理超过 90 天的日频缓存（保留财务/行业缓存）。
- 对 `daily_basic` 等全市场接口，按 `trade_date` 做单文件缓存，避免按股票代码重复请求。
- 监控 `.cache/tushare/` 目录大小，必要时迁移到 SQLite。

---

## 3. 因子层增强

### 3.1 目标

- **已完成**：从 5 个因子扩展到 6 大类 18+ 因子（动量、估值、质量、资金、事件、反转、行业相对强度），落地到 `scripts/screen.py`（2026-07-03）。
- **下一阶段**：优化资金流动量（ratio）与 5日/20日背离、range 市场下 contrarian 权重动态提升、调优 quality_growth。

> **状态**：18+ 因子已落地（2026-07-03）。首次回测样本（2026-06-15，10 日持有期）方向准确率从 20% 提升到 70%，超额收益从 -3.20% 提升到 +7.70%。

### 3.2 因子体系

#### 3.2.1 动量/技术类

| 因子名 | 公式 | 说明 |
|---|---|---|
| `mom_5` | `close_t / close_{t-5} - 1` | 超短期动量/转折 |
| `mom_20` | `close_t / close_{t-20} - 1` | 短期动量 |
| `mom_60` | `close_t / close_{t-60} - 1` | 中期动量 |
| `mom_120` | `close_t / close_{t-120} - 1` | 长期动量 |
| `risk_adj_mom` | `mom_20 / volatility_20` | 波动率调整动量 |
| `relative_strength` | `mom_20_stock / mom_300_index` | 相对沪深300 强度 |
| `amount_ratio_5d` | `avg_amount_5 / avg_amount_20` | 近期成交额相对 20 日放量比 |
| `reversal_score` | `-mom_20 * amount_ratio_5d` | 超跌且近期放量反弹得分 |
| `sector_momentum` | 行业成分股平均 mom_20 | 行业相对动量 |
| `relative_to_sector` | `mom_20 - sector_momentum` | 个股相对行业动量 |
| `sector_mom5` | 行业成分股平均 mom_5 | 行业超短期动量 |
| `sector_amount_ratio` | 行业成分股平均 amount_ratio_5d | 行业近期放量比 |

#### 3.2.2 估值类

| 因子名 | 公式 | 说明 |
|---|---|---|
| `ep` | `1 / pe` | 盈利收益率 |
| `bp` | `1 / pb` | 账面市值比 |
| `sp` | `1 / ps` | 营收市值比 |
| `dividend_yield` | `dv_ratio` | 股息率 |
| `peg_proxy` | `pe / max(netprofit_yoy, 0.05)` | PEG 近似，负增速标为高风险 |

#### 3.2.3 质量/成长类

| 因子名 | 公式 | 说明 |
|---|---|---|
| `roe` | `fina_indicator.roe` | 净资产收益率 |
| `roa` | `fina_indicator.roa` | 总资产收益率 |
| `grossprofit_margin` | `fina_indicator.grossprofit_margin` | 毛利率 |
| `revenue_growth` | `fina_indicator.tr_yoy` | 营收同比增速 |
| `profit_growth` | `fina_indicator.netprofit_yoy` | 净利润同比增速 |
| `ocf_growth` | `fina_indicator.ocf_yoy` | 经营现金流增速 |

#### 3.2.4 资金/情绪类

| 因子名 | 公式 | 说明 |
|---|---|---|
| `net_mf_5d` | `sum(net_mf_amount, 5)` | 5 日主力净流入 |
| `net_mf_20d` | `sum(net_mf_amount, 20)` | 20 日主力净流入 |
| `net_mf_ratio` | `net_mf_5d / amount_5d` | 主力净流入占比 |
| `northbound_delta` | `近 5 日持股变化` | 北向资金变化（待接入） |
| `margin_delta` | `rzye 5 日变化` | 融资余额变化（待接入） |
| `top_inst_net` | `龙虎榜机构净买入` | 机构异动（待接入） |

#### 3.2.5 事件类（已落地）

| 因子名 | 公式 | 说明 |
|---|---|---|
| `forecast_type_score` | `forecast.type` 映射得分 | 业绩预告类型（预增/扭亏等） |
| `forecast_pchange_mid` | `(p_change_min + p_change_max) / 2` | 业绩预告预增幅度中值 |
| `express_diluted_roe` | `express.diluted_roe` | 业绩快报 ROE |

> **防穿越与 freshness**：事件因子严格使用 `ann_date <= trade_date` 的数据，且只取最近 120 天内公告，避免用旧预告误导当前决策。

### 3.3 财务数据披露期处理

`fina_indicator` 为季度数据，T 日选股时应使用**最近已披露报告期**，避免未来函数。

**已实现做法**：不预设定报告期，而是查询该股票全部历史财报，过滤 `ann_date <= trade_date`，取 `ann_date` 最近的一份。

```python
def fetch_fina_factors(ts_code: str, trade_date: str) -> dict:
    df = tushare_call("fina_indicator", {"ts_code": ts_code})
    df["ann_date"] = df["ann_date"].fillna("").astype(str)
    df = df[df["ann_date"] <= trade_date]  # 只使用已公告财报
    if df.empty:
        return {}
    row = df.sort_values("ann_date", ascending=False).iloc[0]
    return { ... }
```

若某股票在 `trade_date` 前无已披露财报，则该股票的财务类因子置为 NaN，不参与该维度打分。

### 3.4 因子打分方法

对每个因子在当日截面上做 **rank 标准化**（0~1），再按权重合成总分：

```python
score = (
    w_mom   * rank(momentum_factors) +
    w_value * rank(value_factors) +
    w_quality * rank(quality_factors) +
    w_fund   * rank(fund_factors) +
    w_liquid * rank(liquidity_factors)
)
```

权重初始值：

- `momentum_value_hybrid`：动量 45%，估值 30%，质量 15%，流动性 10%
- `quality_growth`：质量 50%，成长 30%，估值 10%，流动性 10%
- `contrarian`：估值 40%，质量 30%，动量 -20%（反向），情绪 10%

---

## 4. 多策略与市场 Regime

### 4.1 策略定义

```python
STRATEGIES = {
    "momentum_value_hybrid": {
        "description": "趋势向上或震荡时，买入低估值高动量蓝筹",
        "pass1": {"filters": {"min_amount": 50000, "max_volatility": 0.07},
                    "weights": {"mom_20": 0.30, "mom_60": 0.20, "ep": 0.15, "bp": 0.15, "size": 0.10, "liquidity": 0.10}},
        "pass2": {"filters": {"min_roe": 0.05},
                    "weights": {"mom_20": 0.25, "mom_60": 0.10, "ep": 0.10, "bp": 0.10, "sp": 0.05,
                                "roe": 0.10, "profit_growth": 0.05, "revenue_growth": 0.05,
                                "net_mf_5d": 0.10, "net_mf_ratio": 0.05, "liquidity": 0.05}}
    },
    "quality_growth": {
        "description": "高波动或财报季，买入高 ROE + 高成长标的",
        "pass1": {"filters": {"min_amount": 50000, "max_volatility": 0.08},
                    "weights": {"mom_20": 0.20, "mom_60": 0.10, "ep": 0.10, "bp": 0.10, "size": 0.05,
                                "liquidity": 0.15, "sp": 0.15, "dividend": 0.15}},
        "pass2": {"filters": {"min_roe": 0.08, "min_profit_growth": 0.10},
                    "weights": {"roe": 0.25, "profit_growth": 0.20, "revenue_growth": 0.10, "ocf_growth": 0.10,
                                "ep": 0.10, "bp": 0.05, "net_mf_5d": 0.05, "net_mf_ratio": 0.05,
                                "liquidity": 0.05, "mom_20": 0.05}}
    },
    "contrarian": {
        "description": "大盘急跌后，买入超跌但基本面稳健标的",
        "pass1": {"filters": {"min_amount": 30000, "max_volatility": 0.10, "max_mom_60": 0.05},
                    "weights": {"bp": 0.25, "ep": 0.20, "sp": 0.10, "dividend": 0.10, "liquidity": 0.15,
                                "mom_20": -0.10, "mom_60": -0.05}},
        "pass2": {"filters": {"min_roe": 0.03},
                    "weights": {"bp": 0.25, "ep": 0.20, "roe": 0.15, "profit_growth": 0.10, "net_mf_5d": 0.10,
                                "mom_20": -0.10, "mom_60": -0.05, "liquidity": 0.05}}
    }
}
```

### 4.2 Regime 判断规则

基于沪深300 日线计算（实现见 `scripts/market_regime.py`）：

| Regime | 条件 | 默认映射策略 |
|---|---|---|
| `trend_up` | 20 日均线向上，近 20 日涨幅 > 5%，波动率 < 5% | `momentum_value_hybrid` |
| `range` | 20 日均线走平，波动率 5%~10% | `momentum_value_hybrid`（后续拟提升 contrarian 权重） |
| `trend_down` | 20 日均线向下，近 20 日跌幅 > 8% 或单日大跌 > 3% | `contrarian` |
| `high_vol` | 20 日年化波动率 > 25% | `quality_growth` |

实现位置：`scripts/market_regime.py`，每日由 `daily-screen.ts` 调用并写入报告。

---

## 5. 与 `daily-screen.ts` 的集成

`daily-screen.ts` 当前 prompt 只需做最小改动：

1. 让 agent 先调用 `market_regime` 工具判断当前 regime。
2. 根据 regime 选择策略，再调用 `screen_candidates(strategy=<selected>, trade_date=<date>, top_n=10)`。
3. agent 使用扩展后的 Tushare 工具获取行业、资金、财务数据，生成报告。
4. 报告新增字段：
   - `regime`: 当前市场状态
   - `strategy`: 使用的策略
   - `sector_weights`: 行业权重分布
   - `position_sizes`: 每只标的建议仓位

伪代码：

```typescript
const prompt = `执行 AlphaHelix 每日选股：
1. 调用 market_regime(trade_date=${today}) 判断市场状态。
2. 根据 regime 选择策略（trend_up→momentum_value_hybrid, range→quality_growth, down→contrarian）。
3. 调用 screen_candidates(strategy=<selected>, trade_date=${today}, top_n=15) 获取候选。
4. 对候选股获取行业、财务、资金流向，生成 Top-10 组合。
5. 每只股票给出：代码、名称、得分、排名、逻辑、置信度、止损价、建议仓位。
6. 确保单一行业权重 ≤ 40%，单票权重 ≤ 20%。
7. 写入 ${jsonPath} 和 ${mdPath}。
8. 不要调用 memory 工具。`
```

---

## 6. 风控与 Cardinal 规则

### 5.1 目标

把风险拦截从 prompt 描述落地到代码，确保不符合条件的股票不会进入组合。

### 5.2 规则清单

| 规则 | 实现位置 | 动作 |
|---|---|---|
| 剔除 ST/*ST/退市 | `screen.py` | 过滤 |
| 剔除上市 < 120 日 | `screen.py` | 过滤 |
| 剔除日均成交额 < 5000 万 | `screen.py` | 过滤 |
| 剔除 20 日波动率 > 15% | `screen.py` | 过滤 |
| 剔除 ROE < 5% | `screen.py`（质量策略） | 过滤 |
| 剔除净利润增速连续两季度为负 | `screen.py`（质量策略） | 过滤 |
| 单一行业权重 ≤ 40% | agent 定性检查 | 报告中提示风险 |
| 单一标的权重 ≤ 20% | `screen.py` 后处理 / agent 检查 | 截断/提示 |
| 缺少止损价 | agent prompt | 禁止输出 |
| 业绩预亏/暴雷 | `disclosure_date` + `fina_indicator` | 剔除 |
| 连续涨停/一字板 | `limit_list` | 剔除或标记高风险 |
| 大盘高波动/急跌 | `market_regime.py` | 减仓或空仓 |

### 5.3 止损计算

```python
def stop_loss(close: float, atr14: float, low10: float) -> float:
    return max(close - 2 * atr14, low10 * 0.98)
```

### 5.4 仓位分配

按波动率反比分配：

```python
def position_size(volatility: float, max_risk: float = 0.02) -> float:
    weight = max_risk / volatility
    return min(weight, 0.20)  # 单票上限 20%
```

---

## 7. 评估与反馈闭环

### 7.1 Feedback Harness 层（新增）

为把时效性结果反哺回系统，新增 Feedback Harness：

```
┌──────────────┐    ┌─────────────┐    ┌─────────────────┐
│ walk-forward │───▶│ factor IC   │───▶│ weight optimizer│
│   results    │    │  strategy   │    │  prompt adapter │
└──────────────┘    │  tracker    │    └─────────────────┘
                    └─────────────┘             │
                            │                   │
                            ▼                   ▼
                    memory/strategy_tracker   memory/weights/*_latest.json
                    memory/factor_ic          memory/prompt_adaptations/latest.md
```

**组件**：

| 脚本 | 职责 | 输出 |
|---|---|---|
| `scripts/factor_ic.py` | 计算每个因子 rank IC | `memory/factor_ic/{date}_pooled_h{h}.json` |
| `scripts/strategy_tracker.py` | 滚动跟踪策略表现，softmax 输出配置权重 | `memory/strategy_tracker/weights_*.json` |
| `scripts/weight_optimizer.py` | 基于 IC 调整因子权重 | `memory/weights/{strategy}_latest.json` |
| `scripts/feedback_harness.py` | 一键运行上述流程 + 生成 prompt 自适应提示 | `memory/prompt_adaptations/latest.md` |

**权重更新规则**：

```python
new_weight = old_weight * (1 + lr * IC)
# 保持正向/负向权重各自的和不变，再做归一化
```

**prompt 自适应示例**（来自 `memory/prompt_adaptations/latest.md`）：

- 近期有效的因子（如 `ocf_growth` IC=+0.142）→ 在 rationale 中可侧重现金流改善逻辑。
- 近期失效的因子（如 `net_mf_5d` IC=-0.189）→ 降低对主力资金流入的依赖。
- 近期最大回撤 > 8% → 建议降低仓位、收紧止损。

**集成点**：

- `screen.py` 启动时自动加载 `memory/weights/{strategy}_latest.json`，覆盖硬编码权重。
- `alpha-analyst` prompt 要求先读取 `memory/prompt_adaptations/latest.md`。
- `daily-screen.ts` 的 prompt 已改用 `strategy=regime` 并提示读取 feedback 文件。

### 7.2 扩展 `evaluate.py`

已有指标：组合收益、超额收益、方向准确率、Top3 命中率、最大回撤、置信度相关性。

未来可新增：

- **分行业命中率**：识别哪些行业模型更准。
- **置信度校准**：`high/medium/low` 与实际收益的相关性。
- **夏普比率、Calmar 比率**：风险调整后收益。

### 7.3 Walk-forward 回测

`scripts/walkforward.py` 已实现，支持：

- `--start/--end`：回测区间（月频复调）。
- `--strategy`：当前仅 `momentum_value_hybrid`。
- `--horizon`：持有期交易日数。
- `--top-n`：每期入选数量。
- `--universe-size`：覆盖 `screen.py` 的 `UNIVERSE_SAMPLE`（默认 400，回测可用 200 加速）。
- `--skip-st-check`：关闭历史 ST 检查以加速，**仅用于快速实验，不能用于生产评估**。
- `--no-resume`：强制重新运行（默认会跳过已存在的快照/评估）。
- `--progress-file`：写出当前进度 JSON，便于长时间运行监控。

示例：

```bash
python scripts/walkforward.py \
  --start 20250101 \
  --end 20250531 \
  --strategy momentum_value_hybrid \
  --horizon 10 \
  --top-n 10 \
  --universe-size 200 \
  --progress-file memory/eval/walkforward_progress.json
```

输出示例：

```json
{
  "strategy": "momentum_value_hybrid",
  "periods": 6,
  "avg_direction_accuracy": 0.52,
  "avg_excess_return": 0.008,
  "max_drawdown": -0.12,
  "best_month": "202604",
  "worst_month": "202606",
  "factor_ic": {
    "mom_20": 0.03,
    "roe": 0.08,
    "net_mf_5d": 0.05
  }
}
```

### 7.4 因子权重自动优化

每月根据 IC 和超额收益调整权重：

```python
for factor in factors:
    ic = rolling_ic[factor]
    weights[factor] *= (1 + learning_rate * ic)
weights = weights / weights.sum()
```

### 7.5 DPO 数据集生成

新增 `scripts/generate_dpo.py`：

```python
{
  "prompt": "2026-06-15 市场环境：沪深300...",
  "chosen": "选股组合 A（未来收益高）",
  "rejected": "选股组合 B（未来收益低）"
}
```

保存到 `memory/dpo/`，未来用于模型微调。

### 7.6 测试方法论

为避免过拟合，所有改进必须按以下流程验证：

1. **单因子 IC 测试**：在 2026-01 至 2026-06 区间，计算每个因子的 rank IC 和 IR。
2. **样本外验证**：用 2026-01 至 2026-04 训练权重，用 2026-05 至 2026-06 验证。
3. **参数稳健性**：对权重、阈值做 ±20% 扰动，观察收益变化是否剧烈。
4. **交易成本**：回测中扣除 0.1% 单边印花税 + 0.02% 佣金。
5. **对照组**：始终保留原始 `momentum_value_hybrid` 作为 baseline，新策略必须显著跑赢 baseline 才合并。

### 7.7 记忆复用（待 HelixAgent 修复）

`memory_search` 修复后，选股前查询：

> "A股 2026-07 大跌 券商保险领涨 选股"

把最相似的 3 次历史案例注入 prompt，让 LLM 避免重复犯错。

---

## 8. 兼容性与迁移

1. **保留 `momentum_value_hybrid`**：作为 baseline 策略，因子权重和过滤条件不变。
2. **新增策略独立实现**：`quality_growth` 和 `contrarian` 作为新的 strategy 参数，不影响现有调用。
3. **工具向后兼容**：新增的 Tushare 工具不强制调用；agent prompt 可逐步启用。
4. **snapshot 格式兼容**：JSON 报告新增 `regime`, `strategy`, `sector_weights` 字段，旧字段保持不变。
5. **evaluate.py 兼容**：继续支持原始 snapshot 格式，新增字段仅用于增强分析。

---

## 9. 实施路线图

| 阶段 | 时间 | 关键任务 | 验收标准 |
|---|---|---|---|
| 1. 因子扩展 | 已完成 | 用现有 `daily_basic`/`fina_indicator`/`moneyflow` 扩展 12+ 因子 | `screen.py` 已输出质量、资金、估值因子；2026-06-15 方向准确率 80%，超额 +11.64% |
| 2. 防时间穿越加固 | 已完成 | 财报按 `ann_date` 动态取最近已披露期、历史 ST/退市 过滤、`trade_date` 交易记录校验、行业仅用于报告不做量化截断 | 规则写入 `docs/agents.md`；2025/2026 多期回测验证通过 |
| 3. 历史 ST 过滤 | 已完成 | `_tushare_utils.py:is_st_historical` + `namechange` | 关闭 ST 检查时回测显著恶化，证明其必要性 |
| 4. 回测自动化 | 已完成 | `walkforward.py` 支持月度回测、断点续跑、进度文件、universe-size、skip-st-check | 已产出 8 个月回测结果 |
| 5. 多策略 ensemble | 已完成 v1 | 新增 `quality_growth`, `contrarian`，`regime` 自动切换；`market_regime.py` 已落地 | regime 策略 2025 累计超额 +8.86%，优于 momentum 的 +4.92% |
| 6. Feedback Harness v1 | 已完成 | `factor_ic.py`、`strategy_tracker.py`、`weight_optimizer.py`、`feedback_harness.py`；`screen.py` 加载动态权重；prompt 自适应 | 可生成 `memory/weights/*_latest.json` 与 `memory/prompt_adaptations/latest.md` |
| 7. Feedback Harness 在线化 | 1-2 周 | 自动发现新增回测、增量更新权重、分行业命中率反馈、置信度校准 | 无需手动传 `--dates`，每月自动更新 |
| 8. 风控落地 | 1-2 天 | ROE 过滤（已完成）、行业集中度 ≤40%、止损/仓位计算、大盘急跌空仓 | 不符合规则的股票不会入选 |
| 9. 自动化调度 | 1 周 | cron 配置 `daily-screen.ts` 与 `feedback_harness.py`；日志轮转与告警 | 无人值守运行 |
| 10. 数据补齐 | 2-3 周 | 新增 `tushare_industry`, `margin`, `northbound`, `top_list` 工具 | 所有新接口可独立返回数据 |
| 11. 记忆复用 | 待定 | `memory_search` 修复后接入 | 选股前检索相似市场环境 |

---

## 10. 立即可执行的 9 个动作

### 10.1 L6 Feedback Harness：多目标离线权重优化（当前最高优先级）

**目标**：在方向准确率硬约束下，搜索使超额收益最大的因子权重组合。

**优化形式**：

```
maximize  avg_excess_return
subject to avg_direction_accuracy >= threshold（初始 threshold = 70%）
```

**实现思路**：
1. 复用已有 walk-forward 产物：`memory/stock/YYYYMMDD.json`（含因子值）和 `memory/eval/YYYYMMDD_{strategy}_h{horizon}.json`（含实际收益）。
2. 对每套策略的 pass2 权重做网格/随机搜索。
3. 对每一组权重，在历史截面上重新计算综合得分、选 top-n、计算该期组合收益与方向。
4. 跨期聚合后，筛选方向准确率 ≥ threshold 的组合，取超额收益最高者。
5. 输出 `memory/weights/{strategy}_mo_latest.json`，`screen.py` 优先加载。

**与当前 IC 优化的区别**：
- 当前 `weight_optimizer.py` 优化的是因子 rank IC，间接希望提升收益，但没有显式约束方向准确率。
- 多目标优化直接以「方向准确率 ≥ 70%」为硬约束，「超额收益」为唯一目标，更贴近实盘目标函数。

**注意事项**：
- 搜索空间随因子数量指数增长，先用随机搜索 + 约束筛选；若效果稳定，再引入贝叶斯优化或遗传算法。
- 必须保留 out-of-sample 区间验证，防止在 8 个月样本上过拟合。
- 不同市场 regime 应分别优化权重，而非全样本一套权重。

**首次运行结果（2026-07-03）**：
- `event_driven`：baseline avg_excess=+2.60%，dir_acc=50.0%；随机搜索 10,000 组 pass2 权重后，无组合达到 55% 方向准确率，最佳收益与 baseline 相同。
- `contrarian`：baseline avg_excess=-0.88%，dir_acc=50.0%；同样无组合达到 55% 方向准确率。
- **结论**：当前 8 期样本下，仅靠 pass2 权重调整无法将方向准确率提升到 55% 以上；baseline 权重对各自候选池已接近最优。要达到 70% 方向准确率，需要扩大样本、优化 pass1 权重、引入新因子，或采用 regime 条件优化。

### 10.2 L7 自动化：接入 cron

配置 cron 实现无人值守：

```bash
# 交易日 15:30 选股
30 15 * * 1-5 cd /path/to/AlphaHelix && bun run scripts/daily-screen.ts

# 每月第一个交易日 09:00 更新 feedback harness
0 9 1 * * cd /path/to/AlphaHelix && python scripts/feedback_harness.py --auto
```

同时添加日志轮转与失败通知（可选飞书/邮件）。

### 10.3 L6 Feedback Harness：在线化

让 `feedback_harness.py` 支持 `--auto` 模式：自动扫描 `memory/eval/` 中最新的 walk-forward 结果，识别新增日期，增量更新 `memory/weights/` 与 `memory/prompt_adaptations/latest.md`。

### 10.4 L6 Feedback Harness：分行业命中率反馈

新增 `scripts/sector_tracker.py`，计算最近 N 期每个行业的方向准确率与超额收益，输出到 `memory/sector_tracker/latest.json`，供 agent 调整行业配置。

### 10.5 L4 Agent：置信度校准

统计 agent 给出的 `high/medium/low` 置信度与实际收益的关系。若 `high` 置信度股票实际命中率 < 60%，在 prompt 中提示收紧「high」标准。

### 10.6 L5 评估：加入交易成本

在 `evaluate.py` 中扣除 0.1% 单边印花税 + 0.02% 双边佣金 + 滑点，使回测更接近真实收益。

### 10.7 L5 评估：扩展回测样本

运行 `walkforward.py` 覆盖 2024-2025 年更多月份，至少达到 12 个月以上样本，覆盖不同市场环境。

### 10.8 L3 风控：行业市值权重控制

在 `screen.py` 中按 `total_mv` 计算行业市值权重，单一行业权重超过 40% 时截断，替代当前的数量控制。

### 10.9 L2 策略：调优 quality_growth

针对 `quality_growth` 在回测中持续偏弱的问题，通过网格搜索调整其 pass1/pass2 权重与过滤阈值，或将其触发条件限制在财报密集披露期。

### 10.10 L4/L6：建立 AlphaHelix Trace 与 DPO 数据集

**现状**：当前没有全局 Trace 覆盖与持久化；只有最终选股结果、评估结果和运行日志。

**目标**：
1. 在关键节点（`screen.py` 选股、`evaluate.py` 评估、`feedback_harness.py` 更新权重、agent 决策）写入结构化 trace。
2. Trace 格式采用 JSONL，包含：timestamp、step、inputs、outputs、metadata（权重、策略、regime）。
3. 定期（每月）根据命中率把 trace 标记为 chosen/rejected，导出 DPO 数据集。

**输出**：
- `memory/trace/YYYYMMDD.jsonl`：单次选股全流程 trace。
- `memory/dpo/chosen.jsonl` / `rejected.jsonl`：用于未来微调模型。

---

## 11. 预期目标

| 层级 | 指标 | 当前 | 3 周后目标 |
|---|---|---|---|
| L5 评估 | 方向准确率 | 56%（8 个月平均） | ≥ 58% |
| L5 评估 | 月度超额收益 | +1.34%（8 个月平均） | > 1.5% |
| L5 评估 | 组合最大回撤 | 单期 -9.16% | < -7% |
| L5 评估 | 回测样本数 | 8 个月 | ≥ 12 个月 |
| L5 评估 | 交易成本 | 未扣除 | 已扣除印花税+佣金+滑点 |
| L3 风控 | 行业集中度 | 数量控制已落地 | 市值权重 ≤ 40% |
| L2 策略 | quality_growth 累计超额 | 接近 0% | > 2% |
| L6 Harness | 在线学习 | 手动运行 | `--auto` 增量更新 |
| L6 Harness | 分行业命中率反馈 | 无 | 覆盖主要行业 |
| L4 Agent | 置信度校准 | 无 | high 置信度命中率 ≥ 60% |
| L7 运维 | cron 调度 | 无 | 选股 + harness 均接入 |

---

## 12. 风险与注意事项

1. **过拟合风险**：因子越多、策略越多，越容易过拟合历史数据。必须保留 out-of-sample 验证。
2. **数据质量**：Tushare 免费接口有调用频次限制，需确保缓存有效。
3. **未来函数禁忌**：不能用 T+1 之后的数据计算 T 日因子；财报数据必须满足 `ann_date <= trade_date`；ST/退市状态必须查历史名称。
4. **行业分类偏差**：`stock_basic` 的 `industry` 为当前分类，报告中的行业分布可能与历史真实分布存在偏差，目前仅作展示和定性提示。
5. **市场环境变化**：A 股风格切换快，单一策略难以持续有效，必须做多策略 ensemble。
6. **合规责任声明**：我们对研究方法和数据质量负责，但不承诺收益，市场存在不确定性。
