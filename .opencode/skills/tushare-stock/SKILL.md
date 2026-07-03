---
name: tushare-stock
description: A-share quantitative stock selection using Tushare data and HelixAgent evolution
---

# Tushare 智能选股 Skill

## 目标

基于 Tushare 金融数据，选出未来 1 个月（约 20 个交易日）相对沪深 300 具有超额收益潜力的 A 股组合。

## 数据源与接口

| 数据维度 | Tushare 接口 | 用途 |
|---|---|---|
| 股票基础信息 | `stock_basic` | 过滤 ST、退市、次新股 |
| 日线行情 | `daily` | 计算动量、波动率、成交额 |
| 每日指标 | `daily_basic` | 获取 PE、PB、换手率、总市值 |
| 财务指标 | `fina_indicator` | ROE、毛利率、营收增速、净利润增速 |
| 资金流向 | `moneyflow` | 主力净流入、散户资金流向 |
| 指数行情 | `index_daily` | 沪深300、上证指数贝塔参考 |
| 交易日历 | `trade_cal` | 获取真实交易日，避免未来函数 |
| 新闻/公告 | `news` / `major_news` | 事件驱动与情绪分析 |

## 选股 SOP

### 第一步：初筛（本地计算）

调用 `screen_candidates` 工具，策略 `momentum_value_hybrid`：

1. 剔除 ST、*ST、上市不足 120 日、近 20 日日均成交额 < 5000 万的股票
2. 计算因子：
   - 20 日动量 = (close_t / close_t-20) - 1
   - 60 日动量
   - 20 日波动率（日收益率标准差）
   - 近 20 日平均成交额
   - 最新 PE、PB、总市值
3. 综合打分：动量 40% + 估值 30% + 质量（规模）20% + 流动性 10%
4. 返回 Top 50 候选

### 第二步：定性分析（LLM 推理）

对候选股逐个分析：

1. 调用 `tushare_daily` 获取近 60 日价格走势
2. 调用 `tushare_fina_indicator` 获取最新季度财务指标
3. 调用 `tushare_moneyflow` 获取近 5 日资金流向
4. 使用 `memory` 工具检索历史相似市场环境下的选股结果
5. 结合当前宏观与行业背景（可用 webfetch 抓取公开研报）

### 第三步：精选组合

输出 JSON 格式：

```json
{
  "date": "20260702",
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

### 第四步：风控检查

- 单行业集中度不超过 40%
- 避免推荐 ST、*ST、退市整理期股票
- 避免日均成交额 < 3000 万的标的
- 每只推荐股票必须给出 stop_loss

### 第五步：记忆写入

使用 `write` 工具同时写入两个文件（`memory` 工具目前只支持检索）：

- `memory/stock/YYYY-MM-DD.md`：人类可读的选股报告
- `memory/stock/YYYY-MM-DD.json`：机器可读的选股快照，供后续回测

## 未来函数禁忌

- 严禁使用选股日之后的收盘价做决策
- 财务数据以公告日（ann_date）为准，不可用未披露的财报
- 资金流数据必须是 T-1 及之前
- 使用 `tushare_trade_cal` 获取真实交易日，不要用自然日估算

## 评估与进化

选股后 1 个月（20 个交易日）：

1. 调用 `evaluate_picks` 工具或运行 `scripts/evaluate.py` 计算实际收益
2. 指标：方向准确率、Top3 命中率、相对沪深300超额收益、最大回撤、置信度相关性
3. 将高命中率 trace 标记为 chosen，低命中率且逻辑有缺陷的标记为 rejected
4. 定期导出 DPO 数据集，优化 prompt 与模型

## 责任声明

本 Skill 对研究方法、数据准确性与回测过程的严谨性负责，并通过持续迭代优化模型。但证券市场受宏观环境、政策变化、市场情绪等不可控因素影响，模型输出不代表对未来收益的保证。使用者应结合自身判断审慎决策，过往表现不代表未来收益。
