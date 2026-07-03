# AlphaHelix Code Review 2026-07-03

> 本次 review 针对 MVP 阶段代码实现质量、准确性与可维护性。

## 严重问题（已修复）

### 1. 快照日期格式不一致 ✅

- **位置**：`.opencode/agent/alpha-analyst.md`、`scripts/evaluate.py`、`scripts/walkforward.py`、`scripts/screen.py`
- **问题**：agent prompt 和部分文档使用 `memory/stock/YYYY-MM-DD.json` 示例，但脚本实际使用 `YYYYMMDD.json`；仓库中还存在 `memory/stock/2026-07-03.json` 与 `20260703.json` 并存。
- **修复**：统一使用 `YYYYMMDD` 格式，更新 agent prompt 和相关文档，删除不一致的 `2026-07-03.json`。
- **提交**：`8c840b4`

### 2. `cap_sector_weight` 是死代码 ✅

- **位置**：`scripts/screen.py:506`
- **问题**：定义了行业数量集中度截断函数，但 `screen()` 中从未调用。
- **修复**：在 `screen()` 中 `df_pass2.head(top_n)` 前调用 `cap_sector_weight(df_pass2, top_n, max_pct=MAX_SECTOR_PCT)`。
- **提交**：`3de89bd`

### 3. 新因子未进入 IC 计算 ✅

- **位置**：`scripts/factor_ic.py`、`scripts/walkforward.py`
- **问题**：`FACTOR_FIELDS` 和 `build_snapshot` 的因子字段列表仍是旧版本，未包含 `mom_5`、`reversal_score`、`sector_momentum`、`forecast_type_score` 等。
- **修复**：在两个文件中同步扩展因子字段列表。
- **提交**：`8e2a3d9`

## 中等问题（已修复）

### 4. `feedback_harness.py` 硬编码日期区间 ✅

- **位置**：`scripts/feedback_harness.py:153`
- **问题**：硬编码 `("20250101", "20250531")` 和 `("20260401", "20260615")`。
- **修复**：直接加载 `args.start`/`args.end` 区间的策略汇总，删除硬编码合并逻辑。
- **提交**：`0b3fb59`

### 5. `reversal_score` 公式设计存疑 ✅

- **位置**：`scripts/screen.py:163`
- **问题**：`reversal_score = -mom_20 * (1 + mom_5) * amount_ratio_5d`，当 `mom_5` 为负时会压低分数。
- **修复**：改为 `reversal_score = -mom_20 * amount_ratio_5d`；在 contrarian pass2 中显式加入 `mom_5` 权重。
- **提交**：`e1408c5`

### 6. `_tushare_utils.py` 模块级 token 加载 ✅

- **位置**：`scripts/_tushare_utils.py:15-20`
- **问题**：导入时就检查 `TUSHARE_TOKEN` 并调用 `ts.set_token`。
- **修复**：使用 `_get_pro()` 延迟初始化，模块导入不再依赖环境变量。
- **提交**：`3cb828a`

### 7. `is_delisted_historical` 未使用 ✅

- **位置**：`scripts/_tushare_utils.py:104-108`
- **问题**：函数已定义但无调用方。
- **修复**：删除该函数；同时放宽 `get_trade_date_before/after` 的日历缓冲天数。
- **提交**：`d8143a4`

## 轻微问题 / 已知限制

### 8. `evaluate.py` 未复权、无交易成本

- 短期影响较小，长期会失真。已在 roadmap 中列为待办。

### 9. JSON 输出含 `NaN`

- `json.dumps` 默认输出 `NaN`，非严格 JSON。HelixAgent 和 Python 端可消费，但与其他工具集成可能报错。
- 建议：将 NaN 替换为 `null`。

### 10. `alpha-analyst.md` 未提及 `event_driven`

- prompt 中说 regime 只在 momentum/quality/contrarian 间切换，需更新为四策略。

### 11. `get_trade_date_before/after` 缓冲天数偏保守

- `days * 2 + 30` 在超长假期前后可能不够。建议放大到 `days * 3 + 60` 或循环获取。

### 12. `build_universe` 用当前 `list_status='L'` 构建历史股票池

- 轻微未来函数：历史上已退市但当前不在 list_status 的股票会被排除。对回测影响有限，因为 `pass1` 还会用价格存在性过滤。

## 修复状态

- ✅ P0：统一快照日期格式（#1）— `8c840b4`
- ✅ P0：新因子进入 IC 计算（#3）— `8e2a3d9`
- ✅ P1：调用 `cap_sector_weight`（#2）— `3de89bd`
- ✅ P1：去掉 `feedback_harness.py` 硬编码日期（#4）— `0b3fb59`
- ✅ P2：优化 `reversal_score` 公式（#5）— `e1408c5`
- ✅ P2：延迟加载 Tushare token（#6）— `3cb828a`
- ✅ P3：清理 `is_delisted_historical` 并放宽日历缓冲（#7）— `d8143a4`

## 剩余建议（后续按需处理）

1. JSON `NaN` 输出改为严格 `null`
2. `evaluate.py` 加入复权与交易成本
3. `alpha-analyst.md` 已更新提及 `event_driven`
4. `build_universe` 当前 `list_status='L'` 的轻微历史偏差（已有价格存在性二次过滤）

## 做得好的地方

- 反穿越逻辑扎实：`ann_date <= trade_date`、历史名称查 ST、退市用当天价格校验。
- 缓存与限流机制完整。
- 策略/权重/提示三层反馈闭环结构清晰。
- 事件因子加了 120 天 freshness 过滤。
- 行业因子明确标注了当前行业分类的历史局限性。
