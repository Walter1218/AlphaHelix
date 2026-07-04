# AlphaHelix Code Review - 2026-07-04

## 1. 本次 review 范围

针对 2026-07-02 ~ 2026-07-04 新增/改动的核心模块：

- `scripts/walkforward.py`：在线学习、周度频率、分数阈值、IC 校准、pass2 权重覆盖
- `scripts/screen.py`：并发加载、按日期截面加载日线/资金流、动态权重加载
- `scripts/_tushare_utils.py`：并发、数据窗口隔离
- `scripts/prefetch_data.py`：批量预取
- `scripts/calibrate_weights_from_ic.py`：历史 IC 权重校准
- `scripts/online_weight_updater.py`：在线滚动权重更新
- `docs/AGENTS.md`、`docs/risk.md`、`docs/decisions.md`、`docs/roadmap.md`

## 2. 发现的问题

### 2.1 时间穿越/样本内优化风险（严重）

**问题**：`screen.py` 在回测时会自动回退到 `memory/weights/{strategy}_latest.json`。`latest.json` 通常由 `feedback_harness.py` 基于一段历史日期生成；如果 walk-forward 区间与该历史区间重叠，`latest.json` 就可能包含未来权重，导致时间穿越。

**影响**：walk-forward 绩效失真，可能把 in-sample 优化结果误当作有效结果。

**修复**：
- `walkforward.py` 设置 `AH_BACKTEST_MODE=1`。
- `screen.py` 的 `load_dynamic_weights` 在该模式下直接返回 `None`，不加载任何动态权重。
- walk-forward 在线学习/IC 校准通过显式 `pass2_weights_override` 传入权重，避免自动回退。

### 2.2 `--pass2-weights` 容易被误用（中等）

**问题**：`walkforward.py --pass2-weights` 允许加载任意 JSON 作为 pass2 权重。用户可能直接加载 `calibrate_weights_from_ic.py` 生成的全样本 IC 权重，从而违反 AGENTS.md C38。

**修复**：
- 命令行增加警告，提示只能使用「回测期之前的数据」生成的权重。
- 兼容嵌套 JSON（`{"weights": {"pass2": {...}}}`），避免误传整个文件。

### 2.3 `calibrate_weights_from_ic.py` 未标注诊断属性（中等）

**问题**：脚本输出文件没有明显标识「样本内、不可用于同区间回测」，容易被误用。

**修复**：
- 脚本运行时打印醒目的 WARNING。
- 输出 JSON 增加 `"diagnostic_only": true` 和 `"warning"` 字段。
- 删除未使用的 `EVAL_DIR` 变量。

### 2.4 文档与约束清单不一致（中等）

**问题**：AGENTS.md 新增了 C38，但 `risk.md` 约束清单没有同步；ADR-040 中样本内结果未明确标注为诊断/违规上界。

**修复**：
- `risk.md` 0.1 节增加 C38，指向 AGENTS.md 2.5 节。
- ADR-040 表格下增加 ⚠️ 说明，明确「IC 权重（样本内）」违反 C38，只有 walk-forward 版本合规。

### 2.5 并发异常处理不严谨（低）

**问题**：`concurrent_map` 把未捕获的异常作为结果存入列表。当前 worker 函数内部已 try/except，风险可控，但若未来扩展可能踩坑。

**建议**：后续对 `concurrent_map` 结果统一过滤 `Exception` 实例，或让 `concurrent_map` 直接 re-raise。

### 2.6 回测缓存未命中（低）

**问题**：`evaluate.py` 仍按 `ts_code + start_date + end_date` 拉价格，而 `prefetch_data.py` 只预取了 `trade_date` 截面。回测时这部分会重新请求，但数量不大（每只股票 2 次），暂不影响正确性。

**建议**：后续把 `evaluate.py` 也改为按 `trade_date` 截面查缓存，进一步提速。

## 3. 已执行的修复

| 文件 | 修复内容 |
|---|---|
| `scripts/screen.py` | `load_dynamic_weights` 在 `AH_BACKTEST_MODE=1` 时返回 `None` |
| `scripts/walkforward.py` | 回测时设置 `AH_BACKTEST_MODE=1`；`--pass2-weights` 加警告并兼容嵌套 JSON |
| `scripts/calibrate_weights_from_ic.py` | 增加诊断警告、`diagnostic_only` 标记；删除未使用变量 |
| `docs/AGENTS.md` | 已存在 C38 与 2.5 节 |
| `docs/risk.md` | 同步增加 C38 |
| `docs/decisions.md` | ADR-040 明确标注样本内结果为诊断/违规上界 |

## 4. 仍待改进

1. **`--pass2-weights` 进一步约束**：可考虑要求权重文件带 `"walk_forward": true` 元数据才允许加载。
2. **`feedback_harness.py` 产出管理**：`{strategy}_latest.json` 应在文件名或元数据中标注生成日期/截止日期，避免生产环境误用过期权重。
3. **`concurrent_map` 异常处理**：统一过滤或抛出，避免静默混入异常对象。
4. **`evaluate.py` 截面缓存**：与 `screen.py` 一致，按 `trade_date` 读取预取数据。
5. **更严格的 out-of-sample 流程**：训练集固定为 2024 年，测试集固定为 2025-2026 年，避免多次迭代测试集导致信息泄露。

## 5. 合规结论

修复后，默认 walk-forward 回测不再自动加载可能含未来数据的动态权重；显式权重覆盖有警告；诊断性 IC 权重文件有明显标记；文档约束一致。满足 AGENTS.md C38 与 C01 的基本要求。
