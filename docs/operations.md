# AlphaHelix 运维手册

> 本手册面向部署与维护 AlphaHelix 的运行环境，涵盖 cron 配置、日志查看、缓存管理、故障排查与日常巡检。

## 1. 运行环境要求

- **OS**：macOS / Linux（当前开发环境为 macOS）
- **Node/Bun**：`bun` 已安装并可执行
- **Python**：3.9+，已安装 `tushare`、`pandas`、`numpy`
- **HelixAgent**：本地已克隆并配置好 `.mimo` 目录
- **Tushare Token**：写入 `.env` 的 `TUSHARE_TOKEN`

## 2. 目录约定

```
AlphaHelix/
├── memory/
│   ├── stock/                  # 选股报告与快照
│   ├── eval/                   # 评估与回测结果
│   ├── weights/                # 动态因子权重
│   ├── prompt_adaptations/     # prompt 自适应提示
│   └── log/                    # 运行日志
├── .cache/tushare/             # Tushare API 本地缓存
└── .env                        # 环境变量
```

## 3. 手动运行流程

### 3.1 每日选股

```bash
cd /path/to/AlphaHelix
bun run scripts/daily-screen.ts
```

成功标志：
  - `memory/stock/YYYYMMDD.md` 已生成
  - `memory/stock/YYYYMMDD.json` 已生成
- `memory/log/daily-screen-*.log` 无 `Unexpected server error` 或 Python 异常

### 3.2 历史评估

```bash
# 评估 10 个交易日前（20260602）的选股
python scripts/evaluate.py 20260602 10
```

输出：`memory/eval/20260602_h10.json`

### 3.3 Walk-forward 回测

```bash
python scripts/walkforward.py \
  --start 20250101 --end 20250630 \
  --strategy regime --horizon 10 --top-n 10 --universe-size 200
```

输出：`memory/eval/walkforward_regime_20250101_20250630_h10.json`

### 3.4 Feedback Harness

```bash
python scripts/feedback_harness.py \
  --dates 20250127,20250228,20250331,20250430,20250530,20260430,20260529,20260615 \
  --start 20250101 --end 20260615 --horizon 10 --strategy regime
```

输出：
- `memory/weights/momentum_value_hybrid_latest.json`
- `memory/weights/quality_growth_latest.json`
- `memory/weights/contrarian_latest.json`
- `memory/prompt_adaptations/latest.md`

> **注意**：当前 Feedback Harness 仍需手动指定 `--dates`。`--auto` 在线学习模式实现后，可接入 cron 自动增量更新。

## 4. Cron 自动化（Phase 5）

### 4.1 交易日每日选股

```bash
crontab -e
```

添加（以 macOS 为例，工作日 15:30 执行）：

```cron
30 15 * * 1-5 cd /path/to/AlphaHelix && /usr/local/bin/bun run scripts/daily-screen.ts >> memory/log/cron-daily-screen.log 2>&1
```

> 若 `bun` 路径不同，请用 `which bun` 确认。

### 4.2 历史评估（可选）

```cron
0 9 * * 1-5 cd /path/to/AlphaHelix && /usr/local/bin/python3 scripts/evaluate.py $(date -v-20d +%Y%m%d) 20 >> memory/log/cron-evaluate.log 2>&1
```

> Linux 用户将 `date -v-20d +%Y%m%d` 替换为 `date -d '20 days ago' +%Y%m%d`。

### 4.3 月度 Feedback Harness（手动日期版）

在每月第一个交易日收盘后手动执行：

```bash
cd /path/to/AlphaHelix
# 更新 dates 为上个月末各选股日
python scripts/feedback_harness.py \
  --dates <逗号分隔日期> \
  --start <区间起点> --end <区间终点> \
  --horizon 10 --strategy regime
```

待 `--auto` 模式实现后，可改为 cron：

```cron
0 18 1 * * cd /path/to/AlphaHelix && /usr/local/bin/python3 scripts/feedback_harness.py --auto >> memory/log/cron-feedback.log 2>&1
```

## 5. 日志管理

### 5.1 日志位置

| 日志 | 路径 | 说明 |
|---|---|---|
| 每日选股日志 | `memory/log/daily-screen-YYYYMMDD-<timestamp>.log` | HelixAgent 子进程 stdout/stderr |
| Cron 汇总日志 | `memory/log/cron-*.log` | cron 任务输出 |
| Feedback Harness | `memory/log/feedback_harness-*.log` | 权重更新与 prompt 生成日志 |
| Walk-forward | `memory/log/walkforward-*.log` | 多期回测日志 |

### 5.2 日志轮转

建议每月清理一次超过 90 天的日志：

```bash
find memory/log -name "*.log" -mtime +90 -delete
```

或配置 `logrotate`（Linux）/ `newsyslog`（macOS）。

## 6. 缓存管理

Tushare 数据缓存位于 `.cache/tushare/`。首次运行较慢，后续命中缓存会显著提速。

### 6.1 清理缓存

当发现历史数据异常或接口字段变更时：

```bash
rm -rf .cache/tushare
```

### 6.2 缓存大小监控

```bash
du -sh .cache/tushare
```

## 7. 常见故障排查

### 7.1 `Unexpected server error`

**现象**：HelixAgent 子进程报错 `Unexpected server error`。

**可能原因与处理**：
- prompt 中包含 `memory_search`：暂时移除该工具调用。
- 子进程 stdout 使用 pipe：已改为重定向到日志文件，参见 `daily-screen.ts`。
- HelixAgent 服务端异常：检查是否有多个 HelixAgent 进程冲突，但不要杀死用户正在使用的 TUI 进程。

### 7.2 Tushare 限流

**现象**：API 返回 `freq limited` 或请求明显变慢。

**处理**：
- 免费版 Tushare 限流约 1 次/秒，系统已内置 sleep 与缓存。
- 避免在短时间内手动高频调用 walk-forward。
- 可临时增大 `scripts/_tushare_utils.py` 中的 `sleep_time`。

### 7.3 选股结果为空

**检查项**：
- `.env` 中 `TUSHARE_TOKEN` 是否正确。
- `trade_date` 是否为交易日。
- `screen.py` 的 universe-size 是否过小导致过滤后无标的。
- 查看 `memory/log/daily-screen-*.log` 中的 Python  traceback。

### 7.4 回测结果异常偏高/偏低

**检查项**：
- 是否使用了未来数据（检查 `ann_date`、`trade_date` 边界）。
- 是否用当前 ST/退市状态过滤了历史股票池。
- 是否混用了指数代码与个股代码的 API（`daily` vs `index_daily`）。

### 7.5 权重文件未生效

**检查项**：
- `memory/weights/{strategy}_latest.json` 是否存在且为有效 JSON。
- `screen.py` 启动日志是否打印 `Loaded dynamic weights from ...`。
- 权重键名是否与 `screen.py` 中定义的因子名称一致。

## 8. 安全与纪律

- **禁止杀死用户的 TUI HelixAgent 进程**。
- **禁止在代码或日志中硬编码 TUSHARE_TOKEN**。
- **禁止在回测中使用未来数据**：任何改动后必须重新跑完整 walk-forward。
- **不要把 `memory/log/` 下的日志提交到 git**：已在 `.gitignore` 中排除。

## 9. 日常巡检清单

| 频率 | 操作 | 命令/路径 |
|---|---|---|
| 每日 | 检查最新选股报告 | `ls -lt memory/stock/*.md | head -5` |
| 每日 | 检查选股日志是否有报错 | `tail -n 50 memory/log/daily-screen-*.log` |
| 每周 | 检查磁盘占用 | `du -sh .cache/tushare memory/log` |
| 每月 | 运行 Feedback Harness 更新权重 | `python scripts/feedback_harness.py ...` |
| 每月 | 清理过期日志 | `find memory/log -mtime +90 -delete` |
| 每季 | 重新跑完整 walk-forward 验证策略 | `python scripts/walkforward.py ...` |

## 10. 升级与变更

修改 `screen.py`、因子权重、策略逻辑或数据过滤规则后：

1. 本地跑通单期选股：`python scripts/screen.py regime 20260702 10`
2. 跑完整 walk-forward：`python scripts/walkforward.py --start 20250101 --end 20260630 ...`
3. 对比新/旧指标，确认无未来函数与明显退化。
4. 更新 `docs/decisions.md` 记录 ADR。
5. 再提交代码或部署到 cron。
