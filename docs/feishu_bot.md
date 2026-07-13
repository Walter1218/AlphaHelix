# 飞书机器人配置

## 环境变量

在 `.env` 文件中添加以下配置：

```bash
# 飞书机器人配置
FEISHU_APP_ID=your_app_id
FEISHU_APP_SECRET=your_app_secret
FEISHU_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/your_webhook_id
FEISHU_WEBHOOK_SECRET=your_webhook_secret
```

## 使用方法

### 1. 推送文本消息

```bash
python scripts/feishu_bot.py --push "测试消息"
```

### 2. 推送告警

```bash
python scripts/feishu_bot.py --push-alert INFO "告警标题" "告警详情"
```

### 3. 启动 Webhook 服务器

```bash
python scripts/feishu_bot.py --start-server --port 8080
```

### 4. 在代码中使用

```python
from scripts.feishu_bot import send_text, push_alert, push_stock_selection

# 发送文本消息
send_text("Hello from AlphaHelix!")

# 推送告警
push_alert("WARNING", "回撤警告", "当前回撤超过 20%")

# 推送选股结果
push_stock_selection(predictions, "2026-07-13")
```

## 飞书机器人配置步骤

1. 登录飞书开放平台：https://open.feishu.cn/
2. 创建应用 → 企业自建应用
3. 获取 App ID 和 App Secret
4. 添加机器人能力
5. 创建 Webhook，获取 Webhook URL 和 Secret
6. 配置环境变量

## 消息格式

### 文本消息
```
📊 AlphaHelix 每日选股报告 - 2026-07-13

选出股票:
1. 000001.SZ - 预测分数: 0.1234
2. 000002.SZ - 预测分数: 0.1122
...
```

### 告警消息
```
⚠️ 回撤警告
⏰ 时间: 2026-07-13 10:30:00
📍 级别: WARNING

📝 详情:
当前回撤超过 20%，请关注风险。
```

### 回测结果
```
📈 AlphaHelix 回测结果

回测配置:
- 模型: DoubleEnsemble
- 特征数: 30
- 测试期: 2024-01 ~ 2026-06

回测指标:
- IC: 0.0417
- ICIR: 0.80
- 服务胜率: 69.6%
- 总收益: 101.52%
- 最大回撤: -22.34%
- 夏普比率: 4.44
```
