"""
飞书机器人模块

支持：
1. 推送选股结果到飞书
2. 推送告警通知
3. 接收飞书命令（Webhook 方式）

用法：
    python feishu_bot.py --push "测试消息"
    python feishu_bot.py --start-server
"""
import sys
import os
import json
import hashlib
import hmac
import base64
import time
import logging
import argparse
from typing import Dict, Any, Optional
from datetime import datetime

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 飞书配置
FEISHU_APP_ID = os.getenv("FEISHU_APP_ID", "")
FEISHU_APP_SECRET = os.getenv("FEISHU_APP_SECRET", "")
FEISHU_WEBHOOK_URL = os.getenv("FEISHU_WEBHOOK_URL", "")
FEISHU_WEBHOOK_SECRET = os.getenv("FEISHU_WEBHOOK_SECRET", "")


def _gen_sign(secret: str, timestamp: int) -> str:
    """生成飞书签名"""
    string_to_sign = f"{timestamp}\n{secret}"
    hmac_code = hmac.new(
        string_to_sign.encode('utf-8'),
        digestmod=hashlib.sha256
    ).digest()
    return base64.b64encode(hmac_code).decode('utf-8')


def send_text(text: str, webhook_url: str = None, secret: str = None) -> Dict[str, Any]:
    """
    发送文本消息到飞书
    
    Args:
        text: 消息内容
        webhook_url: 飞书 Webhook URL
        secret: 签名密钥
    
    Returns:
        发送结果
    """
    webhook_url = webhook_url or FEISHU_WEBHOOK_URL
    secret = secret or FEISHU_WEBHOOK_SECRET
    
    if not webhook_url:
        return {'status': 'error', 'message': '未配置飞书 webhook_url'}
    
    try:
        timestamp = int(time.time())
        
        msg = {
            "timestamp": timestamp,
            "msg_type": "text",
            "content": {
                "text": text
            }
        }
        
        if secret:
            msg["sign"] = _gen_sign(secret, timestamp)
        
        response = requests.post(
            webhook_url,
            json=msg,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        result = response.json()
        
        if result.get('code') == 0:
            return {'status': 'ok', 'message': '推送成功'}
        else:
            return {'status': 'error', 'message': f'推送失败: {result}'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def send_rich_text(title: str, content: str, webhook_url: str = None, secret: str = None) -> Dict[str, Any]:
    """
    发送富文本消息到飞书
    
    Args:
        title: 消息标题
        content: 消息内容（Markdown 格式）
        webhook_url: 飞书 Webhook URL
        secret: 签名密钥
    
    Returns:
        发送结果
    """
    webhook_url = webhook_url or FEISHU_WEBHOOK_URL
    secret = secret or FEISHU_WEBHOOK_SECRET
    
    if not webhook_url:
        return {'status': 'error', 'message': '未配置飞书 webhook_url'}
    
    try:
        timestamp = int(time.time())
        
        msg = {
            "timestamp": timestamp,
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": title
                    },
                    "template": "blue"
                },
                "elements": [
                    {
                        "tag": "markdown",
                        "content": content
                    }
                ]
            }
        }
        
        if secret:
            msg["sign"] = _gen_sign(secret, timestamp)
        
        response = requests.post(
            webhook_url,
            json=msg,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        result = response.json()
        
        if result.get('code') == 0:
            return {'status': 'ok', 'message': '推送成功'}
        else:
            return {'status': 'error', 'message': f'推送失败: {result}'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def push_stock_selection(predictions: pd.DataFrame, date: str = None) -> Dict[str, Any]:
    """
    推送选股结果到飞书
    
    Args:
        predictions: 预测数据
        date: 日期
    
    Returns:
        推送结果
    """
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    # 选择 Top10
    top10 = predictions.nlargest(10, 'predicted')
    
    # 构建消息
    title = f"📊 AlphaHelix 每日选股报告 - {date}"
    
    content = f"**选股日期**: {date}\n"
    content += f"**选股模型**: DoubleEnsemble\n"
    content += f"**持仓数量**: Top10\n\n"
    content += "**选出股票**:\n"
    content += "| 排名 | 股票代码 | 预测分数 | 行业 |\n"
    content += "| --- | --- | --- | --- |\n"
    
    for idx, (_, row) in enumerate(top10.iterrows(), 1):
        content += f"| {idx} | {row['ts_code']} | {row['predicted']:.4f} | {row.get('industry', '-')} |\n"
    
    return send_rich_text(title, content)


def push_alert(level: str, title: str, message: str, metadata: Dict = None) -> Dict[str, Any]:
    """
    推送告警到飞书
    
    Args:
        level: 告警级别 (INFO/WARNING/ERROR)
        title: 告警标题
        message: 告警详情
        metadata: 附加元数据
    
    Returns:
        推送结果
    """
    emoji_map = {
        "INFO": "ℹ️",
        "WARNING": "⚠️",
        "ERROR": "🚨"
    }
    emoji = emoji_map.get(level, "")
    
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    content = f"{emoji} **{title}**\n"
    content += f"⏰ 时间: {now}\n"
    content += f"📍 级别: {level}\n"
    content += f"\n📝 详情:\n{message}"
    
    if metadata:
        content += f"\n\n🔧 元数据:\n"
        for key, value in metadata.items():
            content += f"- {key}: {value}\n"
    
    return send_rich_text(f"{emoji} {title}", content)


def push_backtest_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    推送回测结果到飞书
    
    Args:
        result: 回测结果
    
    Returns:
        推送结果
    """
    title = "📈 AlphaHelix 回测结果"
    
    content = "**回测配置**:\n"
    content += f"- 模型: {result.get('model', 'Unknown')}\n"
    content += f"- 特征数: {result.get('features', 'Unknown')}\n"
    content += f"- 测试期: {result.get('test_period', 'Unknown')}\n\n"
    content += "**回测指标**:\n"
    content += f"- IC: {result.get('ic', 0):.4f}\n"
    content += f"- ICIR: {result.get('icir', 0):.2f}\n"
    content += f"- 服务胜率: {result.get('win_rate', 0):.1%}\n"
    content += f"- 总收益: {result.get('total_return', 0):.2%}\n"
    content += f"- 最大回撤: {result.get('max_drawdown', 0):.2%}\n"
    content += f"- 夏普比率: {result.get('sharpe', 0):.2f}\n"
    
    return send_rich_text(title, content)


def start_webhook_server(port: int = 8080):
    """
    启动飞书 Webhook 服务器
    
    Args:
        port: 端口号
    """
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    @app.route("/feishu/webhook", methods=["GET", "POST"])
    def webhook():
        """飞书 Webhook 入口"""
        # GET: URL 验证
        if request.method == "GET":
            challenge = request.args.get("challenge", "")
            if challenge:
                return jsonify({"challenge": challenge})
            return jsonify({"error": "missing challenge"}), 400
        
        # POST: 接收事件
        body = request.get_json()
        if not body:
            return jsonify({"error": "empty body"}), 400
        
        # URL 验证事件
        if body.get("type") == "url_verification":
            return jsonify({"challenge": body.get("challenge", "")})
        
        # 处理消息事件
        event = body.get("event", {})
        if event.get("msg_type") == "text":
            message_id = event.get("message_id", "")
            chat_id = event.get("chat_id", "")
            text = event.get("text", "")
            
            logger.info(f"收到消息: {text}")
            
            # TODO: 处理命令
            reply = f"收到命令: {text}"
            
            # 回复消息（需要使用飞书 API）
            # 这里简化处理，直接返回成功
            return jsonify({"status": "ok"}), 200
        
        return jsonify({"error": "unsupported event"}), 400
    
    @app.route("/feishu/health", methods=["GET"])
    def health():
        """健康检查"""
        return jsonify({
            "status": "ok",
            "adapter": "feishu",
            "timestamp": datetime.now().isoformat()
        })
    
    logger.info(f"启动飞书 Webhook 服务器: http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)


def main():
    parser = argparse.ArgumentParser(description="飞书机器人")
    parser.add_argument("--push", type=str, help="推送文本消息")
    parser.add_argument("--push-alert", nargs=3, metavar=("LEVEL", "TITLE", "MESSAGE"), help="推送告警")
    parser.add_argument("--start-server", action="store_true", help="启动 Webhook 服务器")
    parser.add_argument("--port", type=int, default=8080, help="服务器端口")
    args = parser.parse_args()
    
    if args.push:
        result = send_text(args.push)
        print(f"推送结果: {result}")
    elif args.push_alert:
        level, title, message = args.push_alert
        result = push_alert(level, title, message)
        print(f"推送结果: {result}")
    elif args.start_server:
        start_webhook_server(args.port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
