"""
飞书机器人模块 - WebSocket 长连接方式

使用飞书官方 lark-oapi SDK 的 WSClient，通过 WebSocket 长连接接收消息，无需公网 IP。

依赖：pip install lark-oapi

用法：
    python feishu_bot.py --start-ws          # 启动 WebSocket 长连接
    python feishu_bot.py --push "测试消息"    # 推送消息（需要 Webhook）
"""
import sys
import os
import json
import logging
import argparse
import threading
from typing import Dict, Any, Optional
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 飞书配置
FEISHU_APP_ID = os.getenv("FEISHU_APP_ID", "")
FEISHU_APP_SECRET = os.getenv("FEISHU_APP_SECRET", "")
FEISHU_WEBHOOK_URL = os.getenv("FEISHU_WEBHOOK_URL", "")
FEISHU_WEBHOOK_SECRET = os.getenv("FEISHU_WEBHOOK_SECRET", "")


# ==================== WebSocket 长连接模式 ====================

class FeishuWSBot:
    """
    飞书 WebSocket 长连接机器人
    
    使用飞书官方 lark-oapi SDK 的 WSClient，通过 WebSocket 长连接接收消息，无需公网 IP。
    """
    
    def __init__(self, app_id: str, app_secret: str):
        self.app_id = app_id
        self.app_secret = app_secret
        self.ws_client = None
        self._handlers = {}
    
    def register_handler(self, event_type: str, handler):
        """注册事件处理器"""
        self._handlers[event_type] = handler
        logger.info(f"注册事件处理器: {event_type}")
    
    def _default_message_handler(self, data):
        """默认消息处理器"""
        try:
            event = data.event
            message = event.message
            sender = event.sender
            
            message_id = message.message_id
            chat_id = message.chat_id
            chat_type = message.chat_type
            msg_type = message.message_type
            content = json.loads(message.content)
            sender_id = sender.sender_id.open_id
            
            logger.info(f"收到消息: chat_id={chat_id}, type={msg_type}, sender={sender_id}")
            
            if msg_type == "text":
                text = content.get("text", "")
                logger.info(f"消息内容: {text[:100]}")
                
                # 处理命令
                reply = self._process_command(text, sender_id)
                
                # 回复消息
                self._reply_message(message_id, reply)
            else:
                logger.info(f"忽略非文本消息: {msg_type}")
                
        except Exception as e:
            logger.error(f"处理消息失败: {e}")
    
    def _process_command(self, text: str, sender_id: str) -> str:
        """处理命令"""
        # TODO: 实现命令处理逻辑
        return f"收到命令: {text}\n\n⏳ 功能开发中..."
    
    def _reply_message(self, message_id: str, text: str):
        """回复消息"""
        try:
            import lark_oapi as lark
            from lark_oapi.api.im.v1 import ReplyMessageRequest, ReplyMessageRequestBody
            
            client = lark.Client.builder() \
                .app_id(self.app_id) \
                .app_secret(self.app_secret) \
                .build()
            
            body = ReplyMessageRequestBody.builder() \
                .msg_type("text") \
                .content(json.dumps({"text": text})) \
                .build()
            
            request = ReplyMessageRequest.builder() \
                .message_id(message_id) \
                .request_body(body) \
                .build()
            
            response = client.im.v1.message.reply(request)
            
            if response.success():
                logger.info(f"回复成功: {message_id}")
            else:
                logger.error(f"回复失败: {response.code} {response.msg}")
                
        except Exception as e:
            logger.error(f"回复消息失败: {e}")
    
    def start(self):
        """启动 WebSocket 长连接"""
        try:
            import lark_oapi as lark
            from lark_oapi.api.im.v1 import P2ImMessageReceiveV1
            
            logger.info(f"启动飞书 WebSocket 长连接...")
            logger.info(f"App ID: {self.app_id[:8]}...")
            
            # 创建事件处理器
            event_handler = lark.EventDispatcherHandler.builder("", "") \
                .register_p2_im_message_receive_v1(self._default_message_handler) \
                .build()
            
            # 创建 WebSocket 客户端
            self.ws_client = lark.ws.Client(
                self.app_id,
                self.app_secret,
                event_handler=event_handler,
                log_level=lark.LogLevel.INFO
            )
            
            # 启动连接
            self.ws_client.start()
            
        except ImportError:
            logger.error("请安装 lark-oapi: pip install lark-oapi")
            sys.exit(1)
        except Exception as e:
            logger.error(f"启动失败: {e}")
            sys.exit(1)
    
    def stop(self):
        """停止 WebSocket 长连接"""
        if self.ws_client:
            self.ws_client.close()
            logger.info("WebSocket 已关闭")


# ==================== Webhook 推送模式 ====================

def _gen_sign(secret: str, timestamp: int) -> str:
    """生成飞书签名"""
    import hmac
    import hashlib
    import base64
    
    string_to_sign = f"{timestamp}\n{secret}"
    hmac_code = hmac.new(
        string_to_sign.encode('utf-8'),
        digestmod=hashlib.sha256
    ).digest()
    return base64.b64encode(hmac_code).decode('utf-8')


def send_text(text: str, webhook_url: str = None, secret: str = None) -> Dict[str, Any]:
    """
    发送文本消息到飞书（Webhook 方式）
    
    Args:
        text: 消息内容
        webhook_url: 飞书 Webhook URL
        secret: 签名密钥
    
    Returns:
        发送结果
    """
    import requests
    import time
    
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
    content = f"📊 AlphaHelix 每日选股报告 - {date}\n\n"
    content += f"选股模型: DoubleEnsemble\n"
    content += f"持仓数量: Top10\n\n"
    content += "选出股票:\n"
    
    for idx, (_, row) in enumerate(top10.iterrows(), 1):
        content += f"{idx}. {row['ts_code']} - 预测分数: {row['predicted']:.4f}\n"
    
    return send_text(content)


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
    
    content = f"{emoji} {title}\n"
    content += f"时间: {now}\n"
    content += f"级别: {level}\n\n"
    content += f"详情:\n{message}"
    
    if metadata:
        content += f"\n\n元数据:\n"
        for key, value in metadata.items():
            content += f"- {key}: {value}\n"
    
    return send_text(content)


def send_message(chat_id: str, text: str, msg_type: str = "text") -> Dict[str, Any]:
    """
    发送消息到指定聊天（使用飞书 API）
    
    Args:
        chat_id: 聊天 ID
        text: 消息内容
        msg_type: 消息类型 (text/post)
    
    Returns:
        发送结果
    """
    try:
        import lark_oapi as lark
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody
        
        client = lark.Client.builder() \
            .app_id(FEISHU_APP_ID) \
            .app_secret(FEISHU_APP_SECRET) \
            .build()
        
        if msg_type == "post":
            content = json.dumps({
                'zh_cn': {
                    'title': '📊 AlphaHelix 通知',
                    'content': [[{'tag': 'text', 'text': text}]]
                }
            })
        else:
            content = json.dumps({'text': text})
        
        request = CreateMessageRequest.builder() \
            .receive_id_type('chat_id') \
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(chat_id)
                .msg_type(msg_type)
                .content(content)
                .build()
            ) \
            .build()
        
        response = client.im.v1.message.create(request)
        
        if response.success():
            return {'status': 'ok', 'message_id': response.data.message_id}
        else:
            return {'status': 'error', 'message': f'{response.code} {response.msg}'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def get_chat_list() -> list:
    """
    获取机器人参与的聊天列表
    
    Returns:
        聊天列表
    """
    try:
        import lark_oapi as lark
        from lark_oapi.api.im.v1 import ListChatRequest
        
        client = lark.Client.builder() \
            .app_id(FEISHU_APP_ID) \
            .app_secret(FEISHU_APP_SECRET) \
            .build()
        
        request = ListChatRequest.builder() \
            .page_size(10) \
            .build()
        
        response = client.im.v1.chat.list(request)
        
        if response.success():
            return [{'chat_id': item.chat_id, 'name': item.name} for item in response.data.items]
        else:
            return []
    except Exception as e:
        logger.error(f"获取聊天列表失败: {e}")
        return []


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="飞书机器人")
    parser.add_argument("--start-ws", action="store_true", help="启动 WebSocket 长连接")
    parser.add_argument("--push", type=str, help="推送文本消息（Webhook 方式）")
    parser.add_argument("--send", nargs=2, metavar=("CHAT_ID", "MESSAGE"), help="发送消息到指定聊天")
    parser.add_argument("--list-chats", action="store_true", help="获取聊天列表")
    parser.add_argument("--push-alert", nargs=3, metavar=("LEVEL", "TITLE", "MESSAGE"), help="推送告警")
    args = parser.parse_args()
    
    if args.start_ws:
        if not FEISHU_APP_ID or not FEISHU_APP_SECRET:
            print("❌ 请配置 FEISHU_APP_ID 和 FEISHU_APP_SECRET")
            sys.exit(1)
        
        bot = FeishuWSBot(FEISHU_APP_ID, FEISHU_APP_SECRET)
        
        # 注册自定义处理器（可选）
        # bot.register_handler("im.message.receive_v1", custom_handler)
        
        # 启动 WebSocket 长连接
        bot.start()
    elif args.push:
        result = send_text(args.push)
        print(f"推送结果: {result}")
    elif args.send:
        chat_id, message = args.send
        result = send_message(chat_id, message)
        print(f"发送结果: {result}")
    elif args.list_chats:
        chats = get_chat_list()
        print(f"找到 {len(chats)} 个聊天:")
        for chat in chats:
            print(f"  - {chat['name']}: {chat['chat_id']}")
    elif args.push_alert:
        level, title, message = args.push_alert
        result = push_alert(level, title, message)
        print(f"推送结果: {result}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
