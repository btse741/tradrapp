import requests
import os
import yaml
import pandas as pd
from telegram import Bot

def send_telegram_message(bot_token, chat_id, message):
    """
    Sends a Telegram message via Bot API.
    Parameters:
        bot_token (str): Telegram bot token from BotFather
        chat_id (str or int): Chat ID for recipient
        message (str): Message text to send
    """
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'Markdown'  # Optional: you can remove or use HTML
        }
        response = requests.post(url, data=payload, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to send telegram message: {e}")
