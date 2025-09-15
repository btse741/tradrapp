import os
import yaml
import pandas as pd
from telegram import Bot
import asyncio

def load_config():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))
    config_path = os.path.join(project_root, 'config.yml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

TOKEN = config['telegram']['bot_token']
CHAT_ID = config['telegram']['chat_id']

bot = Bot(token=TOKEN)

async def send_strategy_update(text=None, perf_img=None, chart_img=None, trades_img=None):
    tasks = []
    if text:
        tasks.append(bot.send_message(chat_id=CHAT_ID, text=text, parse_mode='Markdown'))
    if perf_img:
        with open(perf_img, 'rb') as f:
            tasks.append(bot.send_photo(chat_id=CHAT_ID, photo=f.read()))
    if chart_img:
        with open(chart_img, 'rb') as f:
            tasks.append(bot.send_photo(chat_id=CHAT_ID, photo=f.read()))
    if trades_img:
        with open(trades_img, 'rb') as f:
            tasks.append(bot.send_photo(chat_id=CHAT_ID, photo=f.read()))
    await asyncio.gather(*tasks)

def send_strategy_update_sync(text=None, perf_img=None, chart_img=None, trades_img=None):
    asyncio.run(send_strategy_update(text, perf_img, chart_img, trades_img))

async def send_test_message():
    await bot.send_message(chat_id=CHAT_ID, text="Test message from Telegram bot.")

def test_bot_connection():
    asyncio.run(send_test_message())

def format_table_md(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    try:
        table_str = df.to_markdown(index=False)
    except Exception:
        table_str = df.to_string(index=False)
    table_str = table_str.replace('`', '\'')
    return f"``````"

def prepare_trades_text(trades_df: pd.DataFrame) -> str:
    """Prepare portfolio holdings and trades actions text for Telegram message."""
    if trades_df.empty:
        return "No trade data available."

    # Use 'Current Position' column instead of 'Weight' for holdings
    holdings_df = trades_df.groupby(['Strategy', 'ETF']).agg({'Current Position': 'last'}).reset_index()
    holdings_df.rename(columns={'Current Position': 'Current Weight'}, inplace=True)

    # Filter trades actions with non-zero weight change and not 'Hold/Adjust'
    actions_df = trades_df[(trades_df['Weight Change'] != 0) & (trades_df['Action'] != 'Hold/Adjust')]

    message_parts = []

    holdings_text = format_table_md(holdings_df)
    if holdings_text:
        message_parts.append(f"*Current Portfolio Holdings:*\n{holdings_text}")
    else:
        message_parts.append("_No current holdings data available._")

    if actions_df.empty:
        message_parts.append("_No trade actions currently._")
    else:
        trades_table = actions_df[['Strategy', 'ETF', 'Action', 'Weight Change']]
        trades_text = format_table_md(trades_table)
        message_parts.append(f"*Trade Actions:*\n{trades_text}")

    # Join parts separated by two newlines
    return "\n\n".join(message_parts)


async def send_trades_text(trades_df: pd.DataFrame):
    message = prepare_trades_text(trades_df)
    if not message.strip():
        message = "No trade data available."
    await bot.send_message(chat_id=CHAT_ID, text=message, parse_mode='Markdown')

def send_trades_text_sync(trades_df: pd.DataFrame):
    asyncio.run(send_trades_text(trades_df))

async def send_image_async(image_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with open(image_path, 'rb') as img_file:
        await bot.send_photo(chat_id=CHAT_ID, photo=img_file)

def send_image_sync(image_path: str):
    asyncio.run(send_image_async(image_path))
