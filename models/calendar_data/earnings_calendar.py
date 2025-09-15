import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import yaml

def load_config():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    config_path = os.path.join(project_root, 'config.yml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()
FINNHUB_API_KEY = config.get('finn', {}).get('api_key')

def get_upcoming_earnings_finnhub(days_ahead=7):
    today = datetime.today()
    to_date = today + timedelta(days=days_ahead)
    url = f"https://finnhub.io/api/v1/calendar/earnings?from={today.strftime('%Y-%m-%d')}&to={to_date.strftime('%Y-%m-%d')}&token={FINNHUB_API_KEY}"
    response = requests.get(url)
    data = response.json()
    # The data is under 'earningsCalendar' key
    earnings_data = data.get('earningsCalendar', [])
    if not earnings_data:
        return pd.DataFrame()
    return pd.DataFrame(earnings_data)
