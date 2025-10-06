import yfinance as yf
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy.optimize import brentq
import yaml
from sqlalchemy import create_engine
import os
import sys
print("sys.path:", sys.path)
print("cwd:", os.getcwd())

from utils.trading_calendar import (
    is_trading_day,
    next_trading_day,
    last_trading_day_of_week,
    last_trading_day_of_month,
    find_expiry_within_trading_days
)

# Setup project paths and config
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))
print(project_root)
config_path = os.path.join(project_root, 'config.yml')
data_folder = os.path.join(project_root, 'data', 'sf_data')

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

params = config['database']
api_key = config['simfin']['api_key']

def create_engine_db():
    connection_string = (
        f"postgresql+psycopg2://{params['user']}:{params['password']}"
        f"@{params['host']}:{params['port']}/{params['dbname']}"
    )
    engine = create_engine(connection_string)
    return engine

def connect_db():
    print("Connecting to PostgreSQL database...")
    conn = psycopg2.connect(
        dbname=params['dbname'],
        user=params['user'],
        password=params['password'],
        host=params['host'],
        port=params['port']
    )
    conn.set_client_encoding('UTF8')
    print("Database connection established.")
    return conn

def insert_option_metrics(conn, result_dict):
    with conn.cursor() as cur:
        insert_query = """
            INSERT INTO option_metrics (
                symbol, date, gamma_flip_line, underlying_price,
                day_ahead_high, day_ahead_low,
                wk_ahead_high, wk_ahead_low,
                mth_ahead_high, mth_ahead_low
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cur.execute(insert_query, (
            result_dict['symbol'],
            result_dict['date'],
            result_dict['gamma_flip_line'],
            result_dict['underlying_price'],
            result_dict.get('1_day_ahead_high'),
            result_dict.get('1_day_ahead_low'),
            result_dict.get('1_wk_ahead_high'),
            result_dict.get('1_wk_ahead_low'),
            result_dict.get('1_mth_ahead_high'),
            result_dict.get('1_mth_ahead_low'),
        ))
    conn.commit()

def black_scholes_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

def fetch_option_chain(ticker, expiration):
    try:
        tk = yf.Ticker(ticker)
        opt = tk.option_chain(expiration)
        calls = opt.calls.copy()
        puts = opt.puts.copy()
        calls['optionType'] = 'call'
        puts['optionType'] = 'put'
        df = pd.concat([calls, puts], ignore_index=True)
        df['expiration'] = expiration
        return df
    except Exception as e:
        print(f"Failed to fetch option chain for {ticker} expiration {expiration}: {e}")
        return pd.DataFrame()

def filter_strikes_by_oi_and_atm(df_opt, underlying_price, top_n_strikes=20):
    if df_opt.empty:
        return df_opt
    df_opt = df_opt.dropna(subset=['openInterest', 'impliedVolatility', 'strike'])
    df_opt = df_opt[df_opt['openInterest'] > 0]
    if df_opt.empty:
        return df_opt
    df_opt = df_opt.sort_values(by='openInterest', ascending=False)
    top_strikes_oi = df_opt['strike'].unique()[:top_n_strikes]
    df_filtered = df_opt[df_opt['strike'].isin(top_strikes_oi)].copy()
    df_filtered['atm_distance'] = abs(df_filtered['strike'] - underlying_price)
    df_filtered = df_filtered.sort_values(by='atm_distance')
    return df_filtered.drop(columns=['atm_distance'])

def get_atm_iv(df_opt, underlying_price):
    # Find the ATM strike closest to underlying price
    strikes = df_opt['strike'].unique()
    atm_strike = min(strikes, key=lambda x: abs(x - underlying_price))
    atm_options = df_opt[df_opt['strike'] == atm_strike].copy()
    iv_call = atm_options[atm_options['optionType'] == 'call']['impliedVolatility']
    oi_call = atm_options[atm_options['optionType'] == 'call']['openInterest']
    iv_put = atm_options[atm_options['optionType'] == 'put']['impliedVolatility']
    oi_put = atm_options[atm_options['optionType'] == 'put']['openInterest']
    iv_calls = np.average(iv_call, weights=oi_call) if len(iv_call) > 0 and oi_call.sum() > 0 else np.nan
    iv_puts = np.average(iv_put, weights=oi_put) if len(iv_put) > 0 and oi_put.sum() > 0 else np.nan
    if not np.isnan(iv_calls) and not np.isnan(iv_puts):
        atm_iv = (iv_calls + iv_puts) / 2
    elif not np.isnan(iv_calls):
        atm_iv = iv_calls
    elif not np.isnan(iv_puts):
        atm_iv = iv_puts
    else:
        atm_iv = np.nan
    return atm_iv

def compute_expected_move_asymmetric(underlying_price, atm_iv, days_to_expiry):
    T = days_to_expiry / 365
    if np.isnan(atm_iv) or atm_iv == 0 or T <= 0:
        return (0, 0)
    up_move = underlying_price * (np.exp(atm_iv * np.sqrt(T)) - 1)
    down_move = underlying_price * (1 - np.exp(-atm_iv * np.sqrt(T)))
    return (up_move, down_move)

def net_gamma_exposure(S, df_opt, r):
    total_gamma_exp = 0.0
    for idx, row in df_opt.iterrows():
        K = row['strike']
        T = row['daysToExpiry'] / 365
        sigma = row['impliedVolatility']
        oi = row['openInterest']
        option_type = row['optionType']
        gamma = black_scholes_gamma(S, K, T, r, sigma)
        gamma_exp = gamma * oi * 100
        if option_type == 'put':
            gamma_exp = -gamma_exp
        total_gamma_exp += gamma_exp
    return total_gamma_exp

def find_gamma_flip_level(df_opt, underlying_price, r=0.01, price_range_pct=0.2):
    low = underlying_price * (1 - price_range_pct)
    high = underlying_price * (1 + price_range_pct)
    try:
        gamma_flip_price = brentq(lambda S: net_gamma_exposure(S, df_opt, r), low, high)
        return gamma_flip_price
    except ValueError:
        return float('nan')

def process_ticker_wrapper(ticker, as_of_date):
    as_of_date = as_of_date.replace(hour=0, minute=0, second=0, microsecond=0)
    tk = yf.Ticker(ticker)
    expirations = tk.options
    if not expirations:
        print(f"No expirations found for {ticker}")
        return None
    expirations_dt = [datetime.strptime(d, "%Y-%m-%d").date() for d in expirations]
    dte_dict = {d: (d - as_of_date.date()).days for d in expirations_dt}
    history = tk.history(period='1d')
    if history.empty:
        print(f"No price history for {ticker}")
        return None
    underlying_price = history['Close'].iloc[0]

    as_of_date_only = as_of_date.date()

    # 1-day ahead expiry: find next trading day then expiry within next 5 trading days
    start_1d = next_trading_day(as_of_date_only)
    selected_1d = find_expiry_within_trading_days(expirations_dt, start_1d, 5)

    # 1-week ahead expiry: check calendar days +7 to +10
    selected_1w = None
    for offset in range(7, 11):
        candidate = as_of_date_only + timedelta(days=offset)
        if candidate in expirations_dt:
            selected_1w = candidate
            break

    # 1-month ahead expiry: check calendar days +30 to +35
    selected_1m = None
    for offset in range(30, 36):
        candidate = as_of_date_only + timedelta(days=offset)
        if candidate in expirations_dt:
            selected_1m = candidate
            break

    # Only update week ahead if last trading day of week
    if not last_trading_day_of_week(as_of_date_only):
        selected_1w = None

    # Only update month ahead if last trading day of month
    if not last_trading_day_of_month(as_of_date_only):
        selected_1m = None

    expected_ranges = {
        '1_day_ahead_high': np.nan,
        '1_day_ahead_low': np.nan,
        '1_wk_ahead_high': np.nan,
        '1_wk_ahead_low': np.nan,
        '1_mth_ahead_high': np.nan,
        '1_mth_ahead_low': np.nan,
    }

    r = 0.01  # risk free rate

    def process_expected_move(expiry, label_prefix):
        if expiry is None:
            print(f"No expiry selected for {label_prefix}")
            return
        dte = dte_dict[expiry]
        df_opt = fetch_option_chain(ticker, expiry.strftime("%Y-%m-%d"))
        df_filtered = filter_strikes_by_oi_and_atm(df_opt, underlying_price, top_n_strikes=40)
        if df_filtered.empty:
            print(f"No valid options data for expiry {expiry.strftime('%Y-%m-%d')}")
            return
        atm_iv = get_atm_iv(df_filtered, underlying_price)
        up_move, down_move = compute_expected_move_asymmetric(underlying_price, atm_iv, dte)
        expected_ranges[f'{label_prefix}_high'] = underlying_price + up_move
        expected_ranges[f'{label_prefix}_low'] = underlying_price - down_move
        print(f"{label_prefix}: expiry={expiry.strftime('%Y-%m-%d')}, DTE={dte}, ATM_IV={atm_iv:.4f}, Up Move={up_move:.4f}, Down Move={down_move:.4f}")

    process_expected_move(selected_1d, '1_day_ahead')
    process_expected_move(selected_1w, '1_wk_ahead')
    process_expected_move(selected_1m, '1_mth_ahead')

    all_expiries_for_gamma = sorted(set(
        [d for d in expirations_dt if 1 <= dte_dict[d] <= 3] +
        [d for d in expirations_dt if 5 <= dte_dict[d] <= 7] +
        [d for d in expirations_dt if 20 <= dte_dict[d] <= 24]
    ))

    df_gamma = pd.DataFrame()
    for exp in all_expiries_for_gamma:
        df_opt = fetch_option_chain(ticker, exp.strftime("%Y-%m-%d"))
        dte = dte_dict[exp]
        df_opt['daysToExpiry'] = dte
        df_filtered = filter_strikes_by_oi_and_atm(df_opt, underlying_price, top_n_strikes=40)
        df_gamma = pd.concat([df_gamma, df_filtered])

    gamma_flip = find_gamma_flip_level(df_gamma, underlying_price, r)
    print(f"Gamma flip line level: {gamma_flip}")

    result = {
        'symbol': ticker,
        'date': as_of_date.strftime("%Y-%m-%d"),
        'gamma_flip_line': gamma_flip,
        'underlying_price': underlying_price,
        **expected_ranges
    }
    return result

if __name__ == "__main__":
    today = datetime.now()
    conn = None
    try:
        result = process_ticker_wrapper("SPY", today)
        print(result)
        if result:
            conn = connect_db()
            insert_option_metrics(conn, result)
            print("Result saved to PostgreSQL database.")
    except Exception as e:
        print("Error:", e)
    finally:
        if conn:
            conn.close()
