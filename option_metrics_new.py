import os
import yaml
import yfinance as yf
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy.optimize import brentq
from fredapi import Fred

# Trading calendar utilities
import datetime as dt
import holidays

def get_us_market_holidays(years=None):
    if years is None:
        years = [dt.datetime.now().year]
    us_holidays = holidays.US(years=years)
    return set(us_holidays.keys())

def is_trading_day(date):
    if date.weekday() >= 5:
        return False
    holidays_set = get_us_market_holidays([date.year])
    return date not in holidays_set

def next_trading_day(date):
    next_day = date + timedelta(days=1)
    while not is_trading_day(next_day):
        next_day += timedelta(days=1)
    return next_day

def last_trading_day_of_week(date):
    if not is_trading_day(date):
        return False
    next_day = date + timedelta(days=1)
    while next_day.weekday() < 5:
        if is_trading_day(next_day):
            return False
        next_day += timedelta(days=1)
    return True

def last_trading_day_of_month(date):
    if not is_trading_day(date):
        return False
    next_day = date + timedelta(days=1)
    while next_day.month == date.month:
        next_day += timedelta(days=1)
    last_day = next_day - timedelta(days=1)
    while not is_trading_day(last_day):
        last_day -= timedelta(days=1)
    return date == last_day

def last_trading_day_of_quarter(date):
    q_month_ends = [3, 6, 9, 12]
    month = date.month
    quarter_end_month = next((m for m in q_month_ends if m >= month), 12)
    if quarter_end_month == 12:
        next_month = datetime(date.year + 1, 1, 1).date()
    else:
        next_month = datetime(date.year, quarter_end_month + 1, 1).date()
    last_day = next_month - timedelta(days=1)
    while not is_trading_day(last_day):
        last_day -= timedelta(days=1)
    return date == last_day

def find_expiry_within_trading_days(expirations, start_date, max_trading_days):
    candidate_dates = []
    current = start_date
    count = 0
    while count < max_trading_days:
        if is_trading_day(current):
            candidate_dates.append(current)
            count += 1
        current += timedelta(days=1)
    for d in candidate_dates:
        if d in expirations:
            return d
    return None

def find_closest_expiry_in_range(expirations, target_date, start_days_offset, end_days_offset):
    expirations = sorted(expirations)
    candidate_expiries = [
        d for d in expirations
        if start_days_offset <= (d - target_date).days <= end_days_offset
    ]
    if candidate_expiries:
        return min(candidate_expiries, key=lambda x: abs((x - target_date).days))
    future_expiries = [d for d in expirations if d > target_date]
    if future_expiries:
        return min(future_expiries)
    return None

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
        if df.empty:
            print(f"Warning: Empty option chain dataframe for {ticker} expiry {expiration}")
        return df
    except Exception as e:
        print(f"Failed to fetch option chain for {ticker} expiry {expiration}: {e}")
        return pd.DataFrame()

def filter_strikes_by_oi_and_atm(df_opt, underlying_price, top_n_strikes=40):
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

def compute_expected_move_asymmetric(underlying_price, atm_iv, days_to_expiry, r):
    T = days_to_expiry / 365
    if np.isnan(atm_iv) or atm_iv <= 0 or T <= 0:
        return (0, 0)
    up_move = underlying_price * (np.exp(atm_iv * np.sqrt(T)) - 1)
    down_move = underlying_price * (1 - np.exp(-atm_iv * np.sqrt(T)))
    return (up_move, down_move)

def black_scholes_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

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

# Config and FRED client setup  
project_root = os.path.abspath(os.path.dirname(__file__))
config_path = os.path.join(project_root, 'config.yml')

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

params = config['database']
fred_api_key = config['fred']['api_key']
fred = Fred(api_key=fred_api_key)

def get_risk_free_rate():
    try:
        treasury_yield = fred.get_series('DGS10')
        latest_yield = treasury_yield[-1] / 100
        return latest_yield
    except Exception as e:
        print(f"Failed to fetch risk free rate from FRED: {e}")
        return 0.01

def connect_db():
    return psycopg2.connect(
        dbname=params['dbname'],
        user=params['user'],
        password=params['password'],
        host=params['host'],
        port=params['port']
    )

def insert_option_metrics(conn, result_dict):
    with conn.cursor() as cur:
        insert_query = """
            INSERT INTO option_metrics (
                symbol, date, gamma_flip_line, underlying_price,
                day_implied_vol, week_implied_vol, month_implied_vol, quarter_implied_vol,
                day_ahead_high, day_ahead_low,
                wk_ahead_high, wk_ahead_low,
                mth_ahead_high, mth_ahead_low,
                qtr_ahead_high, qtr_ahead_low
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, date) DO UPDATE SET
                gamma_flip_line = EXCLUDED.gamma_flip_line,
                underlying_price = EXCLUDED.underlying_price,
                day_implied_vol = EXCLUDED.day_implied_vol,
                week_implied_vol = EXCLUDED.week_implied_vol,
                month_implied_vol = EXCLUDED.month_implied_vol,
                quarter_implied_vol = EXCLUDED.quarter_implied_vol,
                day_ahead_high = EXCLUDED.day_ahead_high,
                day_ahead_low = EXCLUDED.day_ahead_low,
                wk_ahead_high = EXCLUDED.wk_ahead_high,
                wk_ahead_low = EXCLUDED.wk_ahead_low,
                mth_ahead_high = EXCLUDED.mth_ahead_high,
                mth_ahead_low = EXCLUDED.mth_ahead_low,
                qtr_ahead_high = EXCLUDED.qtr_ahead_high,
                qtr_ahead_low = EXCLUDED.qtr_ahead_low
        """
        cur.execute(insert_query, (
            result_dict['symbol'],
            result_dict['date'],
            result_dict['gamma_flip_line'],
            result_dict['underlying_price'],
            result_dict.get('1_day_implied_vol'),
            result_dict.get('1_wk_implied_vol'),
            result_dict.get('1_mth_implied_vol'),
            result_dict.get('1_qtr_implied_vol'),
            result_dict.get('1_day_ahead_high'),
            result_dict.get('1_day_ahead_low'),
            result_dict.get('1_wk_ahead_high'),
            result_dict.get('1_wk_ahead_low'),
            result_dict.get('1_mth_ahead_high'),
            result_dict.get('1_mth_ahead_low'),
            result_dict.get('1_qtr_ahead_high'),
            result_dict.get('1_qtr_ahead_low'),
        ))
    conn.commit()

def process_ticker_wrapper(ticker, as_of_date):
    as_of_date = as_of_date.replace(hour=0, minute=0, second=0, microsecond=0)
    tk = yf.Ticker(ticker)
    expirations = tk.options
    if not expirations:
        print(f"No expirations found for {ticker}")
        return None

    expirations_dt = [datetime.strptime(d, "%Y-%m-%d").date() for d in expirations]
    print(f"Available expirations for {ticker} on {as_of_date.strftime('%Y-%m-%d')}: {expirations_dt}")

    dte_dict = {d: (d - as_of_date.date()).days for d in expirations_dt}
    history = tk.history(period='1d')
    if history.empty:
        print(f"No price history for {ticker}")
        return None
    underlying_price = history['Close'].iloc[0]

    as_of_date_only = as_of_date.date()
    r = get_risk_free_rate()

    start_1d = next_trading_day(as_of_date_only)
    selected_1d = find_closest_expiry_in_range(expirations_dt, start_1d, 0, 5)

    selected_1w = None
    if last_trading_day_of_week(as_of_date_only):
        selected_1w = find_closest_expiry_in_range(expirations_dt, as_of_date_only + timedelta(days=7), 0, 3)

    selected_1m = None
    if last_trading_day_of_month(as_of_date_only):
        selected_1m = find_closest_expiry_in_range(expirations_dt, as_of_date_only + timedelta(days=30), 20, 60)

    selected_1q = None
    if last_trading_day_of_quarter(as_of_date_only):
        selected_1q = find_closest_expiry_in_range(expirations_dt, as_of_date_only + timedelta(days=90), 70, 130)

    expected_ranges = {
        '1_day_ahead_high': np.nan,
        '1_day_ahead_low': np.nan,
        '1_wk_ahead_high': np.nan,
        '1_wk_ahead_low': np.nan,
        '1_mth_ahead_high': np.nan,
        '1_mth_ahead_low': np.nan,
        '1_qtr_ahead_high': np.nan,
        '1_qtr_ahead_low': np.nan,
    }

    implied_vols = {
        '1_day_implied_vol': np.nan,
        '1_wk_implied_vol': np.nan,
        '1_mth_implied_vol': np.nan,
        '1_qtr_implied_vol': np.nan,
    }

    def process_expected_move(expiry, label_prefix):
        if expiry is None:
            print(f"No expiry selected for {label_prefix}")
            return
        dte = dte_dict.get(expiry, (expiry - as_of_date_only).days)
        df_opt = fetch_option_chain(ticker, expiry.strftime("%Y-%m-%d"))
        if df_opt.empty:
            print(f"Skipping empty option chain data for expiry {expiry.strftime('%Y-%m-%d')}")
            return
        df_filtered = filter_strikes_by_oi_and_atm(df_opt, underlying_price, top_n_strikes=40)
        if df_filtered.empty:
            print(f"No valid strikes for expiry {expiry.strftime('%Y-%m-%d')}")
            return
        atm_iv = get_atm_iv(df_filtered, underlying_price)
        implied_vols[f'{label_prefix}_implied_vol'] = atm_iv
        up_move, down_move = compute_expected_move_asymmetric(underlying_price, atm_iv, dte, r)
        expected_ranges[f'{label_prefix}_ahead_high'] = underlying_price + up_move
        expected_ranges[f'{label_prefix}_ahead_low'] = underlying_price - down_move
        print(f"{label_prefix}: expiry={expiry.strftime('%Y-%m-%d')}, DTE={dte}, ATM_IV={atm_iv:.4f}, Up Move={up_move:.4f}, Down Move={down_move:.4f}")

    process_expected_move(selected_1d, '1_day')
    process_expected_move(selected_1w, '1_wk')
    process_expected_move(selected_1m, '1_mth')
    process_expected_move(selected_1q, '1_qtr')

    all_expiries_for_gamma = sorted(set(
        [d for d in expirations_dt if 1 <= dte_dict[d] <= 3] +
        [d for d in expirations_dt if 5 <= dte_dict[d] <= 7] +
        [d for d in expirations_dt if 20 <= dte_dict[d] <= 24]
    ))

    df_gamma = pd.DataFrame()
    for exp in all_expiries_for_gamma:
        df_opt = fetch_option_chain(ticker, exp.strftime("%Y-%m-%d"))
        if df_opt.empty:
            print(f"Skipping empty gamma options data for expiry {exp.strftime('%Y-%m-%d')}")
            continue
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
        **implied_vols,
        **expected_ranges,
    }
    return result

if __name__ == "__main__":
    today = datetime.now()
    conn = None
    try:
        etf_list = ["SPY", "QQQ", "IWM", "DIA", "HYG", "TLT", "SLV", "GLD"]
        conn = connect_db()
        for etf in etf_list:
            print(f"Processing {etf} on {today.strftime('%Y-%m-%d')}")
            result = process_ticker_wrapper(etf, today)
            if result:
                insert_option_metrics(conn, result)
                print(f"Result for {etf} saved to PostgreSQL database.")
            else:
                print(f"No valid data for {etf} on {today.strftime('%Y-%m-%d')}")
    except Exception as e:
        print("Error:", e)
    finally:
        if conn:
            conn.close()
