import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from scipy.stats import norm
from scipy.optimize import brentq

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

def compute_expected_move(underlying_price, iv, days_to_expiry):
    T = days_to_expiry / 365
    if iv == 0 or T <= 0:
        return 0
    return underlying_price * iv * np.sqrt(T)

def net_gamma_exposure(S, df_opt, r):
    total_gamma_exp = 0.0
    for idx, row in df_opt.iterrows():
        K = row['strike']
        T = row['daysToExpiry'] / 365
        sigma = row['impliedVolatility']
        oi = row['openInterest']
        option_type = row['optionType']
        gamma = black_scholes_gamma(S, K, T, r, sigma)
        gamma_exp = gamma * oi * 100  # contracts * contract size
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
    expirations_dt = [datetime.strptime(d, "%Y-%m-%d") for d in expirations]
    dte_dict = {d: (d - as_of_date).days for d in expirations_dt}

    history = tk.history(period='1d')
    if history.empty:
        print(f"No price history for {ticker}")
        return None
    underlying_price = history['Close'].iloc[0]

    one_day_expiries = [d for d in expirations_dt if 0 < dte_dict[d] <= 5]
    one_week_expiries = [d for d in expirations_dt if 5 < dte_dict[d] <= 9]
    one_month_expiries = [d for d in expirations_dt if 20 <= dte_dict[d] <= 30]

    def select_expiry_with_highest_oi(expiry_list):
        max_oi = -1
        chosen_exp = None
        for exp in expiry_list:
            opt_df = fetch_option_chain(ticker, exp.strftime("%Y-%m-%d"))
            opt_df = opt_df.dropna(subset=['openInterest'])
            oi_sum = opt_df['openInterest'].sum() if not opt_df.empty else 0
            if oi_sum > max_oi:
                max_oi = oi_sum
                chosen_exp = exp
        return chosen_exp

    selected_1d = select_expiry_with_highest_oi(one_day_expiries)
    selected_1w = select_expiry_with_highest_oi(one_week_expiries)
    selected_1m = select_expiry_with_highest_oi(one_month_expiries)

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
        df_filtered = filter_strikes_by_oi_and_atm(df_opt, underlying_price, top_n_strikes=20)
        if df_filtered.empty:
            print(f"No valid options data for expiry {expiry.strftime('%Y-%m-%d')}")
            return
        iv_values = df_filtered['impliedVolatility'].dropna()
        weights = df_filtered['openInterest'].dropna()
        if weights.sum() == 0 or len(weights) == 0:
            iv_weighted_avg = iv_values.mean() if len(iv_values) > 0 else 0
        else:
            iv_weighted_avg = np.average(iv_values, weights=weights)
        expected_move = compute_expected_move(underlying_price, iv_weighted_avg, dte)
        expected_ranges[f'{label_prefix}_high'] = underlying_price + expected_move
        expected_ranges[f'{label_prefix}_low'] = underlying_price - expected_move
        print(f"{label_prefix}: expiry={expiry.strftime('%Y-%m-%d')}, DTE={dte}, IV_avg={iv_weighted_avg:.4f}, Expected Move={expected_move:.4f}")

    process_expected_move(selected_1d, '1_day_ahead')
    process_expected_move(selected_1w, '1_wk_ahead')
    process_expected_move(selected_1m, '1_mth_ahead')

    # Gather aggregated data for gamma flip calculation
    all_expiries_for_gamma = sorted(set(one_day_expiries + one_week_expiries + one_month_expiries))
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
    r = process_ticker_wrapper("SPY", today)
    print(r)
