import os
import pandas as pd
import numpy as np
from scipy.stats import linregress
from datetime import timedelta

# Calculate project root from current file location, navigating up as needed

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

DATA_FOLDER = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, 'output')

# Make sure folders exist
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

TRADING_DAYS_PER_YEAR = 252
QUANTILE_CUT = 0.25
MOMENTUM_DAYS_3 = 60
MOMENTUM_DAYS_6 = 126
MOMENTUM_DAYS_12 = 252
CLENOW_WINDOW_DAYS = 100
TRANSACTION_COST_RATE = 0.001

def load_prices():
    file_path = os.path.join(DATA_FOLDER, "strat1_etf.csv")
    prices = pd.read_csv(file_path, index_col=0, parse_dates=True).sort_index()
    return prices

def calc_blended_momentum_daily(prices):
    mom_3m = prices.pct_change(MOMENTUM_DAYS_3, fill_method=None).shift(20).dropna(how='all')
    mom_6m = prices.pct_change(MOMENTUM_DAYS_6, fill_method=None).shift(20).dropna(how='all')
    mom_12m = prices.pct_change(MOMENTUM_DAYS_12, fill_method=None).shift(20).dropna(how='all')

    def standardize(df):
        return (df - df.mean(axis=1).values[:, None]) / df.std(axis=1).values[:, None]

    mom_3m_std = standardize(mom_3m)
    mom_6m_std = standardize(mom_6m)
    mom_12m_std = standardize(mom_12m)

    blended_score = 0.2 * mom_3m_std + 0.5 * mom_6m_std + 0.3 * mom_12m_std
    return blended_score.rank(axis=1, ascending=True, na_option='keep')

def clenow_momentum(prices, window=CLENOW_WINDOW_DAYS):
    log_prices = np.log(prices)
    clenow_scores = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for ticker in prices.columns:
        series = log_prices[ticker]
        scores = []
        for i in range(len(series)):
            if i < window - 1:
                scores.append(np.nan)
                continue
            y = series.iloc[i - window + 1: i + 1].values
            x = np.arange(window)
            slope, intercept, r_value, _, _ = linregress(x, y)
            annualized_slope = slope * TRADING_DAYS_PER_YEAR
            scores.append(annualized_slope * (r_value ** 2))
        clenow_scores[ticker] = scores
    return clenow_scores

def calc_annualized_volatility(prices, window=60):
    daily_ret = prices.pct_change(fill_method=None).dropna()
    rolling_vol = daily_ret.rolling(window=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return rolling_vol

def calc_carver_momentum(prices):
    mom_3m = prices.pct_change(MOMENTUM_DAYS_3, fill_method=None)
    mom_6m = prices.pct_change(MOMENTUM_DAYS_6, fill_method=None)
    mom_12m = prices.pct_change(MOMENTUM_DAYS_12, fill_method=None)

    vol_3m = calc_annualized_volatility(prices, window=MOMENTUM_DAYS_3).replace(0, np.nan)
    vol_6m = calc_annualized_volatility(prices, window=MOMENTUM_DAYS_6).replace(0, np.nan)
    vol_12m = calc_annualized_volatility(prices, window=MOMENTUM_DAYS_12).replace(0, np.nan)

    adj_mom_3m = mom_3m / vol_3m
    adj_mom_6m = mom_6m / vol_6m
    adj_mom_12m = mom_12m / vol_12m

    rank_3m = adj_mom_3m.rank(axis=1, ascending=True, na_option='keep')
    rank_6m = adj_mom_6m.rank(axis=1, ascending=True, na_option='keep')
    rank_12m = adj_mom_12m.rank(axis=1, ascending=True, na_option='keep')

    carver_score = rank_3m + rank_6m + rank_12m
    return carver_score

def calc_ewac_momentum_daily(prices):
    ema_pairs = [(12, 26), (24, 52), (36, 72)]
    combined_scores = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for short_win, long_win in ema_pairs:
        short_ema = prices.ewm(span=short_win, adjust=False).mean()
        long_ema = prices.ewm(span=long_win, adjust=False).mean()
        momentum = short_ema / long_ema
        rank = momentum.rank(axis=1, ascending=True, na_option='keep')
        combined_scores += rank
    ewac_scores = combined_scores / len(ema_pairs)
    return ewac_scores.dropna(how='all')

def calc_breadth_score(mom_scores, threshold=0):
    positive_mom = (mom_scores > threshold).sum(axis=1)
    total_etfs = mom_scores.notna().sum(axis=1)
    breadth_score = positive_mom / total_etfs
    return breadth_score

def hybrid_short_weight_control(date, breadth_score_series, base_short_weight=1.0):
    score = breadth_score_series.loc[date]
    lower_quantile = breadth_score_series.quantile(0.15)
    upper_quantile = breadth_score_series.quantile(0.85)
    if score <= lower_quantile:
        return base_short_weight * 0.2
    elif score >= upper_quantile:
        return base_short_weight * 0.6
    else: 
        return base_short_weight 

def get_custom_rebalance_dates(prices, n_rebalances=3):
    month_ends = prices.resample('ME').last().index
    last_data_date = prices.index[-1]
    rebalance_dates = []
    for month_end in month_ends:
        if month_end.month >= last_data_date.month and month_end.year == last_data_date.year:
            continue
        loc = prices.index.get_indexer([month_end], method='pad')[0]
        window_start_idx = max(0, loc - 6)
        candidate_days = prices.index[window_start_idx:loc + 1]
        selected_days = candidate_days[-n_rebalances:]
        rebalance_dates.extend(selected_days)
    return pd.to_datetime(sorted(set(rebalance_dates)))

def run_daily_with_monthly_signal_equal_weight(
    mom_scores, method_name, rebalance_dates,
    base_short_weight=1.0,
    top_quantile=QUANTILE_CUT,
    bottom_quantile=QUANTILE_CUT,
    progress_callback=None):

    prices = load_prices()
    daily_returns = prices.pct_change(fill_method=None).dropna()
    daily_returns_aligned, _ = daily_returns.align(prices, join='inner', axis=0)

    portfolio_returns = []
    prev_weights = pd.Series(0.0, index=prices.columns)
    rebalance_dates = pd.to_datetime(sorted(set(rebalance_dates)))
    rebalance_dates = [d for d in rebalance_dates if d in mom_scores.index and d in prices.index]

    breadth_score = calc_breadth_score(mom_scores)
    full_index = daily_returns_aligned.index
    daily_returns_dict = {}

    trade_logs = []

    total_rebalances = len(rebalance_dates)
    for i, date in enumerate(rebalance_dates):
        if progress_callback:
            progress_callback(int(90 * i / total_rebalances))
        m_scores = mom_scores.loc[date].dropna()

        top_cutoff = m_scores.quantile(1 - top_quantile)
        bottom_cutoff = m_scores.quantile(bottom_quantile)
        allowed_longs = m_scores[m_scores >= top_cutoff].index.tolist()
        allowed_shorts = m_scores[m_scores <= bottom_cutoff].index.tolist()

        if not allowed_longs:
            allowed_longs = [etf for etf in prev_weights[prev_weights > 0].index]
        if not allowed_shorts:
            allowed_shorts = [etf for etf in prev_weights[prev_weights < 0].index]

        short_weight_pct = hybrid_short_weight_control(date, breadth_score, base_short_weight)

        long_weight_val = 1.0 / len(allowed_longs) if len(allowed_longs) > 0 else 0
        short_weight_val = 1.0 / len(allowed_shorts) if len(allowed_shorts) > 0 else 0

        weights = pd.Series(0.0, index=prices.columns)
        for etf in allowed_longs:
            weights[etf] = long_weight_val
        for etf in allowed_shorts:
            weights[etf] = - short_weight_pct * short_weight_val

        turnover = np.abs(weights - prev_weights).sum()

        for etf, w in weights.items():
            if w != 0:
                pos = 'Long' if w > 0 else 'Short'
                trade_logs.append({'Date': date, 'ETF': etf, 'Position': pos,
                                   'Weight': w, 'Method': method_name,
                                   'Turnover': turnover})

        next_date = rebalance_dates[i + 1] if i < len(rebalance_dates) - 1 else prices.index[-1]

        mask = (daily_returns_aligned.index >= date) & (daily_returns_aligned.index < next_date)
        daily_rets = daily_returns_aligned.loc[mask]
        daily_portfolio_rets = (daily_rets * weights.values).sum(axis=1)

        if date in daily_portfolio_rets.index:
            daily_portfolio_rets.loc[date] -= turnover * TRANSACTION_COST_RATE

        for dt, ret in daily_portfolio_rets.items():
            daily_returns_dict[dt] = ret

        prev_weights = weights.copy()

    portfolio_returns_series = pd.Series(0.0, index=full_index)
    for dt, ret in daily_returns_dict.items():
        portfolio_returns_series.loc[dt] = ret

    cum_returns = (1 + portfolio_returns_series).cumprod() - 1

    trade_log_df = pd.DataFrame(trade_logs)
    trade_log_df.to_csv(os.path.join(OUTPUT_FOLDER,
                                     f'quantile_trade_log_{method_name}_equal_weight.csv'), index=False)

    if progress_callback:
        progress_callback(100)

    return pd.DataFrame({'Return': portfolio_returns_series}), cum_returns

def run_all(progress_callback=None):
    if progress_callback:
        progress_callback(0)

    prices = load_prices()
    if progress_callback:
        progress_callback(10)

    custom_rebalance_dates = get_custom_rebalance_dates(prices, n_rebalances=3)
    if progress_callback:
        progress_callback(20)

    blended_momentum = calc_blended_momentum_daily(prices)
    if progress_callback:
        progress_callback(35)

    clenow_ranks = clenow_momentum(prices)
    if progress_callback:
        progress_callback(45)

    carver_momentum = calc_carver_momentum(prices)
    if progress_callback:
        progress_callback(55)

    ewac_momentum = calc_ewac_momentum_daily(prices)
    if progress_callback:
        progress_callback(65)

    composite_rank = blended_momentum.rank(axis=1, ascending=True, na_option='keep') + \
                     clenow_ranks.rank(axis=1, ascending=True, na_option='keep') + \
                     carver_momentum.rank(axis=1, ascending=True, na_option='keep') + \
                     ewac_momentum.rank(axis=1, ascending=True, na_option='keep')

    if progress_callback:
        progress_callback(70)

    returns_blended_eq, cum_blended_eq = run_daily_with_monthly_signal_equal_weight(
        blended_momentum, 'blended_momentum', custom_rebalance_dates,
        base_short_weight=1,
        progress_callback=progress_callback
    )

    returns_clenow_eq, cum_clenow_eq = run_daily_with_monthly_signal_equal_weight(
        clenow_ranks, 'clenow_momentum', custom_rebalance_dates,
        base_short_weight=1,
        progress_callback=None
    )

    returns_carver_eq, cum_carver_eq = run_daily_with_monthly_signal_equal_weight(
        carver_momentum, 'carver_momentum', custom_rebalance_dates,
        base_short_weight=1,
        progress_callback=None
    )

    returns_ewac_eq, cum_ewac_eq = run_daily_with_monthly_signal_equal_weight(
        ewac_momentum, 'ewac_momentum', custom_rebalance_dates,
        base_short_weight=1,
        progress_callback=None
    )

    returns_composite_eq, cum_composite_eq = run_daily_with_monthly_signal_equal_weight(
        composite_rank, 'composite_momentum', custom_rebalance_dates,
        base_short_weight=1,
        progress_callback=None
    )

    daily_returns = prices.pct_change(fill_method=None).dropna()
    spy_daily_returns = daily_returns['SPY'].dropna()
    spy_cum_returns = (1 + spy_daily_returns).cumprod() - 1

    returns_dict = {
        'Blended': returns_blended_eq['Return'],
        'Clenow': returns_clenow_eq['Return'],
        'Carver': returns_carver_eq['Return'],
        'EWAC': returns_ewac_eq['Return'],
        'Composite': returns_composite_eq['Return'],
        'SPY': spy_daily_returns,
    }

    if progress_callback:
        progress_callback(100)

    return {
        'prices': prices,
        'returns_dict': returns_dict,
        'rebalance_dates': custom_rebalance_dates,
        'cum_returns': {
            'Blended': cum_blended_eq,
            'Clenow': cum_clenow_eq,
            'Carver': cum_carver_eq,
            'EWAC': cum_ewac_eq,
            'Composite': cum_composite_eq,
            'SPY': spy_cum_returns
        }
    }