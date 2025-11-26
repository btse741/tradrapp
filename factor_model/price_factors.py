import os
import yaml
import psycopg2
import pandas as pd
import numpy as np
import time 
import io
import gc
import multiprocessing as mp
from tqdm import tqdm
from sqlalchemy import create_engine
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
from numba import njit

def decide_mode(today=None):
    if today is None:
        today = date.today()
    yesterday = today - timedelta(days=1)
    month_changed = yesterday.month != today.month
    
    if today.weekday() == 1:  # Tuesday
        saturday = today - timedelta(days=3)
        sunday = today - timedelta(days=2)
        monday = today - timedelta(days=1)
        month_changed = any(d.month != today.month for d in [saturday, sunday, monday])
    
    if not month_changed:
        return None
    
    if today.weekday() == 1:
        for day in [monday, sunday, saturday]:
            if day.month != today.month:
                month_change_day = day
                break
    else:
        month_change_day = yesterday


    if (month_change_day.month == 1 or month_change_day.month == 7) and \
       month_change_day.weekday() == 4 and 1 <= month_change_day.day <= 7:
        return 'full'
    return 'incremental'


# Optimized KAMA calculation with caching and fastmath enabled.
@njit(cache=True, fastmath=True)
def calculate_kama_numba(prices, n=10, fast=2, slow=30):
    length = len(prices)
    kama = np.empty(length, dtype=np.float64)
    kama[:] = np.nan

    if length < n:
        return kama

    sc_fast = 2 / (fast + 1)
    sc_slow = 2 / (slow + 1)

    count = 0
    total = 0.0
    for i in range(n):
        if not np.isnan(prices[i]):
            total += prices[i]
            count += 1
    if count > 0:
        kama[n-1] = total / count
    else:
        kama[n-1] = np.nan

    for i in range(n, length):
        if np.isnan(prices[i]) or np.isnan(kama[i-1]):
            kama[i] = np.nan
            continue
        change = abs(prices[i] - prices[i - n])

        volatility = 0.0
        for j in range(i-n+1, i+1):
            if not np.isnan(prices[j]) and not np.isnan(prices[j-1]):
                volatility += abs(prices[j] - prices[j-1])
        er = change / volatility if volatility != 0 else 0
        sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2
        kama[i] = kama[i-1] + sc * (prices[i] - kama[i-1])
    return kama


def optimize_df_memory(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df.loc[:, col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int64']).columns:
        df.loc[:, col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df[col]) < 0.5:
            df.loc[:, col] = df[col].astype('category')


def batch_rolling_calculations(df):
    adj_close = df['adj_close']
    return pd.DataFrame({
        'sma_50d': adj_close.rolling(50, min_periods=1).mean(),
        'sma_200d': adj_close.rolling(200, min_periods=1).mean(),
        'ema_20d': adj_close.ewm(span=20, adjust=False).mean(),
        'ema_50d': adj_close.ewm(span=50, adjust=False).mean(),
        'ema_100d': adj_close.ewm(span=100, adjust=False).mean(),
        'ema_200d': adj_close.ewm(span=200, adjust=False).mean(),
    })


def compute_indicators(df):
    df = df.copy()

    df.loc[:, 'adjustment_ratio'] = df['adj_close'] / df['close']
    df.loc[:, 'adjustment_ratio'] = df['adjustment_ratio'].replace([np.inf, -np.inf], np.nan)
    df.loc[:, 'adj_high'] = df['high'] * df['adjustment_ratio']
    df.loc[:, 'adj_low'] = df['low'] * df['adjustment_ratio']

    df.loc[:, 'high_52w'] = df.groupby('ticker')['adj_high'].transform(lambda x: x.rolling(252, min_periods=1).max())
    df.loc[:, 'low_52w'] = df.groupby('ticker')['adj_low'].transform(lambda x: x.rolling(252, min_periods=1).min())

    df.loc[:, 'dist52hi'] = np.where(df['adj_close'] != 0, df['high_52w'] / df['adj_close'] - 1, np.nan)
    df.loc[:, 'dist52lo'] = np.where(df['low_52w'] != 0, df['adj_close'] / df['low_52w'] - 1, np.nan)

    df.loc[:, 'market_cap'] = df['adj_close'] * df['shares_outstanding']
    df.loc[:, 'typical_price'] = (df['adj_high'] + df['adj_low'] + df['adj_close']) / 3

    def rolling_vwap(x):
        pv = x['typical_price'] * x['volume']
        vol_sum = x['volume'].rolling(10, min_periods=1).sum().replace(0, np.nan)
        return pv.rolling(10, min_periods=1).sum() / vol_sum

    df.loc[:, 'vwap_10d'] = df.groupby('ticker').apply(rolling_vwap, include_groups=False).reset_index(level=0, drop=True)

    df.loc[:, 'month'] = df['date'].dt.to_period('M')

    def monthly_vwap(sub):
        cum_vol = sub['volume'].cumsum().replace(0, np.nan)
        cum_pv = (sub['typical_price'] * sub['volume']).cumsum()
        return cum_pv / cum_vol

    df.loc[:, 'vwap_mtd'] = df.groupby(['ticker', 'month']).apply(monthly_vwap, include_groups=False).reset_index(level=[0, 1], drop=True)

    rolling_df = df.groupby('ticker').apply(batch_rolling_calculations, include_groups=False).reset_index(level=0, drop=True)
    df = df.join(rolling_df)

    df.loc[:, 'kama_10'] = df.groupby('ticker')['adj_close'].transform(
        lambda x: pd.Series(calculate_kama_numba(x.values, n=10, fast=2, slow=20), index=x.index))
    df.loc[:, 'kama_50'] = df.groupby('ticker')['adj_close'].transform(
        lambda x: pd.Series(calculate_kama_numba(x.values, n=50, fast=4, slow=50), index=x.index))

    df.loc[:, 'daily_turnover'] = np.where(df['shares_outstanding'] != 0, df['volume'] / df['shares_outstanding'], np.nan)
    df.loc[:, 'daily_return'] = df.groupby('ticker')['adj_close'].pct_change(fill_method=None)

    mask_invalid = (~np.isfinite(df['daily_return'])) | (df['daily_return'] == 0)
    df.loc[:, 'illiquidity'] = np.where(mask_invalid, np.nan, 1 / df['daily_return'].abs())

    df.loc[:, 'prev_close'] = df.groupby('ticker')['adj_close'].shift(1)
    df.loc[:, 'tr1'] = df['adj_high'] - df['adj_low']
    df.loc[:, 'tr2'] = (df['adj_high'] - df['prev_close']).abs()
    df.loc[:, 'tr3'] = (df['adj_low'] - df['prev_close']).abs()
    df.loc[:, 'true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df.loc[:, 'atr_14d'] = df.groupby('ticker')['true_range'].transform(lambda x: x.rolling(14, min_periods=14).mean())

    df.loc[:, 'range_volatility'] = np.where(df['adj_close'] != 0, (df['adj_high'] - df['adj_low']) / df['adj_close'], np.nan)
    df.loc[:, 'std_30d'] = df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(30, min_periods=20).std())
    df.loc[:, 'std_3m'] = df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(63, min_periods=45).std())
    df.loc[:, 'std_6m'] = df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(126, min_periods=90).std())
    df.loc[:, 'std_12m'] = df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(252, min_periods=182).std())

    optimize_df_memory(df)

    return df


def safe_momentum_calc(series, lag1, lag2):
    prev1 = series.shift(lag1)
    prev2 = series.shift(lag2)
    valid = (prev2 != 0) & prev2.notna() & prev1.notna()
    result = pd.Series(np.nan, index=series.index)
    result.loc[valid] = (prev1.loc[valid] / prev2.loc[valid] - 1) * 100
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def safe_pct_change(s, lag):
    prev = s.shift(lag)
    valid = (prev != 0) & prev.notna()
    result = pd.Series(np.nan, index=s.index)
    result.loc[valid] = ((s.loc[valid] - prev.loc[valid]) / prev.loc[valid]) * 100
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def compute_monthly_factors(df_daily):
    
    df_daily = df_daily.copy()
    df_daily['month_end'] = df_daily['date'].dt.to_period('M').apply(lambda r: r.end_time.date())

    monthly_avg = df_daily.groupby(['ticker', 'month_end'])[['daily_turnover', 'illiquidity']].mean().reset_index()
    monthly_avg.rename(columns={'month_end': 'factor_date', 
                                'daily_turnover': 'avg_daily_turnover', 
                                'illiquidity': 'avg_illiquidity',
                                'volume': 'avg_volume'}, inplace=True)

    last_adj_close = df_daily.groupby(['ticker', 'month_end']).last().reset_index()[['ticker', 'month_end', 'adj_close']]
    last_adj_close.rename(columns={'month_end': 'factor_date'}, inplace=True)

    last_market_cap = df_daily.groupby(['ticker', 'month_end']).last().reset_index()[['ticker', 'month_end', 'market_cap']]
    last_market_cap.rename(columns={'month_end': 'factor_date'}, inplace=True)

    monthly_avg = monthly_avg.merge(last_adj_close, on=['ticker', 'factor_date'], how='left')
    monthly_avg = monthly_avg.merge(last_market_cap, on=['ticker', 'factor_date'], how='left')  
    gc.collect()

    monthly_avg['mom_12m'] = safe_momentum_calc(monthly_avg['adj_close'], 1, 12)
    monthly_avg['mom_6m'] = safe_momentum_calc(monthly_avg['adj_close'], 1, 7)
    monthly_avg['mom_3m'] = safe_momentum_calc(monthly_avg['adj_close'], 1, 4)

    lag_map = {
        'pct_change_1m': 1, 'pct_change_3m': 3, 'pct_change_6m': 6,
        'pct_change_12m': 12, 'pct_change_24m': 24, 'pct_change_60m': 60
    }

    for factor_name, lag in lag_map.items():
        monthly_avg[factor_name] = safe_pct_change(monthly_avg['adj_close'], lag)

    return monthly_avg


def load_daily_prices(engine, start_date, end_date):
    sql = """
    SELECT ticker, date, adj_close, close, shares_outstanding, volume, high, low
    FROM daily_prices
    WHERE date BETWEEN %s AND %s
    ORDER BY ticker, date;
    """
    df = pd.read_sql_query(sql, engine, params=(start_date, end_date))
    df['date'] = pd.to_datetime(df['date'])
    return df


def process_factor_df_in_chunks(conn, df, factor_names, date_col='date',
                               target_table='daily_factors',
                               conflict_cols=['ticker', 'factor_date', 'factor_name'],
                               update_cols=['factor_value'],
                               db_params=None,
                               factor_group_name=""):

    df[date_col] = pd.to_datetime(df[date_col])
    months = df[date_col].dt.to_period('M').sort_values().unique()

    batch_size = 1  # 1 month per batch for controlled memory and progress tracking
    batched_months = [months[i:i+batch_size] for i in range(0, len(months), batch_size)]

    for month_batch in tqdm(batched_months, desc=f'Processing {factor_group_name} month batches'):
        df_batch = df[df[date_col].dt.to_period('M').isin(month_batch)]
        if df_batch.empty:
            continue
        
        cols = ['ticker', date_col] + factor_names
        df_factor = df_batch[cols].copy()
        df_melted = df_factor.melt(id_vars=['ticker', date_col], var_name='factor_name', value_name='factor_value')
        if date_col != 'factor_date':
            df_melted.rename(columns={date_col: 'factor_date'}, inplace=True)

        df_melted.drop_duplicates(subset=['ticker', 'factor_date', 'factor_name'], inplace=True)
        df_melted.dropna(subset=['ticker'], inplace=True)
        df_melted.sort_values(by=['ticker', 'factor_date', 'factor_name'], inplace=True)

        safe_month = "_".join(str(m) for m in month_batch).replace('-', '_')
        temp_table = f"temp_upsert_{safe_month}"

        try:
            with conn.cursor() as cur:
                cur.execute("SET maintenance_work_mem = '256MB';")
                cur.execute("SET synchronous_commit = OFF;")
                cur.execute("SET temp_buffers = '16MB';")
                cur.execute(f"CREATE TEMP TABLE {temp_table} (LIKE {target_table} INCLUDING DEFAULTS) ON COMMIT DROP;")

                f = io.BytesIO()
                csv_bytes = df_melted.to_csv(sep='\t', header=False, index=False, na_rep='\\N').encode('utf-8')
                f.write(csv_bytes)
                f.seek(0)
                cur.copy_from(f, temp_table, null='\\N', sep='\t')

                updates = ', '.join([f"{col} = EXCLUDED.{col}" for col in update_cols])
                conflict_keys = ', '.join(conflict_cols)

                upsert_sql = f"""
                    INSERT INTO {target_table} (ticker, factor_date, factor_name, factor_value)
                    SELECT ticker, factor_date, factor_name, factor_value FROM {temp_table}
                    ON CONFLICT ({conflict_keys}) DO UPDATE SET {updates};
                """
                cur.execute(upsert_sql)
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"Error in bulk upsert for months {month_batch}: {e}")
            raise

        del df_batch, df_factor, df_melted
        gc.collect()

def worker_upsert(df_chunk, factor_names, date_col, target_table, conflict_cols, update_cols, db_params, factor_group_name):
    max_retries = 3
    attempt = 0
    while attempt < max_retries:
        try:
            conn = psycopg2.connect(**db_params)
            try:
                process_factor_df_in_chunks(conn, df_chunk, factor_names, date_col, target_table,
                                           conflict_cols, update_cols, db_params, factor_group_name)
            finally:
                conn.close()
            break  # success, exit retry loop
        except psycopg2.InterfaceError as e:
            attempt += 1
            if attempt >= max_retries:
                raise
            time.sleep(5)  # wait before retrying

def parallel_process_factor_df_in_chunks(df, factor_names, date_col='date',
                                        target_table='daily_factors',
                                        conflict_cols=['ticker', 'factor_date', 'factor_name'],
                                        update_cols=['factor_value'],
                                        db_params=None,
                                        n_jobs=None,
                                        factor_group_name=""):
    if n_jobs is None:
        n_jobs = max(mp.cpu_count() - 1, 1)

    unique_tickers = df['ticker'].unique()
    chunks = np.array_split(unique_tickers, n_jobs)
    df_chunks = [df[df['ticker'].isin(tickers)].copy() for tickers in chunks]

    args = [(chunk, factor_names, date_col, target_table, conflict_cols, update_cols, db_params, factor_group_name) for chunk in df_chunks]

    with mp.Pool(n_jobs) as pool:
        list(tqdm(pool.starmap(worker_upsert, args), total=len(args), desc=f'Parallel upsert {factor_group_name}'))
    
    gc.collect

# The monthly equivalents can be similarly updated, here is the monthly factor chunk processing:

def process_monthly_factor_df_in_chunks(conn, df, factor_names, date_col='factor_date',
                                       target_table='monthly_factors',
                                       conflict_cols=['ticker', 'factor_date', 'factor_name'],
                                       update_cols=['factor_value'],
                                       db_params=None,
                                       factor_group_name=""):

    df[date_col] = pd.to_datetime(df[date_col])
    months = df[date_col].dt.to_period('M').sort_values().unique()

    batch_size = 1
    batched_months = [months[i:i+batch_size] for i in range(0, len(months), batch_size)]

    for month_batch in tqdm(batched_months, desc=f'Processing {factor_group_name} monthly batches'):
        df_batch = df[df[date_col].dt.to_period('M').isin(month_batch)]
        if df_batch.empty:
            continue

        cols = ['ticker', date_col] + factor_names
        df_factor = df_batch[cols].copy()
        df_melted = df_factor.melt(id_vars=['ticker', date_col], var_name='factor_name', value_name='factor_value')
        if date_col != 'factor_date':
            df_melted.rename(columns={date_col: 'factor_date'}, inplace=True)

        df_melted.drop_duplicates(subset=['ticker', 'factor_date', 'factor_name'], inplace=True)
        df_melted.dropna(subset=['ticker'], inplace=True)
        df_melted.sort_values(by=['ticker', 'factor_date', 'factor_name'], inplace=True)

        safe_month = "_".join(str(m) for m in month_batch).replace('-', '_')
        temp_table = f"temp_upsert_monthly_{safe_month}"

        try:
            with conn.cursor() as cur:
                cur.execute("SET maintenance_work_mem = '256MB';")
                cur.execute("SET synchronous_commit = OFF;")
                cur.execute("SET temp_buffers = '16MB';")
                cur.execute(f"CREATE TEMP TABLE {temp_table} (LIKE {target_table} INCLUDING DEFAULTS) ON COMMIT DROP;")

                f = io.BytesIO()
                csv_bytes = df_melted.to_csv(sep='\t', header=False, index=False, na_rep='\\N').encode('utf-8')
                f.write(csv_bytes)
                f.seek(0)
                cur.copy_from(f, temp_table, null='\\N', sep='\t')

                updates = ', '.join([f"{col} = EXCLUDED.{col}" for col in update_cols])
                conflict_keys = ', '.join(conflict_cols)

                upsert_sql = f"""
                    INSERT INTO {target_table} (ticker, factor_date, factor_name, factor_value)
                    SELECT ticker, factor_date, factor_name, factor_value FROM {temp_table}
                    ON CONFLICT ({conflict_keys}) DO UPDATE SET {updates};
                """
                cur.execute(upsert_sql)
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"Error in bulk upsert for monthly batch {month_batch}: {e}")
            raise

        del df_batch, df_factor, df_melted
        gc.collect()

def worker_monthly_upsert(df_chunk, factor_names, date_col, target_table, conflict_cols, update_cols, db_params, factor_group_name):
    import psycopg2
    conn = psycopg2.connect(**db_params)
    try:
        process_monthly_factor_df_in_chunks(conn, df_chunk, factor_names, date_col, target_table,
                                           conflict_cols, update_cols, db_params, factor_group_name)
    finally:
        conn.close()

def parallel_process_monthly_factor_df_in_chunks(df, factor_names, date_col='factor_date',
                                                target_table='monthly_factors',
                                                conflict_cols=['ticker', 'factor_date', 'factor_name'],
                                                update_cols=['factor_value'],
                                                db_params=None,
                                                n_jobs=None,
                                                factor_group_name=""):
    if n_jobs is None:
        n_jobs = max(mp.cpu_count() - 1, 1)

    df[date_col] = pd.to_datetime(df[date_col])
    months = df[date_col].dt.to_period('M').sort_values().unique()

    batch_size = 1
    batched_months = [months[i:i+batch_size] for i in range(0, len(months), batch_size)]

    df_batches = [df[df[date_col].dt.to_period('M').isin(batch)].copy() for batch in batched_months]

    args = [(batch, factor_names, date_col, target_table, conflict_cols, update_cols, db_params, factor_group_name)
            for batch in df_batches if not batch.empty]

    with mp.Pool(n_jobs) as pool:
        list(tqdm(pool.starmap(worker_monthly_upsert, args), total=len(args), desc=f'Parallel monthly upsert {factor_group_name}'))


def date_chunks(start_date, end_date, chunk_size_months=1):
    current_start = start_date
    while current_start <= end_date:
        current_end = min(current_start + relativedelta(months=chunk_size_months) - timedelta(days=1), end_date)
        yield current_start, current_end
        current_start = current_end + timedelta(days=1)

MAX_DAILY_LOOKBACK_MONTHS = 84  # extended to 6 years for monthly factor lookbacks
MAX_MONTHLY_LOOKBACK_MONTHS = 84

daily_factors = [
    'adj_close', 'market_cap', 'dist52hi', 'dist52lo', 'vwap_10d', 'vwap_mtd',
    'sma_50d', 'sma_200d', 'ema_20d', 'ema_50d', 'ema_100d', 'ema_200d',
    'kama_10', 'kama_50', 'daily_turnover', 'atr_14d', 'illiquidity',
    'range_volatility', 'std_30d', 'std_3m', 'std_6m', 'std_12m'
]

monthly_factors = [
    'adj_close', 'market_cap', 
    'avg_daily_turnover', 'avg_illiquidity', 
    'mom_12m', 'mom_6m', 'mom_3m', 
    'pct_change_1m', 'pct_change_3m', 'pct_change_6m', 'pct_change_12m', 'pct_change_24m', 'pct_change_60m'
]

def ensure_timestamp(dt):
    if not isinstance(dt, pd.Timestamp):
        return pd.Timestamp(dt)
    return dt

def run_full_update(conn, engine, start_date, end_date, db_params):
    start_date = ensure_timestamp(start_date)
    end_date = ensure_timestamp(end_date)

    print(f"Starting full update from {start_date.date()} to {end_date.date()} in chunks...")
    batch_period_months = 24

    for chunk_start, chunk_end in date_chunks(start_date, end_date, chunk_size_months=batch_period_months):
        chunk_start = ensure_timestamp(chunk_start)
        chunk_end = ensure_timestamp(chunk_end)

        extended_daily_start = max(chunk_start - relativedelta(months=MAX_DAILY_LOOKBACK_MONTHS), pd.Timestamp('2000-01-01'))
        print(f"Loading daily prices from {extended_daily_start.date()} to {chunk_end.date()} ...")

        df_daily = load_daily_prices(engine, extended_daily_start, chunk_end)
        df_indicators_full = compute_indicators(df_daily)
        gc.collect()

        df_indicators = df_indicators_full[df_indicators_full['date'] >= chunk_start]
        gc.collect()

        print("Starting daily factors upsert...")
        parallel_process_factor_df_in_chunks(
            df_indicators, daily_factors, date_col='date',
            target_table='daily_factors',
            conflict_cols=['ticker', 'factor_date', 'factor_name'],
            update_cols=['factor_value'], db_params=db_params,
            factor_group_name="daily factors"
        )
        print("Daily factors upsert complete.")

        extended_monthly_start = max(chunk_start - relativedelta(months=MAX_MONTHLY_LOOKBACK_MONTHS), pd.Timestamp('2000-01-01'))
        df_daily_for_monthly = df_indicators_full[df_indicators_full['date'] >= extended_monthly_start]

        df_monthly = compute_monthly_factors(df_daily_for_monthly)
        df_monthly['factor_date'] = pd.to_datetime(df_monthly['factor_date'])

        df_monthly_filtered = df_monthly[(df_monthly['factor_date'] >= chunk_start) & (df_monthly['factor_date'] <= chunk_end)]

        print(f"Starting monthly factors upsert for chunk {chunk_start.date()} to {chunk_end.date()}...")
        parallel_process_factor_df_in_chunks(
            df_monthly_filtered, monthly_factors, date_col='factor_date',
            target_table='monthly_factors',
            conflict_cols=['ticker', 'factor_date', 'factor_name'],
            update_cols=['factor_value'], db_params=db_params,
            factor_group_name=f"monthly factors chunk {chunk_start.date()} to {chunk_end.date()}"
        )
        print("Monthly factors upsert complete for chunk.")
    print("Full update complete.")


def run_incremental_update(conn, engine, start_date, end_date, db_params):
    batch_start = ensure_timestamp(start_date)
    batch_end = ensure_timestamp(end_date)

    extended_daily_start = max(batch_start - relativedelta(months=MAX_DAILY_LOOKBACK_MONTHS), pd.Timestamp('2000-01-01'))
    print(f"Loading daily prices from {extended_daily_start.date()} to {batch_end.date()} ...")

    df_daily = load_daily_prices(engine, extended_daily_start, batch_end)
    df_indicators_full = compute_indicators(df_daily)
    gc.collect()

    df_indicators = df_indicators_full[df_indicators_full['date'] >= batch_start]
    gc.collect()

    print("Starting incremental daily factors upsert...")
    parallel_process_factor_df_in_chunks(
        df_indicators, daily_factors, date_col='date',
        target_table='daily_factors',
        conflict_cols=['ticker', 'factor_date', 'factor_name'],
        update_cols=['factor_value'], db_params=db_params,
        factor_group_name="incremental daily factors"
    )
    print("Incremental daily factors upsert complete.")

    latest_month = batch_end.to_period('M').to_timestamp()
    extended_monthly_start = max(latest_month - relativedelta(months=MAX_MONTHLY_LOOKBACK_MONTHS), pd.Timestamp('2000-01-01'))

    df_daily_for_monthly = df_indicators_full[df_indicators_full['date'] >= extended_monthly_start]

    df_monthly = compute_monthly_factors(df_daily_for_monthly)
    df_monthly['factor_date'] = pd.to_datetime(df_monthly['factor_date'])

    df_monthly_filtered = df_monthly[(df_monthly['factor_date'] >= latest_month) & (df_monthly['factor_date'] <= batch_end)]

    print(f"Starting monthly factors upsert for latest month {latest_month.date()} ...")
    parallel_process_factor_df_in_chunks(
        df_monthly_filtered, monthly_factors, date_col='factor_date',
        target_table='monthly_factors',
        conflict_cols=['ticker', 'factor_date', 'factor_name'],
        update_cols=['factor_value'], db_params=db_params,
        factor_group_name="incremental monthly factors"
    )
    print("Incremental monthly factors upsert complete.")
    print("Incremental update complete.")


def run_daily_incremental_update(conn, engine, db_params):
    update_day = ensure_timestamp(datetime.today().date() - timedelta(days=5))
    batch_start = ensure_timestamp(update_day - relativedelta(years=1))

    extended_daily_start = max(batch_start - relativedelta(months=MAX_DAILY_LOOKBACK_MONTHS), pd.Timestamp('2000-01-01'))
    print(f"Loading daily prices from {extended_daily_start.date()} to {update_day.date()} ...")

    df_daily = load_daily_prices(engine, extended_daily_start, update_day)
    df_indicators_full = compute_indicators(df_daily)
    gc.collect()   

    df_indicators = df_indicators_full[df_indicators_full['date'] >= batch_start]
    gc.collect()

    print("Starting daily incremental factors upsert...")
    parallel_process_factor_df_in_chunks(
        df_indicators, daily_factors, date_col='date',
        target_table='daily_factors',
        conflict_cols=['ticker', 'factor_date', 'factor_name'],
        update_cols=['factor_value'], db_params=db_params,
        factor_group_name="daily incremental factors"
    )
    print("Daily incremental factors upsert complete.")

if __name__ == "__main__":
    today = datetime.today().date()
    start = date(2020, 1, 1)
    end = today

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(project_root, 'config.yml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    params = config['database']
    conn_str = f"postgresql+psycopg2://{params['user']}:{params['password']}@{params['host']}:{params['port']}/{params['dbname']}"
    engine = create_engine(conn_str)

    db_connect_params = {
        'dbname': params['dbname'],
        'user': params['user'],
        'password': params['password'],
        'host': params['host'],
        'port': params['port']
    }

    conn = psycopg2.connect(**db_connect_params)
    conn.set_client_encoding('UTF8')

    try:
        mode = decide_mode()
        # mode = 'full'  # Change as needed
        if mode == 'full':
            run_full_update(conn, engine, start, end, db_connect_params)
        elif mode == 'incremental':
            since_date = datetime.today().date() - relativedelta(months=24)
            run_incremental_update(conn, engine, since_date, end, db_connect_params)

        run_daily_incremental_update(conn, engine, db_connect_params)
    finally:
        conn.close()
        gc.collect()
        print("DB connection closed.")
