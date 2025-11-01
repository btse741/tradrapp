import os
import yaml
import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np
import io
import gc
from tqdm import tqdm
from sqlalchemy import create_engine
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta


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
    if (month_change_day.month == 1 or month_change_day.month == 7) and month_change_day.weekday() == 4 and 1 <= month_change_day.day <= 7:
        return 'full'
    return 'incremental'


def bulk_copy_from_df(conn, df, table_name):
    print(f"Starting bulk copy to {table_name} with {len(df)} rows...")

    # Sanitize string columns
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].astype(str).str.replace('\n', ' ', regex=False).str.replace('\t', ' ', regex=False)

    # Replace infinite values
    df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)

    # Ensure date columns are string formatted as YYYY-MM-DD
    if 'factor_date' in df.columns:
        df['factor_date'] = pd.to_datetime(df['factor_date']).dt.strftime('%Y-%m-%d')

    # Preview sample data in DataFrame form
    print("Sample data:\n", df.head(3))

    # Use BytesIO and encode CSV as utf-8 bytes
    f = io.BytesIO()
    csv_bytes = df.to_csv(sep='\t', header=False, index=False, na_rep='\\N').encode('utf-8')
    f.write(csv_bytes)
    f.seek(0)

    try:
        with conn.cursor() as cur:
            cur.copy_from(f, table_name, null='\\N', sep='\t')
        conn.commit()
        print(f"Bulk copy to {table_name} successful.")
    except Exception as e:
        conn.rollback()
        print(f"Error during bulk copy to {table_name}: {e}")
        raise



def upsert_from_staging(conn, staging_table, target_table, conflict_cols, update_cols):
    updates = ', '.join([f"{col} = EXCLUDED.{col}" for col in update_cols])
    conflict_keys = ', '.join(conflict_cols)
    sql = f"""
    INSERT INTO {target_table} (ticker, factor_date, factor_name, factor_value)
    SELECT ticker, factor_date, factor_name, factor_value FROM {staging_table}
    ON CONFLICT ({conflict_keys})
    DO UPDATE SET {updates};
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        print(f"Upsert from {staging_table} to {target_table} successful.")
    except psycopg2.Error as e:
        conn.rollback()
        print(f"Error during upsert from {staging_table}: {e.pgcode} - {e.pgerror}")
        raise e


def clear_staging_table(conn, staging_table):
    try:
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE {staging_table};")
        conn.commit()
        print(f"Staging table {staging_table} cleared.")
    except psycopg2.Error as e:
        conn.rollback()
        print(f"Error clearing staging table {staging_table}: {e.pgcode} - {e.pgerror}")
        raise e


def load_daily_prices(engine, start_date, end_date):
    print(f"Loading daily prices from {start_date} to {end_date}...")
    sql = """
    SELECT ticker, date, adj_close, close, shares_outstanding, volume, high, low
    FROM daily_prices
    WHERE date BETWEEN %s AND %s
    ORDER BY ticker, date;
    """
    df = pd.read_sql_query(sql, engine, params=(start_date, end_date))
    df['date'] = pd.to_datetime(df['date'])
    return df


def calculate_kama(prices, n=10, fast=2, slow=30):
    change = prices.diff(n).abs()
    volatility = prices.diff().abs().rolling(window=n).sum()
    er = change / volatility
    er = er.replace([np.inf, -np.inf], np.nan)  # catch infinite values

    sc_fast = 2 / (fast + 1)
    sc_slow = 2 / (slow + 1)
    sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2
    sc = sc.replace([np.inf, -np.inf], np.nan)  # also catch here

    kama = pd.Series(index=prices.index, dtype=float)
    kama.iloc[:n] = prices.iloc[:n].mean()
    for i in range(n, len(prices)):
        if pd.isna(sc.iloc[i]) or pd.isna(prices.iloc[i]) or pd.isna(kama.iloc[i-1]):
            kama.iloc[i] = np.nan  # prevent infinite propagation
        else:
            kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (prices.iloc[i] - kama.iloc[i - 1])
    return kama


def safe_momentum_calc(series, lag1, lag2):
    
    prev1 = series.shift(lag1)
    prev2 = series.shift(lag2)
    valid = (prev2 != 0) & prev2.notna() & prev1.notna()
    result = pd.Series(np.nan, index=series.index)
    valid_idxs = valid[valid].index
    result.loc[valid_idxs] = (prev1.loc[valid_idxs] / prev2.loc[valid_idxs] - 1) * 100
    result = result.replace([np.inf, -np.inf], np.nan)
    return result 


def safe_pct_change(s, lag):
    
    prev = s.shift(lag)
    valid = (prev != 0) & prev.notna()
    result = pd.Series(np.nan, index=s.index)
    valid_idxs = valid[valid].index
    result.loc[valid_idxs] = ((s.loc[valid_idxs] - prev.loc[valid_idxs]) / prev.loc[valid_idxs]) * 100
    result = result.replace([np.inf, -np.inf], np.nan)
    return result

def compute_indicators(df):
    print("Computing technical indicators...")
    df['date'] = pd.to_datetime(df['date'])
    df['adjustment_ratio'] = df['adj_close'] / df['close']
    df['adjustment_ratio'] = df['adjustment_ratio'].replace([np.inf, -np.inf], np.nan)

    df['adj_high'] = df['high'] * df['adjustment_ratio']
    df['adj_low'] = df['low'] * df['adjustment_ratio']

    df['high_52w'] = df.groupby('ticker', observed=True)['adj_high'].transform(lambda x: x.rolling(252, min_periods=1).max())
    df['high_52w'] = df['high_52w'].replace([np.inf, -np.inf], np.nan)

    df['low_52w'] = df.groupby('ticker', observed=True)['adj_low'].transform(lambda x: x.rolling(252, min_periods=1).min())
    df['low_52w'] = df['low_52w'].replace([np.inf, -np.inf], np.nan)

    df['dist52hi'] = np.where(df['adj_close'] != 0, df['high_52w'] / df['adj_close'] - 1, np.nan)
    df['dist52hi'] = df['dist52hi'].replace([np.inf, -np.inf], np.nan)

    df['dist52lo'] = np.where(df['low_52w'] != 0, df['adj_close'] / df['low_52w'] - 1, np.nan)
    df['dist52lo'] = df['dist52lo'].replace([np.inf, -np.inf], np.nan)

    df['market_cap'] = df['adj_close'] * df['shares_outstanding']

    df['typical_price'] = (df['adj_high'] + df['adj_low'] + df['adj_close']) / 3

    def rolling_vwap(x):
        pv = x['typical_price'] * x['volume']
        vol_sum = x['volume'].rolling(10, min_periods=1).sum().replace(0, np.nan)
        return pv.rolling(10, min_periods=1).sum().div(vol_sum)

    df['vwap_10d'] = df.groupby('ticker', observed=True).apply(rolling_vwap, include_groups=False).reset_index(level=0, drop=True)

    df['month'] = df['date'].dt.to_period('M')

    def monthly_vwap(sub):
        cum_vol = sub['volume'].cumsum().replace(0, np.nan)
        cum_pv = (sub['typical_price'] * sub['volume']).cumsum()
        return cum_pv.div(cum_vol)

    df['vwap_mtd'] = df.groupby(['ticker', 'month'], observed=True).apply(monthly_vwap, include_groups=False).reset_index(level=[0, 1], drop=True)

    df['sma_50d'] = df.groupby('ticker', observed=True)['adj_close'].transform(lambda x: x.rolling(50, min_periods=1).mean())
    df['sma_200d'] = df.groupby('ticker', observed=True)['adj_close'].transform(lambda x: x.rolling(200, min_periods=1).mean())
    df['ema_20d'] = df.groupby('ticker', observed=True)['adj_close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    df['ema_50d'] = df.groupby('ticker', observed=True)['adj_close'].transform(lambda x: x.ewm(span=50, adjust=False).mean())
    df['ema_100d'] = df.groupby('ticker', observed=True)['adj_close'].transform(lambda x: x.ewm(span=100, adjust=False).mean())
    df['ema_200d'] = df.groupby('ticker', observed=True)['adj_close'].transform(lambda x: x.ewm(span=200, adjust=False).mean())

    df['kama_10'] = df.groupby('ticker', observed=True)['adj_close'].transform(lambda x: calculate_kama(x, n=10, fast=2, slow=20))
    df['kama_50'] = df.groupby('ticker', observed=True)['adj_close'].transform(lambda x: calculate_kama(x, n=50, fast=4, slow=50))

    df['daily_turnover'] = np.where(df['shares_outstanding'] != 0, df['volume'] / df['shares_outstanding'], np.nan)
    df['daily_turnover'] = df['daily_turnover'].replace([np.inf, -np.inf], np.nan)

    df['daily_return'] = df.groupby('ticker', observed=True)['adj_close'].pct_change(fill_method=None)
    df['daily_return'] = df['daily_return'].replace([np.inf, -np.inf], np.nan)

    mask_invalid = (~np.isfinite(df['daily_return'])) | (df['daily_return'] == 0)
    df['illiquidity'] = np.where(mask_invalid, np.nan, 1 / df['daily_return'].abs())
    df['illiquidity'] = df['illiquidity'].replace([np.inf, -np.inf], np.nan)

    df['prev_close'] = df.groupby('ticker', observed=True)['adj_close'].shift(1)
    df['tr1'] = df['adj_high'] - df['adj_low']
    df['tr2'] = (df['adj_high'] - df['prev_close']).abs()
    df['tr3'] = (df['adj_low'] - df['prev_close']).abs()
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr_14d'] = df.groupby('ticker', observed=True)['true_range'].transform(lambda x: x.rolling(14, min_periods=14).mean())
    df['atr_14d'] = df['atr_14d'].replace([np.inf, -np.inf], np.nan)

    df['range_volatility'] = np.where(df['adj_close'] != 0, (df['adj_high'] - df['adj_low']) / df['adj_close'], np.nan)
    df['range_volatility'] = df['range_volatility'].replace([np.inf, -np.inf], np.nan)

    df['std_30d'] = df.groupby('ticker', observed=True)['adj_close'].transform(lambda x: x.rolling(30, min_periods=20).std())
    df['std_3m'] = df.groupby('ticker', observed=True)['adj_close'].transform(lambda x: x.rolling(63, min_periods=45).std())
    df['std_6m'] = df.groupby('ticker', observed=True)['adj_close'].transform(lambda x: x.rolling(126, min_periods=90).std())
    df['std_12m'] = df.groupby('ticker', observed=True)['adj_close'].transform(lambda x: x.rolling(252, min_periods=182).std())

    return df

def compute_monthly_factors(df_daily):
    print("Computing monthly factors...")
    df_daily['month_end'] = df_daily['date'].dt.to_period('M').apply(lambda r: r.end_time.date())

    monthly_avg = df_daily.groupby(['ticker', 'month_end'], observed=True)[['daily_turnover', 'illiquidity']].mean().reset_index()

    monthly_avg = monthly_avg.rename(columns={
        'month_end': 'factor_date',
        'daily_turnover': 'avg_daily_turnover',
        'illiquidity': 'avg_illiquidity'
    })

    last_adj_close = df_daily.groupby(['ticker', 'month_end'], observed=True).last().reset_index()[['ticker', 'month_end', 'adj_close']]
    last_adj_close.rename(columns={'month_end': 'factor_date'}, inplace=True)

    monthly_avg = monthly_avg.merge(last_adj_close, on=['ticker', 'factor_date'], how='left')

    monthly_avg['mom_12m'] = safe_momentum_calc(monthly_avg['adj_close'], 1, 12).replace([np.inf, -np.inf], np.nan)
    monthly_avg['mom_6m'] = safe_momentum_calc(monthly_avg['adj_close'], 1, 7).replace([np.inf, -np.inf], np.nan)
    monthly_avg['mom_3m'] = safe_momentum_calc(monthly_avg['adj_close'], 1, 4).replace([np.inf, -np.inf], np.nan)

    lags = {
        'pct_change_1m': 1,
        'pct_change_3m': 3,
        'pct_change_6m': 6,
        'pct_change_12m': 12,
        'pct_change_24m': 24,
        'pct_change_60m': 60
    }
    for name, lag in lags.items():
        monthly_avg[name] = safe_pct_change(monthly_avg['adj_close'], lag).replace([np.inf, -np.inf], np.nan)

    return monthly_avg


def prepare_factor_df(df, factor_names, date_col='date'):
    print("Preparing factor DataFrame for upsert...")
    cols = ['ticker', date_col] + factor_names
    df_factor = df[cols].copy()
    df_melted = df_factor.melt(id_vars=['ticker', date_col],
                               var_name='factor_name', value_name='factor_value')
    if date_col != 'factor_date':
        df_melted = df_melted.rename(columns={date_col:'factor_date'})
    return df_melted

def optimize_df_memory(df):
    # Downcast numeric columns to reduce memory usage
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    # Convert object columns with few unique values to categorical
    for col in df.select_dtypes(include=['object']).columns:
        num_unique_values = df[col].nunique()
        num_total_values = len(df[col])
        if num_unique_values / num_total_values < 0.5:
            df[col] = df[col].astype('category')

def process_factor_df_in_chunks(conn, df, factor_names, date_col='date',
                               chunk_size=1000, staging_table='staging_daily_factors',
                               target_table='daily_factors',
                               conflict_cols=['ticker', 'factor_date', 'factor_name'],
                               update_cols=['factor_value']):

    tickers = df['ticker'].unique()
    for i in range(0, len(tickers), chunk_size):
        chunk_tickers = tickers[i:i + chunk_size]
        df_chunk = df[df['ticker'].isin(chunk_tickers)].copy()
        cols = ['ticker', date_col] + factor_names
        df_factor = df_chunk[cols].copy()

        # Your existing memory and dtype optimizations here
        # ...

        df_melted = df_factor.melt(id_vars=['ticker', date_col],
                                   var_name='factor_name', value_name='factor_value',
                                   ignore_index=True)

        if date_col != 'factor_date':
            df_melted = df_melted.rename(columns={date_col: 'factor_date'})

        # Add null ticker check and drop before bulk copy
        null_tickers = df_melted['ticker'].isnull().sum()
        if null_tickers > 0:
            print(f"Warning: Dropping {null_tickers} rows with NULL tickers before bulk copy.")
            df_melted = df_melted.dropna(subset=['ticker'])

        # Proceed with bulk copy and upsert
        bulk_copy_from_df(conn, df_melted, staging_table)
        upsert_from_staging(conn, staging_table, target_table, conflict_cols, update_cols)
        clear_staging_table(conn, staging_table)
        print(f"Processed chunk {i // chunk_size + 1} with {len(chunk_tickers)} tickers.")

        # Explicitly delete large DataFrames and collect garbage
        del df_chunk, df_factor, df_melted
        gc.collect()


def run_full_update(conn, engine, start_date, end_date):
    print("Starting full update...")
    df_daily = load_daily_prices(engine, start_date, end_date)
    df_indicators = compute_indicators(df_daily)

    factors_daily = [
        'adj_close','market_cap','dist52hi','dist52lo','vwap_10d','vwap_mtd',
        'sma_50d','sma_200d','ema_20d','ema_50d','ema_100d','ema_200d',
        'kama_10','kama_50','daily_turnover','atr_14d','illiquidity',
        'range_volatility','std_30d','std_3m','std_6m','std_12m'
    ]

    process_factor_df_in_chunks(
        conn,
        df_indicators,
        factors_daily,
        date_col='date',
        chunk_size=1000,
        staging_table='staging_daily_factors',
        target_table='daily_factors',
        conflict_cols=['ticker','factor_date','factor_name'],
        update_cols=['factor_value']
    )
    print("Daily factors upsert complete.")

    df_monthly = compute_monthly_factors(df_indicators)

    factors_monthly = [
        'avg_daily_turnover','avg_illiquidity','mom_12m','mom_6m','mom_3m',
        'pct_change_1m','pct_change_3m','pct_change_6m','pct_change_12m',
        'pct_change_24m','pct_change_60m'
    ]

    df_factors_monthly = prepare_factor_df(df_monthly, factors_monthly, date_col='factor_date')
    bulk_copy_from_df(conn, df_factors_monthly, 'staging_monthly_factors')
    upsert_from_staging(conn, 'staging_monthly_factors', 'monthly_factors',
                        conflict_cols=['ticker','factor_date','factor_name'],
                        update_cols=['factor_value'])
    clear_staging_table(conn, 'staging_monthly_factors')
    print("Monthly factors upsert complete.")
    print("Full update complete.")


def run_incremental_update(conn, engine, since_date):
    print("Starting incremental update...")
    df_daily = load_daily_prices(engine, since_date, date.today())
    df_indicators = compute_indicators(df_daily)

    factors_daily = [
        'adj_close','market_cap','dist52hi','dist52lo','vwap_10d','vwap_mtd',
        'sma_50d','sma_200d','ema_20d','ema_50d','ema_100d','ema_200d',
        'kama_10','kama_50','daily_turnover','atr_14d','illiquidity',
        'range_volatility','std_30d','std_3m','std_6m','std_12m'
    ]

    df_factors_daily = prepare_factor_df(df_indicators, factors_daily, date_col='date')

    bulk_copy_from_df(conn, df_factors_daily, 'staging_daily_factors')
    upsert_from_staging(conn, 'staging_daily_factors', 'daily_factors',
                        conflict_cols=['ticker','factor_date','factor_name'],
                        update_cols=['factor_value'])
    clear_staging_table(conn, 'staging_daily_factors')
    print("Daily factors incremental upsert complete.")

    df_monthly = compute_monthly_factors(df_indicators)

    factors_monthly = [
        'avg_daily_turnover','avg_illiquidity','mom_12m','mom_6m','mom_3m',
        'pct_change_1m','pct_change_3m','pct_change_6m','pct_change_12m',
        'pct_change_24m','pct_change_60m'
    ]

    df_factors_monthly = prepare_factor_df(df_monthly, factors_monthly, date_col='factor_date')

    bulk_copy_from_df(conn, df_factors_monthly, 'staging_monthly_factors')
    upsert_from_staging(conn, 'staging_monthly_factors', 'monthly_factors',
                        conflict_cols=['ticker','factor_date','factor_name'],
                        update_cols=['factor_value'])
    clear_staging_table(conn, 'staging_monthly_factors')
    print("Monthly factors incremental upsert complete.")
    print("Incremental update complete.")


def run_daily_incremental_update(conn, engine):
    update_day = datetime.today().date() - timedelta(days=30)
    lookback_days = 252  # longest rolling window or factors
    extended_start = update_day - timedelta(days=lookback_days)
    print(f"Running daily factors update for {update_day} with lookback from {extended_start}...")
    df_daily = load_daily_prices(engine, extended_start, update_day)
    df_indicators = compute_indicators(df_daily)

    factors_daily = [
        'adj_close','market_cap','dist52hi','dist52lo','vwap_10d','vwap_mtd',
        'sma_50d','sma_200d','ema_20d','ema_50d','ema_100d','ema_200d',
        'kama_10','kama_50','daily_turnover','atr_14d','illiquidity',
        'range_volatility','std_30d','std_3m','std_6m','std_12m'
    ]

    df_factors_daily = prepare_factor_df(df_indicators, factors_daily, date_col='date')

    bulk_copy_from_df(conn, df_factors_daily, 'staging_daily_factors')
    upsert_from_staging(conn, 'staging_daily_factors', 'daily_factors',
                        conflict_cols=['ticker','factor_date','factor_name'],
                        update_cols=['factor_value'])
    clear_staging_table(conn, 'staging_daily_factors')
    print("Daily factors incremental upsert complete.")


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(project_root, 'config.yml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    params = config['database']
    conn_str = f"postgresql+psycopg2://{params['user']}:{params['password']}@{params['host']}:{params['port']}/{params['dbname']}"
    engine = create_engine(conn_str)
    conn = psycopg2.connect(dbname=params['dbname'], user=params['user'], password=params['password'],
                            host=params['host'], port=params['port'])
    conn.set_client_encoding('UTF8')

    try:
        # mode = decide_mode()
        mode = 'full'
        if mode == 'full':
            start = date(2000, 1, 1)
            today = datetime.today().date()
            end = (today.replace(day=1) - timedelta(days=1))
            run_full_update(conn, engine, start, end)
        elif mode == 'incremental':
            since = datetime.today().date() - timedelta(days=180)
            run_incremental_update(conn, engine, since)

        run_daily_incremental_update(conn, engine)

    finally:
        conn.close()
        print("DB connection closed.")
