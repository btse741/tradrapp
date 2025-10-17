import os
import yaml
import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np
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

    if today.weekday() == 1:  # Tuesday
        for day in [monday, sunday, saturday]:
            if day.month != today.month:
                month_change_day = day
                break
    else:
        month_change_day = yesterday

    if (month_change_day.month == 1 or month_change_day.month == 7) and 1 <= month_change_day.day <= 7:
        return "full"

    return "incremental"

def get_monthly_eom_dates(start, end):
    dates = []
    current = start.replace(day=1)
    while current <= end:
        next_month = current + relativedelta(months=1)
        eom = next_month - timedelta(days=1)
        dates.append(eom)
        current = next_month
    return dates

def safe_momentum_calc(series, lag1, lag2):
    prev1 = series.shift(lag1)
    prev2 = series.shift(lag2)
    valid = (prev2 != 0) & prev2.notna() & prev1.notna()
    result = pd.Series(np.nan, index=series.index)
    valid_idxs = valid[valid].index
    result.loc[valid_idxs] = (prev1.loc[valid_idxs] / prev2.loc[valid_idxs] - 1) * 100
    return result


def build_monthly_price_factors(conn, engine, start_date, end_date):
    print(f"Building monthly price factors from {start_date} to {end_date}...")

    query = """
        SELECT ticker, date, adj_close, shares_outstanding
        FROM daily_prices
        WHERE date BETWEEN %s AND %s
        ORDER BY ticker, date
    """

    with engine.connect() as connection:
        df_daily = pd.read_sql_query(query, connection, params=(start_date, end_date))

    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily['ticker'] = df_daily['ticker'].astype('category')
    df_daily = df_daily.sort_values(['ticker', 'date'])

    # Calculate 52-week rolling high/low from daily data
    df_daily['high_52w'] = df_daily.groupby('ticker', observed=False)['adj_close'].transform(
        lambda x: x.rolling(252, min_periods=1).max())
    df_daily['low_52w'] = df_daily.groupby('ticker', observed=False)['adj_close'].transform(
        lambda x: x.rolling(252, min_periods=1).min())

    df_daily['dist52hi'] = np.where(
        (df_daily['high_52w'] > 0) & (df_daily['adj_close'] > 0),
        df_daily['high_52w'] / df_daily['adj_close'] - 1,
        np.nan
    )
    df_daily['dist52lo'] = np.where(
        (df_daily['low_52w'] > 0) & (df_daily['adj_close'] > 0),
        df_daily['adj_close'] / df_daily['low_52w'] - 1,
        np.nan
    )

    # Aggregate daily to monthly by last trading day
    df_daily['month_end'] = df_daily['date'].dt.to_period('M').apply(lambda r: r.end_time.date())
    df_monthly = df_daily.groupby(['ticker', 'month_end'], observed=False).last().reset_index()

    df_monthly['market_cap'] = df_monthly['adj_close'] * df_monthly['shares_outstanding']
    df_monthly.loc[df_monthly['adj_close'].isna() | df_monthly['shares_outstanding'].isna(), 'market_cap'] = np.nan

    df_monthly['mom_12m'] = safe_momentum_calc(df_monthly['adj_close'], 1, 12)
    df_monthly['mom_6m'] = safe_momentum_calc(df_monthly['adj_close'], 1, 7)
    df_monthly['mom_3m'] = safe_momentum_calc(df_monthly['adj_close'], 1, 4)

    lags = {
        'pct_change_1m': 1,
        'pct_change_3m': 3,
        'pct_change_6m': 6,
        'pct_change_12m': 12,
        'pct_change_24m': 24,
        'pct_change_60m': 60
    }

    def safe_pct_change(s, lag):
        prev = s.shift(lag)
        valid = (prev != 0) & prev.notna()
        result = pd.Series(np.nan, index=s.index)
        valid_idxs = valid[valid].index
        result.loc[valid_idxs] = ((s.loc[valid_idxs] - prev.loc[valid_idxs]) / prev.loc[valid_idxs]) * 100
        return result

    for name, lag in lags.items():
        df_monthly[name] = df_monthly.groupby('ticker', observed=False)['adj_close'].transform(lambda x: safe_pct_change(x, lag))

    # Merge monthly dist52hi and dist52lo from daily data monthly last values
    monthly_dist52hi = df_daily.groupby(['ticker', 'month_end'], observed=False)['dist52hi'].last().reset_index()
    monthly_dist52lo = df_daily.groupby(['ticker', 'month_end'], observed=False)['dist52lo'].last().reset_index()

    df_monthly = df_monthly.merge(monthly_dist52hi, on=['ticker', 'month_end'], how='left', validate='many_to_one')
    df_monthly = df_monthly.merge(monthly_dist52lo, on=['ticker', 'month_end'], how='left', validate='many_to_one')

    factors = [
        'adj_close', 'market_cap',
        'mom_12m', 'mom_6m', 'mom_3m'
    ] + list(lags.keys()) + ['dist52hi', 'dist52lo']

    existing_factors = [f for f in factors if f in df_monthly.columns]

    df_clean = df_monthly.dropna(subset=existing_factors + ['adj_close']).copy()
    df_clean['factor_date'] = df_clean['month_end']

    df_subset = df_clean[['ticker', 'factor_date'] + existing_factors]
    df_factors = df_subset.melt(
        id_vars=['ticker', 'factor_date'],
        value_vars=existing_factors,
        var_name='factor_name',
        value_name='factor_value'
    ).dropna(subset=['factor_value'])

    cols = ','.join(df_factors.columns)
    insert_query = f"""
        INSERT INTO monthly_factors ({cols})
        VALUES %s
        ON CONFLICT (ticker, factor_date, factor_name) DO UPDATE
        SET factor_value = EXCLUDED.factor_value;
    """

    batch_size = 2500
    num_rows = len(df_factors)
    print("Inserting price based factors to monthly_factors table...")

    with conn.cursor() as cur:
        for start in tqdm(range(0, num_rows, batch_size), desc="Batch inserting"):
            batch_df = df_factors.iloc[start:start + batch_size]
            batch_tuples = [tuple(x) for x in batch_df.to_numpy()]
            try:
                psycopg2.extras.execute_values(cur, insert_query, batch_tuples)
                conn.commit()
            except Exception as e:
                print(f"Error during batch insert at index {start}: {e}")
                conn.rollback()
                continue

    print("Completed building and upserting monthly price based factors.")


def run_full_rebuild(conn, engine, start_date, end_date):
    monthly_dates = get_monthly_eom_dates(start_date, end_date)
    print(f"Running full rebuild from {monthly_dates[0]} to {monthly_dates[-1]}")
    build_monthly_price_factors(conn, engine, start_date - timedelta(days=365 * 5), end_date)


def run_incremental_update(conn, engine, since_date):
    today = datetime.today().date()
    last_month_end = (today.replace(day=1) - timedelta(days=1))
    print(f"Running incremental update from {since_date} to {last_month_end}")
    build_monthly_price_factors(conn, engine, since_date - timedelta(days=365 * 5), last_month_end)


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(project_root, 'config.yml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    params = config['database']
    conn_str = f"postgresql+psycopg2://{params['user']}:{params['password']}@{params['host']}:{params['port']}/{params['dbname']}"
    engine = create_engine(conn_str)
    conn = psycopg2.connect(dbname=params['dbname'], user=params['user'], password=params['password'], host=params['host'], port=params['port'])
    conn.set_client_encoding('UTF8')

    try:
        mode = decide_mode()
        # mode = 'full'
        if mode == 'full':
            start = date(2000, 1, 1)
            today = datetime.today().date()
            end = (today.replace(day=1) - timedelta(days=1))
            run_full_rebuild(conn, engine, start, end)
        elif mode == 'incremental':
            since = datetime.today().date() - timedelta(days=180)
            run_incremental_update(conn, engine, since)
        else:
            print("Not scheduled to run today")
    finally:
        conn.close()
        print("DB connection closed.")
