import pandas as pd
import numpy as np
import yaml
import os
import logging
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')  # removes timestamps and info prefixes from logs

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
    if (month_change_day.month == 1 or month_change_day.month == 7) and \
           month_change_day.weekday() == 4 and 1 <= month_change_day.day <= 7:
        return "full"
    return "incremental"


def get_monthly_eom_dates(start, end):
    months = []
    current = start.replace(day=1)
    while current <= end:
        next_month = current + relativedelta(months=1)
        eom = next_month - timedelta(days=1)
        months.append(eom)
        current = next_month
    return months


def get_monthly_rebalancing_periods(monthly_dates):
    holding_periods = []
    for i in range(len(monthly_dates) - 1):
        start = monthly_dates[i] + timedelta(days=1)  # day after factor date
        end = monthly_dates[i + 1]
        holding_periods.append((start, end))
    return holding_periods


def fetch_all_scores_for_date(engine, factor_date):
    sql = text("""
        SELECT ticker, strategy_name, score
        FROM factor_ranks
        WHERE factor_date = :factor_date
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"factor_date": factor_date})
    if not df.empty:
        df['ticker'] = df['ticker'].astype(str)
    return df


def fetch_and_compute_returns(engine, tickers, start_date, end_date):
    sql = text("""
        SELECT ticker, date, adj_close
        FROM daily_prices
        WHERE date BETWEEN :start_date AND :end_date
          AND ticker = ANY(:tickers)
        ORDER BY ticker, date ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn,
                        params={"start_date": start_date,
                                "end_date": end_date,
                                "tickers": tickers})
    if df.empty:
        return pd.DataFrame(columns=['ticker', 'date', 'adj_close', 'log_return'])
    
    df['ticker'] = df['ticker'].astype(str)
    df['adj_close'] = df['adj_close'].astype(float)
    
    # Filter out zero or negative prices before log calculation
    df = df[df['adj_close'] > 0].copy()
    
    # Compute log return safely after filtering
    df['log_return'] = df.groupby('ticker')['adj_close'].apply(lambda x: np.log(x).diff()).reset_index(level=0, drop=True)
    
    df = df.dropna(subset=['log_return'])
    return df


def backtest_for_single_factor_date(conn_str, factor_date, quantiles=5):
    engine = create_engine(conn_str, poolclass=NullPool)

    try:
        logging.info(f"Backtesting for factor_date {factor_date}")

        scores_df = fetch_all_scores_for_date(engine, factor_date)
        if scores_df.empty:
            logging.warning(f"No factor scores for date {factor_date}")
            return []

        results = []
        for factor_name in scores_df['strategy_name'].unique():

            factor_data = scores_df[scores_df['strategy_name'] == factor_name][['ticker', 'score']].dropna()
            factor_data = factor_data.reset_index(drop=True)  # Fix: reset index before assign

            factor_data['ranked_score'] = factor_data['score'].rank(method='first').fillna(np.nan)

            # Proceed only if ranked_score has enough unique values
            if factor_data['ranked_score'].isna().all():
                logging.warning(f"All ranked scores are NaN for factor {factor_name} at {factor_date}, skipping")
                continue

            try:
                quantiles_series = pd.qcut(factor_data['ranked_score'], quantiles, labels=range(1, quantiles + 1), duplicates='drop')
            except ValueError as e:
                logging.warning(f"pd.qcut failed for factor {factor_name} at {factor_date} with error: {e}")
                continue

            factor_data['quantile'] = quantiles_series.reset_index(drop=True)

            if factor_data['quantile'].nunique() < quantiles:
                logging.warning(f"Skipping factor {factor_name} {factor_date} due to insufficient quantile variation")
                continue


            tickers = factor_data['ticker'].tolist()
            if not tickers:
                continue

            calendar_start = (factor_date + pd.DateOffset(months=1)).replace(day=1)
            calendar_end = (calendar_start + pd.DateOffset(months=1)) - pd.Timedelta(days=1)

            returns_df = fetch_and_compute_returns(engine, tickers, calendar_start, calendar_end)
            if returns_df.empty:
                logging.warning(f"No returns found for {factor_name} in period {calendar_start} to {calendar_end}")
                continue

            hold_start = returns_df['date'].min()
            hold_end = returns_df['date'].max()
            if hold_start > hold_end:
                logging.warning(f"Invalid holding period for factor_date {factor_date}, skipping")
                continue

            cum_log_returns = returns_df.groupby('ticker')['log_return'].sum()
            cum_returns = np.exp(cum_log_returns) - 1

            # Fix: reset index before merging
            factor_data_reset = factor_data[['ticker', 'quantile']].reset_index(drop=True)
            cum_returns_reset = cum_returns.rename('cum_return').reset_index()

            merged = pd.merge(factor_data_reset, cum_returns_reset, on='ticker', how='inner')
            if merged.empty:
                logging.warning(f"No overlapping tickers for factor {factor_name} at {factor_date}")
                continue

            top_return = merged.loc[merged['quantile'] == quantiles, 'cum_return'].mean()
            bottom_return = merged.loc[merged['quantile'] == 1, 'cum_return'].mean()
            long_short_return = top_return - bottom_return

            results.append({
                'factor_date': factor_date,
                'factor_name': factor_name,
                'hold_start': hold_start,
                'hold_end': hold_end,
                'top_return': top_return,
                'bottom_return': bottom_return,
                'long_short_return': long_short_return
            })

        results_df = pd.DataFrame(results)
        if not results_df.empty:
            # Persist this factor_date’s results immediately
            csv_path = 'backtest_results_progress.csv'
            file_exists = os.path.isfile(csv_path)
            results_df.to_csv(
                csv_path,
                mode='a',
                index=False,
                header=not file_exists  # write header only once
            )

        print(f"Summary returns for factor_date {factor_date}:")
        print(results_df)
        return results

    finally:
        engine.dispose()


def backtest_all_factors_monthly_parallel(conn_str, start_date, end_date, max_workers=6):
    monthly_dates = get_monthly_eom_dates(start_date, end_date)

    results_all = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(backtest_for_single_factor_date, conn_str, dt): dt for dt in monthly_dates}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Backtesting Months"):
            factor_date = futures[future]
            try:
                res = future.result()
                if res:
                    results_all.extend(res)
            except Exception as e:
                logging.error(f"Backtest failed for {factor_date}: {e}", exc_info=True)

    results_df = pd.DataFrame(results_all)
    if not results_df.empty:
        results_df = results_df.sort_values(['factor_name', 'factor_date'])
        results_df['cum_long_short_return'] = (
            results_df
            .groupby('factor_name')['long_short_return']
            .transform(lambda x: (1 + x.fillna(0)).cumprod() - 1)
        )
    logging.info(f"Completed backtesting from {start_date} to {end_date}")
    return results_df


def plot_factor_performance(results_df):
    plt.figure(figsize=(14, 7))
    for factor_name, group in results_df.groupby('factor_name'):
        plt.plot(group['factor_date'], group['cum_long_short_return'], label=factor_name)
    plt.title('Cumulative Long-Short Returns by Factor')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    logging.info("Starting factor backtest script")
    with open('config.yml') as f:
        config = yaml.safe_load(f)
    db = config['database']
    conn_str = f"postgresql+psycopg2://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['dbname']}"

    mode = decide_mode()
    # For testing, you can force modes here
    # mode = 'full'
    today = date.today()
    if mode == 'full':
        start_dt = date(2005, 1, 1)
        end_dt = today - timedelta(days=1)
        logging.info("Starting full backtest mode")
    elif mode == 'incremental':
        start_dt = today - timedelta(days=180)
        end_dt = today - timedelta(days=1)
        logging.info("Starting incremental backtest mode")
    else:
        logging.info("No scheduled backtest today")
        exit()

    results = backtest_all_factors_monthly_parallel(conn_str, start_dt, end_dt)
    logging.info("Backtest finished")
    if not results.empty:
        print(results.tail())
        results.to_csv('backtest_results.csv', index=False)
        plot_factor_performance(results)
    else:
        logging.info("No results to plot")
