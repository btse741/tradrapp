import pandas as pd
import matplotlib.pyplot as plt
import yaml
import numpy as np
from sqlalchemy import create_engine
from datetime import timedelta, date,timedelta, datetime
from dateutil.relativedelta import relativedelta
from sqlalchemy import text


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
    for i in range(len(monthly_dates)-1):
        start = monthly_dates[i] + timedelta(days=1)  # day after factor date
        end = monthly_dates[i+1]  # up to next factor date
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
    # Ensure adj_close is float for pct_change
    df['adj_close'] = df['adj_close'].astype(float)

    # Compute returns safely - handle zero price changes and avoid divide by zero
    df['adj_return'] = df.groupby('ticker')['adj_close'].apply(
        lambda x: x.pct_change().replace([np.inf, -np.inf], np.nan))

    # Drop rows with NaN returns (including first row per ticker and zero division results)
    df = df.dropna(subset=['adj_return'])

    return df

def backtest_all_factors_monthly(engine, start_date, end_date):
    monthly_dates = get_monthly_eom_dates(start_date, end_date)
    holding_periods = get_monthly_rebalancing_periods(monthly_dates)

    all_results = []

    for (factor_date, (hold_start, hold_end)) in zip(monthly_dates[:-1], holding_periods):
        print(f"Backtesting for factor_date {factor_date} with holding {hold_start} to {hold_end}")

        scores_df = fetch_all_scores_for_date(engine, factor_date)
        if scores_df.empty:
            print(f"No factor scores for date {factor_date}")
            continue

        for factor_name in scores_df['strategy_name'].unique():
            print(f"  Processing factor: {factor_name}")
            factor_data = scores_df[scores_df['strategy_name'] == factor_name][['ticker', 'score']].set_index('ticker').dropna()
            try:
                factor_data['quantile'] = pd.qcut(factor_data['score'],
                                                  5,
                                                  labels=[1, 2, 3, 4, 5])
               
            except ValueError as e:
                print(f"Skipping factor {factor_name} on {factor_date} due to qcut error: {e}")
                continue

            if factor_data['quantile'].nunique() < 5:
                print(f"Skipping factor {factor_name} on {factor_date} due to insufficient quantile variation")
                continue

            top_tickers = factor_data[factor_data['quantile'] == 5].index.tolist()
            bottom_tickers = factor_data[factor_data['quantile'] == 1].index.tolist()
            tickers = top_tickers + bottom_tickers

            returns_df = fetch_and_compute_returns(engine, tickers, hold_start, hold_end)
            if returns_df.empty:
                print(f"No returns for holding {hold_start} to {hold_end}")
                continue

            cum_returns = returns_df.groupby('ticker')['adj_return']\
                .apply(lambda x: (1 + x.dropna()).prod() - 1)

            returns_with_quant = factor_data[['quantile']].join(cum_returns.rename('cum_return'), how='inner')

            top_return = returns_with_quant[returns_with_quant['quantile'] == 5]['cum_return'].mean()
            bottom_return = returns_with_quant[returns_with_quant['quantile'] == 1]['cum_return'].mean()
            long_short_return = top_return - bottom_return

            print(f"Top quantile return {top_return}, Bottom quantile return {bottom_return}, Long-Short return {long_short_return}")

            all_results.append({
                'factor_date': factor_date,
                'factor_name': factor_name,
                'top_return': top_return,
                'bottom_return': bottom_return,
                'long_short_return': long_short_return
            })
            

    results_df = pd.DataFrame(all_results)
    results_df['cum_long_short_return'] = results_df.groupby('factor_name')['long_short_return'].apply(lambda x: (1 + x.fillna(0)).cumprod() - 1)

    return results_df


def plot_factor_performance(results_df):
    plt.figure(figsize=(14, 7))
    for factor_name, group in results_df.groupby('factor_name'):
        plt.plot(group['factor_date'], group['cum_long_short_return'], label=factor_name)
    plt.title('Cumulative Long-Short Returns by Factor')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend(loc='upper left', bbox_to_anchor=(1,1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Example main
if __name__ == '__main__':
    
    with open('config.yml') as f:
        config = yaml.safe_load(f)
    db = config['database']
    conn_str = f"postgresql+psycopg2://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['dbname']}"
    engine = create_engine(conn_str)

    start_dt = date(2020, 1, 1)
    today = datetime.today()
    end_dt = (today.replace(day=1) - timedelta(days=1)).date()

    results = backtest_all_factors_monthly(engine, start_dt, end_dt)
    print(results.tail())
    results.to_csv('backtest_results.csv', index=False)

    plot_factor_performance(results)
