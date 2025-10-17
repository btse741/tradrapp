import pandas as pd
from datetime import timedelta

def backtest_factor_model(conn, engine, start_date, end_date, factors_to_score, holding_period_days=21):
    monthly_dates = get_monthly_eom_dates(start_date, end_date)
    portfolio_returns = []

    for factor_date in monthly_dates:
        # Generate factor scores and quantile groups
        scores_df = generate_scores_and_groups(conn, engine, factor_date, factors_to_score)

        # Define holding period after factor_date
        holding_start = factor_date + timedelta(days=1)
        holding_end = factor_date + timedelta(days=holding_period_days)

        # Fetch daily returns over holding period for portfolio stocks
        returns_sql = """
            SELECT ticker, date, adj_return
            FROM daily_returns
            WHERE date BETWEEN %s AND %s AND ticker = ANY(%s)
        """
        with engine.connect() as connection:
            returns_df = pd.read_sql(returns_sql, connection, params=(holding_start, holding_end, scores_df.index.tolist()))

        # Aggregate returns per ticker (cumulative product of (1 + daily return))
        cum_returns = returns_df.groupby('ticker')['adj_return'].apply(lambda x: (1 + x).prod() - 1)

        # Join returns with scores to assign portfolio group returns
        merged = scores_df.merge(cum_returns, left_index=True, right_index=True, how='inner')

        # Average return by quantile group
        avg_returns = merged.groupby('quantile_group')['adj_return'].mean().reset_index()
        avg_returns['factor_date'] = factor_date

        portfolio_returns.append(avg_returns)

    # Combine all monthly results and compute cumulative returns by quantile
    results_df = pd.concat(portfolio_returns)
    results_df['cum_return'] = results_df.groupby('quantile_group')['adj_return'].cumsum()

    return results_df


def run_full_backtest(conn, engine, start_date, end_date, factors_to_score, holding_period_days=21):
    print(f"Running full backtest from {start_date} to {end_date}")
    return backtest_factor_model(conn, engine, start_date, end_date, factors_to_score, holding_period_days)


def run_incremental_backtest(conn, engine, since_date, holding_period_days=21):
    today = pd.Timestamp.today().date()
    last_month_end = (today.replace(day=1) - timedelta(days=1)).date()
    print(f"Running incremental backtest from {since_date} to {last_month_end}")
    return backtest_factor_model(conn, engine, since_date, last_month_end, factors_to_score, holding_period_days)


if __name__ == "__main__":
    # Your existing config and connection setup code here...

    factors_to_score = [
        'roic', 'ebit_yield', 'sales_growth', 'net_profit_growth', 'fcf_growth',
        'debt_to_equity', 'ttm_fcf', 'np_margin', 'shares_diluted_60m',
        'years_to_no_debt', 'dividend_yield', 'ttm_dvd_to_net_income'
    ]

    try:
        mode = 'full'  # or decide_mode()

        if mode == 'full':
            start_date = date(2000, 1, 1)
            today = datetime.today()
            end_date = (today.replace(day=1) - timedelta(days=1)).date()
            backtest_results = run_full_backtest(conn, engine, start_date, end_date, factors_to_score)

        elif mode == 'incremental':
            since_date = datetime.today().date() - timedelta(days=180)
            backtest_results = run_incremental_backtest(conn, engine, since_date)

        else:
            print("Not scheduled to run backtest today")
            backtest_results = None

    finally:
        conn.close()
        engine.dispose()
        print("DB connection closed.")

    if backtest_results is not None:
        print(backtest_results.head())
        # Further analysis or visualization here
