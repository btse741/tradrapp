import os
import yaml
import psycopg2
import psycopg2.extras
import pandas as pd
from sqlalchemy import create_engine, text
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

import os
import yaml
import psycopg2
import psycopg2.extras
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import text
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


def fill_missing_ranks(df, cols, ascending_cols=None):
    """Utility to fill NaNs with penalizing values for ranks"""
    ascending_cols = ascending_cols or []
    df_filled = df.copy()
    for col in cols:
        if col not in df_filled.columns:
            raise KeyError(f"Column '{col}' missing in DataFrame")
        if col in ascending_cols:
            df_filled[col] = df_filled[col].fillna(float('inf'))
        else:
            df_filled[col] = df_filled[col].fillna(float('-inf'))
    return df_filled


def magic_formula_ranks(df):
    required_cols = ['roic', 'ebit_yield']
    if any(col not in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Missing columns for magic formula: {missing}")
        return pd.DataFrame(index=df.index)
    ranks = pd.DataFrame(index=df.index)
    df_filled = fill_missing_ranks(df, required_cols)
    ranks['roic_rank'] = df_filled['roic'].rank(ascending=False, method='average')
    ranks['ebit_yield_rank'] = df_filled['ebit_yield'].rank(ascending=False, method='average')
    ranks['magic_formula_score'] = ranks['roic_rank'] + ranks['ebit_yield_rank']
    return ranks[['magic_formula_score']]


def sales_growth_score(df):
    col = 'ttm_sales_60m'
    if col not in df.columns:
        print(f"Missing column for sales growth: {col}")
        return pd.DataFrame(index=df.index)
    ranks = pd.DataFrame(index=df.index)
    df_filled = df.copy()
    df_filled[col] = df_filled[col].fillna(float('-inf'))
    ranks['sales_growth_rank'] = df_filled[col].rank(ascending=False, method='average')
    return ranks[['sales_growth_rank']]


def pe_score(df):
    col = 'pe'
    if col not in df.columns:
        print(f"Missing column for PE score: {col}")
        return pd.DataFrame(index=df.index)
    ranks = pd.DataFrame(index=df.index)
    df_filled = df.copy()
    df_filled[col] = df_filled[col].fillna(float('inf'))
    ranks['pe_rank'] = df_filled[col].rank(ascending=True, method='average')
    return ranks[['pe_rank']]


def composite_growth_score(df):
    required_cols = ['ttm_sales_60m', 'ttm_net_income_60m', 'ttm_fcf_60m']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing columns for composite growth: {missing_cols}")
        return pd.DataFrame(index=df.index)
    df_filled = fill_missing_ranks(df, required_cols)
    ranks = pd.DataFrame(index=df.index)
    ranks['sales_rank'] = df_filled['ttm_sales_60m'].rank(ascending=False, method='average')
    ranks['np_rank'] = df_filled['ttm_net_income_60m'].rank(ascending=False, method='average')
    ranks['fcf_rank'] = df_filled['ttm_fcf_60m'].rank(ascending=False, method='average')
    ranks['composite_growth_score'] = ranks.sum(axis=1)
    return ranks[['composite_growth_score']]


def composite_strength_score(df):
    required_cols = ['roic', 'np_margin', 'shares_diluted_60m', 'yrs_to_cash']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing columns for composite strength: {missing_cols}")
        return pd.DataFrame(index=df.index)
    df_filled = fill_missing_ranks(df, required_cols, ascending_cols=['shares_diluted_60m', 'yrs_to_cash'])
    ranks = pd.DataFrame(index=df.index)
    ranks['roic_rank'] = df_filled['roic'].rank(ascending=False, method='average')
    ranks['np_margin_rank'] = df_filled['np_margin'].rank(ascending=False, method='average')
    ranks['shares_diluted_60m_rank'] = df_filled['shares_diluted_60m'].rank(ascending=True, method='average')
    ranks['composite_strength_score'] = ranks.sum(axis=1)
    return ranks[['composite_strength_score']]


def composite_value_score(df):
    required_cols = ['ebit_yield', 'div_yield', 'pe', 'pb']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing columns for composite value: {missing_cols}")
        return pd.DataFrame(index=df.index)
    df_filled = fill_missing_ranks(df, required_cols, ascending_cols=['pe', 'pb'])
    ranks = pd.DataFrame(index=df.index)
    ranks['ebit_yield_rank'] = df_filled['ebit_yield'].rank(ascending=False, method='average')
    ranks['div_yield_rank'] = df_filled['div_yield'].rank(ascending=False, method='average')
    ranks['pe_rank'] = df_filled['pe'].rank(ascending=True, method='average')
    ranks['pb_rank'] = df_filled['pb'].rank(ascending=True, method='average')
    ranks['composite_value_score'] = ranks.sum(axis=1)
    return ranks[['composite_value_score']]


def composite_momentum_score(df):
    required_cols = ['mom_12m', 'mom_6m', 'pct_change_60m']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing columns for composite momentum: {missing_cols}")
        return pd.DataFrame(index=df.index)
    df_filled = fill_missing_ranks(df, required_cols)
    ranks = pd.DataFrame(index=df.index)
    ranks['mom12_rank'] = df_filled['mom_12m'].rank(ascending=False, method='average')
    ranks['mom6_rank'] = df_filled['mom_6m'].rank(ascending=False, method='average')
    ranks['mom5yr_rank'] = df_filled['pct_change_60m'].rank(ascending=True, method='average')
    ranks['composite_mom_score'] = ranks.sum(axis=1)
    return ranks[['composite_mom_score']]


# Screening functions unchanged


def generate_scores_and_groups(conn, engine, factor_date):
    factors_needed = [
        'roic', 'ebit_yield', 'roe', 'pe', 'pb', 'ttm_eps_12m', 'ttm_eps_60m',
        'ttm_sales_12m', 'ttm_sales_60m', 'ttm_net_income_12m', 'ttm_net_income_60m',
        'ttm_fcf', 'ttm_fcf_12m', 'ttm_fcf_60m', 'shares_diluted_12m', 'shares_diluted_60m',
        'debt_to_equity', 'yrs_to_cash', 'market_cap', 'enterprise_value',
        'np_margin', 'gp_margin', 'div_yield', 'div_cover',
        'mom_12m', 'mom_6m', 'pct_change_60m', 'dist52hi', 'dist52lo'
    ]

    sql = text("""
        SELECT ticker, factor_name, factor_value
        FROM monthly_factors
        WHERE factor_date = :factor_date AND factor_name = ANY(:factors)
        AND factor_value IS NOT NULL
    """)

    with engine.connect() as connection:
        df = pd.read_sql(sql, connection, params={"factor_date": factor_date, "factors": factors_needed})

    df_pivot = df.pivot(index='ticker', columns='factor_name', values='factor_value')

    mf = magic_formula_ranks(df_pivot)
    pe = pe_score(screen_top_market_cap(debt_to_equity_screen(df_pivot)))
    mom = composite_momentum_score(dist52hi_screen(screen_top_market_cap(debt_to_equity_screen(df_pivot))))
    cg = composite_growth_score(df_pivot)
    cs = composite_strength_score(df_pivot)
    cv = composite_value_score(df_pivot)
    cm = composite_momentum_score(df_pivot)

    combo = cg.join([cs, cv, cm], how='outer')
    combo['fused'] = combo.sum(axis=1, min_count=1)

    all_scores = mf.join([pe, mom, combo], how='outer')

    print(f"Generated all factor scores for {factor_date}: {len(all_scores)} tickers after ranking")
    return all_scores


def upsert_dataframe_to_postgres(engine, df, table_name):
    df_reset = df.reset_index()
    rank_cols = [col for col in df_reset.columns if col not in ['ticker', 'factor_date']]
    melted = df_reset.melt(id_vars=['ticker', 'factor_date'], value_vars=rank_cols,
                          var_name='strategy_name', value_name='score')
    melted['strategy_name'] = melted['strategy_name'].str.replace('_score', '', regex=False).str.replace('_rank', '', regex=False)

    conn = engine.raw_connection()
    cursor = conn.cursor()
    upsert_sql = f"""
    INSERT INTO {table_name} (ticker, factor_date, strategy_name, score)
    VALUES (%(ticker)s, %(factor_date)s, %(strategy_name)s, %(score)s)
    ON CONFLICT (ticker, factor_date, strategy_name) DO UPDATE SET
        score = EXCLUDED.score;
    """
    for _, row in melted.iterrows():
        params = {
            'ticker': row['ticker'],
            'factor_date': row['factor_date'],
            'strategy_name': row['strategy_name'],
            'score': row['score']
        }
        cursor.execute(upsert_sql, params)
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Upsert completed for {len(melted)} rows into {table_name}")


def run_full_rebuild(conn, engine, start_date, end_date):
    monthly_dates = get_monthly_eom_dates(start_date, end_date)
    print(f"Running full rebuild from {monthly_dates[0]} to {monthly_dates[-1]}")
    for factor_date in monthly_dates:
        all_scores = generate_scores_and_groups(conn, engine, factor_date)
        all_scores['factor_date'] = factor_date
        upsert_dataframe_to_postgres(engine, all_scores, 'factor_scores')


def run_incremental_update(conn, engine, since_date):
    today = datetime.today()
    last_month_end = (today.replace(day=1) - timedelta(days=1)).date()
    print(f"Running incremental update from {since_date} to {last_month_end}")
    monthly_dates = get_monthly_eom_dates(since_date, last_month_end)
    for factor_date in monthly_dates:
        all_scores = generate_scores_and_groups(conn, engine, factor_date)
        all_scores['factor_date'] = factor_date
        upsert_dataframe_to_postgres(engine, all_scores, 'factor_scores')


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
            start_date = date(2005, 1, 1)
            today = datetime.today()
            end_date = (today.replace(day=1) - timedelta(days=1)).date()
            run_full_rebuild(conn, engine, start_date, end_date)
        elif mode == 'incremental':
            since_date = datetime.today().date() - timedelta(days=180)
            run_incremental_update(conn, engine, since_date)
        else:
            print("Not scheduled to run today")
    finally:
        conn.close()
        engine.dispose()
        print("DB connection closed.")


# Modular factor model functions
def magic_formula_ranks(df):
    required_cols = ['roic', 'ebit_yield']
    if any(col not in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Missing columns for magic formula: {missing}")
        return pd.DataFrame(index=df.index)
    ranks = pd.DataFrame(index=df.index)
    df_filled = fill_missing_ranks(df, required_cols)
    ranks['roic_rank'] = df_filled['roic'].rank(ascending=False, method='average')
    ranks['ebit_yield_rank'] = df_filled['ebit_yield'].rank(ascending=False, method='average')
    ranks['magic_formula_score'] = ranks['roic_rank'] + ranks['ebit_yield_rank']
    return ranks[['magic_formula_score']]


def sales_growth_score(df):
    col = 'ttm_sales_60m'
    if col not in df.columns:
        print(f"Missing column for sales growth: {col}")
        return pd.DataFrame(index=df.index)
    ranks = pd.DataFrame(index=df.index)
    df_filled = df.copy()
    df_filled[col] = df_filled[col].fillna(float('-inf'))
    ranks['sales_growth_rank'] = df_filled[col].rank(ascending=False, method='average')
    return ranks[['sales_growth_rank']]


def pe_score(df):
    col = 'pe'
    if col not in df.columns:
        print(f"Missing column for PE score: {col}")
        return pd.DataFrame(index=df.index)
    ranks = pd.DataFrame(index=df.index)
    df_filled = df.copy()
    df_filled[col] = df_filled[col].fillna(float('inf'))
    ranks['pe_rank'] = df_filled[col].rank(ascending=True, method='average')
    return ranks[['pe_rank']]


def composite_growth_score(df):
    required_cols = ['ttm_sales_60m', 'ttm_net_income_60m', 'ttm_fcf_60m']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing columns for composite growth: {missing_cols}")
        return pd.DataFrame(index=df.index)
    df_filled = fill_missing_ranks(df, required_cols)
    ranks = pd.DataFrame(index=df.index)
    ranks['sales_rank'] = df_filled['ttm_sales_60m'].rank(ascending=False, method='average')
    ranks['np_rank'] = df_filled['ttm_net_income_60m'].rank(ascending=False, method='average')
    ranks['fcf_rank'] = df_filled['ttm_fcf_60m'].rank(ascending=False, method='average')
    ranks['composite_growth_score'] = ranks.sum(axis=1)
    return ranks[['composite_growth_score']]


def composite_strength_score(df):
    required_cols = ['roic', 'np_margin', 'shares_diluted_60m', 'yrs_to_cash']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing columns for composite strength: {missing_cols}")
        return pd.DataFrame(index=df.index)
    df_filled = fill_missing_ranks(df, required_cols, ascending_cols=['shares_diluted_60m', 'yrs_to_cash'])
    ranks = pd.DataFrame(index=df.index)
    ranks['roic_rank'] = df_filled['roic'].rank(ascending=False, method='average')
    ranks['np_margin_rank'] = df_filled['np_margin'].rank(ascending=False, method='average')
    ranks['shares_diluted_60m_rank'] = df_filled['shares_diluted_60m'].rank(ascending=True, method='average')
    ranks['composite_strength_score'] = ranks.sum(axis=1)
    return ranks[['composite_strength_score']]


def composite_value_score(df):
    required_cols = ['ebit_yield', 'div_yield', 'pe', 'pb']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing columns for composite value: {missing_cols}")
        return pd.DataFrame(index=df.index)
    df_filled = fill_missing_ranks(df, required_cols, ascending_cols=['pe', 'pb'])
    ranks = pd.DataFrame(index=df.index)
    ranks['ebit_yield_rank'] = df_filled['ebit_yield'].rank(ascending=False, method='average')
    ranks['div_yield_rank'] = df_filled['div_yield'].rank(ascending=False, method='average')
    ranks['pe_rank'] = df_filled['pe'].rank(ascending=True, method='average')
    ranks['pb_rank'] = df_filled['pb'].rank(ascending=True, method='average')
    ranks['composite_value_score'] = ranks.sum(axis=1)
    return ranks[['composite_value_score']]


def composite_momentum_score(df):
    required_cols = ['mom_12m', 'mom_6m', 'pct_change_60m']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing columns for composite momentum: {missing_cols}")
        return pd.DataFrame(index=df.index)
    df_filled = fill_missing_ranks(df, required_cols)
    ranks = pd.DataFrame(index=df.index)
    ranks['mom12_rank'] = df_filled['mom_12m'].rank(ascending=False, method='average')
    ranks['mom6_rank'] = df_filled['mom_6m'].rank(ascending=False, method='average')
    ranks['mom5yr_rank'] = df_filled['pct_change_60m'].rank(ascending=True, method='average')
    ranks['composite_mom_score'] = ranks.sum(axis=1)
    return ranks[['composite_mom_score']]


# Screening functions

def debt_to_equity_screen(df, max_debt=1.0):
    screened = df[df['debt_to_equity'] < max_debt]
    print(f"Screened {len(screened)} stocks with debt_to_equity < {max_debt}")
    return screened


def dist52hi_screen(df, pct=0.2):
    screened = df[df['dist52hi'] < pct]
    print(f"Screened {len(screened)} stocks with dist52hi < {pct}")
    return screened


def growth_screen(df, threshold=0.5):
    filtered = df[
        (df['ttm_sales_60m'] >= threshold) &
        (df['ttm_fcf_60m'] > threshold) &
        (df['ttm_eps_60m'] > threshold)
    ]
    print(f"Screened {len(filtered)} stocks meeting profitability thresholds")
    return filtered


def strength_screen(df, threshold=0.1):
    filtered = df[
        (df['debt_to_equity'] <= 1) &
        (df['roic'] >= threshold) &
        (df['np_margin'] > threshold) &
        (df['yrs_to_cash'] < 2) &
        (df['shares_diluted_60m'] < 0)
    ]
    print(f"Screened {len(filtered)} stocks meeting strength thresholds")
    return filtered


def screen_top_market_cap(df, top_n=100):
    if 'market_cap' not in df.columns:
        raise ValueError("market_cap column is required for this screening")
    top_stocks = df.nlargest(top_n, 'market_cap')
    print(f"Selected top {top_n} stocks by market cap")
    return top_stocks


def generate_scores_and_groups(conn, engine, factor_date):
    factors_needed = [
        'roic', 'ebit_yield',
        'roe', 'pe', 'pb',
        'ttm_eps_growth_12m', 'ttm_eps_growth_60m',
        'ttm_sales_growth_12m', 'ttm_sales_growth_60m',
        'ttm_net_income_growth_12m', 'ttm_net_income_growth_60m',
        'ttm_fcf', 'ttm_fcf_growth_12m', 'ttm_fcf_growth_60m',
        'shares_diluted_growth_12m', 'shares_diluted_growth_60m',
        'debt_to_equity', 'yrs_to_cash',
        'market_cap', 'enterprise_value',
        'np_margin', 'gp_margin',
        'div_yield', 'div_cover',
        'mom_12m', 'mom_6m', 'pct_change_60m',
        'dist52hi', 'dist52lo'
    ]

    sql = """
        SELECT ticker, factor_name, factor_value
        FROM monthly_factors
        WHERE factor_date = %s AND factor_name = ANY(%s) AND factor_value IS NOT NULL
    """

    with engine.connect() as connection:
        df = pd.read_sql(sql, connection, params=(factor_date, factors_needed))

    df_pivot = df.pivot(index='ticker', columns='factor_name', values='factor_value')

    mf = magic_formula_ranks(df_pivot)
    pe = pe_score(screen_top_market_cap(debt_to_equity_screen(df_pivot)))
    mom = composite_momentum_score(dist52hi_screen(screen_top_market_cap(debt_to_equity_screen(df_pivot))))
    cg = composite_growth_score(df_pivot)
    cs = composite_strength_score(df_pivot)
    cv = composite_value_score(df_pivot)
    cm = composite_momentum_score(df_pivot)

    combo = cg.join([cs, cv, cm], how='outer')
    combo['fused'] = combo.sum(axis=1, min_count=1)

    all_scores = mf.join([pe, mom, combo], how='outer')

    print(f"Generated all factor scores for {factor_date}: {len(all_scores)} tickers after ranking")
    return all_scores


def upsert_dataframe_to_postgres(engine, df, table_name):
    # Convert wide DataFrame with multiple strategy columns into long format with strategy_name and score
    df_reset = df.reset_index()  # bring ticker from index to column

    # Melt rank columns into strategy_name and score
    rank_cols = [col for col in df_reset.columns if col not in ['ticker', 'factor_date']]
    melted = df_reset.melt(id_vars=['ticker', 'factor_date'], value_vars=rank_cols,
                          var_name='strategy_name', value_name='score')

    # Clean strategy_name strings if needed
    melted['strategy_name'] = melted['strategy_name'].str.replace('_score', '', regex=False).str.replace('_rank', '', regex=False)

    conn = engine.raw_connection()
    cursor = conn.cursor()

    upsert_sql = f"""
    INSERT INTO {table_name} (ticker, factor_date, strategy_name, score)
    VALUES (%(ticker)s, %(factor_date)s, %(strategy_name)s, %(score)s)
    ON CONFLICT (ticker, factor_date, strategy_name) DO UPDATE SET
        score = EXCLUDED.score;
    """

    for _, row in melted.iterrows():
        params = {
            'ticker': row['ticker'],
            'factor_date': row['factor_date'],
            'strategy_name': row['strategy_name'],
            'score': row['score']
        }
        cursor.execute(upsert_sql, params)

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Upsert completed for {len(melted)} rows into {table_name}")


def run_full_rebuild(conn, engine, start_date, end_date):
    monthly_dates = get_monthly_eom_dates(start_date, end_date)
    print(f"Running full rebuild from {monthly_dates[0]} to {monthly_dates[-1]}")
    for factor_date in monthly_dates:
        all_scores = generate_scores_and_groups(conn, engine, factor_date)
        all_scores['factor_date'] = factor_date
        upsert_dataframe_to_postgres(engine, all_scores, 'factor_ranks')


def run_incremental_update(conn, engine, since_date):
    today = datetime.today()
    last_month_end = (today.replace(day=1) - timedelta(days=1)).date()
    print(f"Running incremental update from {since_date} to {last_month_end}")
    monthly_dates = get_monthly_eom_dates(since_date, last_month_end)
    for factor_date in monthly_dates:
        all_scores = generate_scores_and_groups(conn, engine, factor_date)
        all_scores['factor_date'] = factor_date
        upsert_dataframe_to_postgres(engine, all_scores, 'factor_ranks')



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

    
    # sql = text("""SELECT DISTINCT factor_name FROM monthly_factors ORDER BY factor_name""")
    # with engine.connect() as connection:
    #     result = connection.execute(sql)
    #     factor_names = [row[0] for row in result]
    # print(factor_names)


    try:
        # mode = decide_mode()
        mode = 'full'
        if mode == 'full':
            start_date = date(2005, 1, 1)
            today = datetime.today()
            end_date = (today.replace(day=1) - timedelta(days=1)).date()
            run_full_rebuild(conn, engine, start_date, end_date)
        elif mode == 'incremental':
            since_date = datetime.today().date() - timedelta(days=180)
            run_incremental_update(conn, engine, since_date)
        else:
            print("Not scheduled to run today")
    finally:
        conn.close()
        engine.dispose()
        print("DB connection closed.")
