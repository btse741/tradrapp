import os
import gc
import yaml
import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    dates = []
    current = start.replace(day=1)
    while current <= end:
        next_month = current + relativedelta(months=1)
        eom = next_month - timedelta(days=1)
        dates.append(eom)
        current = next_month
    return dates


def adaptive_winsorize_series(s):
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    winsorized = s.clip(lower, upper)
    return winsorized


def standardize_series(s):
    # Using sample std deviation ddof=1 for unbiased estimator
    return (s - s.mean()) / s.std(ddof=1)


def impute_missing(df):
    # Simple median imputation
    return df.fillna(df.median())


def preprocess_factors(df, factor_cols):
    processed = pd.DataFrame(index=df.index)
    for col in factor_cols:
        if col in df.columns:
            clean_series = df[col].dropna()
            if clean_series.empty:
                processed[col] = np.nan
                continue
            wins = adaptive_winsorize_series(clean_series)
            std = standardize_series(wins)
            # Map standardized values back to original index to preserve alignment
            std_full = pd.Series(np.nan, index=df.index)
            std_full.loc[std.index] = std
            processed[col] = std_full
        else:
            processed[col] = np.nan
    processed = impute_missing(processed)
    return processed


def rank_series(s, ascending=True):
    return s.rank(method='average', ascending=ascending)


def weighted_composite_score(df, weights):
    present_cols = [col for col in weights.keys() if col in df.columns]
    weighted_sum = sum(df[col] * weights[col] for col in present_cols)
    return weighted_sum


def generate_scores_only(conn, engine, factor_date, min_unique=5):
    # List all factors used in composites:
    all_factors = [
        'roic', 'ebit_yield', 'pe', 'pb',
        'ttm_sales_growth_60m', 'ttm_net_income_growth_60m', 'ttm_fcf_growth_60m',
        'np_margin', 'shares_diluted_growth_60m', 'div_yield', 'yrs_to_cash',
        'mom_12m', 'mom_6m', 'pct_change_60m'
    ]
    sql = text("""
        SELECT ticker, factor_name, factor_value
        FROM monthly_factors
        WHERE factor_date = :factor_date AND factor_name = ANY(:factors)
        AND factor_value IS NOT NULL
    """)
    with engine.connect() as connection:
        df_raw = pd.read_sql(sql, connection, params={"factor_date": factor_date, "factors": all_factors})

    df_pivot = df_raw.pivot(index='ticker', columns='factor_name', values='factor_value')

    processed = preprocess_factors(df_pivot, all_factors)

    # Composite Scores Definitions with weighted sum
    # Weights from your original, possibly dynamic in future
    composite_weights = {
        'composite_growth_score': {
            'ttm_sales_growth_60m': 0.33,
            'ttm_net_income_growth_60m': 0.33,
            'ttm_fcf_growth_60m': 0.34
        },
        'composite_strength_score': {
            'roic': 0.34,
            'np_margin': 0.33,
            'shares_diluted_growth_60m': 0.17,
            'yrs_to_cash': 0.16
        },
        'composite_value_score': {
            'ebit_yield': 0.25,
            'div_yield': 0.25,
            'pe': 0.25,
            'pb': 0.25
        },
        'composite_momentum_score': {
            'mom_12m': 0.4,
            'mom_6m': 0.3,
            'pct_change_60m': 0.3
        },
        'fama_french_5_factor_score': {
            'np_margin': 0.6,
            'shares_diluted_growth_60m': 0.4
        }
    }

    scores = {}

    # Magic Formula (roic + ebit_yield)
    mf_cols = ['roic', 'ebit_yield']
    mf_valid = processed[mf_cols].dropna()
    mf_composite = mf_valid.sum(axis=1)
    scores['magic_formula_score'] = rank_series(mf_composite, ascending=False)

    # PE Score (ascending is better so rank ascending)
    if 'pe' in processed.columns:
        scores['pe_rank'] = rank_series(processed['pe'], ascending=True)
    else:
        scores['pe_rank'] = pd.Series(np.nan, index=processed.index)

    # Composite Scores
    def compute_composite_score(cols, weights, ascending=True):
        subset = processed[cols]
        weighted = weighted_composite_score(subset, weights)
        return rank_series(weighted, ascending=ascending)

    scores['composite_growth_score'] = compute_composite_score(
        composite_weights['composite_growth_score'].keys(),
        composite_weights['composite_growth_score']
    )

    scores['composite_strength_score'] = compute_composite_score(
        composite_weights['composite_strength_score'].keys(),
        composite_weights['composite_strength_score']
    )

    scores['composite_value_score'] = compute_composite_score(
        composite_weights['composite_value_score'].keys(),
        composite_weights['composite_value_score']
    )

    scores['composite_momentum_score'] = compute_composite_score(
        composite_weights['composite_momentum_score'].keys(),
        composite_weights['composite_momentum_score']
    )

    scores['fama_french_5_factor_score'] = compute_composite_score(
        composite_weights['fama_french_5_factor_score'].keys(),
        composite_weights['fama_french_5_factor_score']
    )

    combined = pd.DataFrame(scores)

    # Check uniqueness and log for diagnostics
    unique_counts = combined.nunique(dropna=True)
    logging.info(f"Unique counts for factor_date {factor_date}:\n{unique_counts}")

    low_unique_cols = unique_counts[unique_counts < min_unique].index.tolist()
    if low_unique_cols:
        for col in low_unique_cols:
            logging.warning(f"Low unique count {unique_counts[col]} in column {col}. Value counts:\n{combined[col].value_counts(dropna=False).head(20)}")

    combined['weighted_composite_score'] = (
        0.25 * combined['composite_growth_score'] +
        0.25 * combined['composite_strength_score'] +
        0.25 * combined['composite_value_score'] +
        0.15 * combined['composite_momentum_score'] +
        0.10 * combined['fama_french_5_factor_score']
    )
    combined['factor_date'] = factor_date

    logging.info(f"Generated factor scores for {factor_date} with {len(combined)} tickers")

    return combined


def upsert_dataframe_to_postgres(engine, df, table_name):
    if 'factor_date' not in df.columns:
        if 'factor_date' in df.index.names:
            df = df.reset_index()
        else:
            raise KeyError("The DataFrame must contain 'factor_date'")
    df = df.reset_index(drop=False)
    melted = df.melt(id_vars=['ticker', 'factor_date'], var_name='strategy_name', value_name='score')
    melted['strategy_name'] = melted['strategy_name'].str.replace('_score', '', regex=False).str.replace('_rank', '', regex=False)
    params_list = melted.to_dict('records')
    conn = engine.raw_connection()
    try:
        cursor = conn.cursor()
        upsert_sql = f"""
            INSERT INTO {table_name} (ticker, factor_date, strategy_name, score)
            VALUES (%(ticker)s, %(factor_date)s, %(strategy_name)s, %(score)s)
            ON CONFLICT (ticker, factor_date, strategy_name) DO UPDATE SET
                score = EXCLUDED.score;
        """
        cursor.executemany(upsert_sql, params_list)
        conn.commit()
        cursor.close()
    finally:
        conn.close()
    logging.info(f"Upsert completed for {len(melted)} rows into {table_name}.")


def process_factor_date(args):
    factor_date, conn_info = args
    engine = create_engine(conn_info['conn_str'])
    conn = psycopg2.connect(
        dbname=conn_info['dbname'],
        user=conn_info['user'],
        password=conn_info['password'],
        host=conn_info['host'],
        port=conn_info['port']
    )
    conn.set_client_encoding('UTF8')
    try:
        all_scores = generate_scores_only(conn, engine, factor_date)
        upsert_dataframe_to_postgres(engine, all_scores, 'factor_ranks')
        gc.collect()
        return (factor_date, "Success", len(all_scores))
    except Exception as e:
        return (factor_date, "Error", str(e))
    finally:
        conn.close()
        engine.dispose()


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(project_root, 'config.yml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    params = config['database']
    conn_info = {
        'conn_str': f"postgresql+psycopg2://{params['user']}:{params['password']}@{params['host']}:{params['port']}/{params['dbname']}",
        'dbname': params['dbname'],
        'user': params['user'],
        'password': params['password'],
        'host': params['host'],
        'port': params['port']
    }
    # mode = decide_mode()
    mode = 'full'
    if mode == 'full':
        start_date = date(2005, 1, 1)
        today = datetime.today()
        end_date = (today.replace(day=1) - timedelta(days=1)).date()
        logging.info(f"Starting full rebuild from {start_date} to {end_date}")
        monthly_dates = get_monthly_eom_dates(start_date, end_date)
        args = [(fd, conn_info) for fd in monthly_dates]
        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(process_factor_date, arg): arg[0] for arg in args}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Months"):
                factor_date = futures[future]
                try:
                    res = future.result()
                    logging.info(f"Completed {factor_date}: {res[1]} ({res[2]} rows)")
                except Exception as e:
                    logging.error(f"Failed {factor_date}: {e}")
        logging.info("Full rebuild completed.")
    elif mode == 'incremental':
        since_date = datetime.today().date() - timedelta(days=180)
        today = datetime.today()
        last_month_end = (today.replace(day=1) - timedelta(days=1)).date()
        logging.info(f"Starting incremental update from {since_date} to {last_month_end}")
        monthly_dates = get_monthly_eom_dates(since_date, last_month_end)
        args = [(fd, conn_info) for fd in monthly_dates]
        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(process_factor_date, arg): arg[0] for arg in args}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Months"):
                factor_date = futures[future]
                try:
                    res = future.result()
                    logging.info(f"Completed incremental {factor_date}: {res[1]} ({res[2]} rows)")
                except Exception as e:
                    logging.error(f"Failed incremental {factor_date}: {e}")
        logging.info("Incremental update completed.")
    else:
        logging.info("No scheduled run today.")
