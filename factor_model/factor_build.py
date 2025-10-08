import os
import yaml
import psycopg2
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta

# def is_yesterday_month_end(today=None):
#     if today is None:
#         today = date.today()
#     yesterday = today - timedelta(days=1)
#     tomorrow_of_yesterday = yesterday + timedelta(days=1)
#     return tomorrow_of_yesterday.month != yesterday.month

# if not is_yesterday_month_end():
#     print("Yesterday was not month-end. Skipping factor calculation.")
#     exit(0)

# Load config
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
config_path = os.path.join(project_root, 'config.yml')

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

params = config['database']

def get_monthly_eom_dates(start, end):
    dates = []
    current = start.replace(day=1)
    while current <= end:
        next_month = current + relativedelta(months=1)
        eom = next_month - timedelta(days=1)
        dates.append(eom)
        current = next_month
    return dates

# Define atomic indicators metadata
indicators = [
    # Flow data indicators (computed as sum over last 4 quarters)
    {
        'name': 'net_income',
        'sector': 'nonfinancials',
        'type': 'flow',
        'income_table': 'income_qtr_nonfinancials',
        'factor_name': 'ttm_net_income'
    },
    {
        'name': 'net_income',
        'sector': 'banks',
        'type': 'flow',
        'income_table': 'income_qtr_banks',
        'factor_name': 'ttm_net_income'
    },
    {
        'name': 'net_income',
        'sector': 'insurance',
        'type': 'flow',
        'income_table': 'income_qtr_insurance',
        'factor_name': 'ttm_net_income'
    },

    # Stock data indicators (last or average)
    {
        'name': 'total_liabilities',
        'sector': 'nonfinancials',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_nonfinancials',
        'statistic': 'last',
        'factor_name': 'total_liabilities_last'
    },
    {
        'name': 'total_equity',
        'sector': 'nonfinancials',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_nonfinancials',
        'statistic': 'average',
        'factor_name': 'total_equity_avg'
    },
    {
        'name': 'total_deposits',
        'sector': 'banks',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_banks',
        'statistic': 'last',
        'factor_name': 'total_deposits_last'
    },
    {
        'name': 'total_assets',
        'sector': 'banks',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_banks',
        'statistic': 'average',
        'factor_name': 'total_assets_avg'
    },
    {
        'name': 'insurance_reserves',
        'sector': 'insurance',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_insurance',
        'statistic': 'last',
        'factor_name': 'insurance_reserves_last'
    },
    {
        'name': 'total_assets',
        'sector': 'insurance',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_insurance',
        'statistic': 'average',
        'factor_name': 'total_assets_avg_insurance'
    },
]

def build_flow_factor(conn, factor_date, indicator):
    sql = f"""
        WITH ttm_value AS (
            SELECT ticker, SUM({indicator['name']}) AS value
            FROM (
                SELECT i.ticker, i.{indicator['name']},
                       ROW_NUMBER() OVER (PARTITION BY i.ticker ORDER BY i.publish_date DESC) AS rn
                FROM {indicator['income_table']} i
                WHERE i.publish_date <= %s
            ) sub
            WHERE rn <= 4
            GROUP BY ticker
            HAVING COUNT(*) = 4
        )
        INSERT INTO monthly_factors(ticker, factor_date, factor_name, factor_value)
        SELECT ticker, %s, %s, value FROM ttm_value
        ON CONFLICT (ticker, factor_date, factor_name) DO UPDATE SET factor_value = EXCLUDED.factor_value;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (factor_date, factor_date, indicator['factor_name']))

def build_stock_factor(conn, factor_date, indicator):
    col = indicator['name']
    table = indicator['balance_sheet_table']
    factor_name = indicator['factor_name']

    if indicator['statistic'] == 'last':
        sql = f"""
            WITH latest_val AS (
                SELECT DISTINCT ON (ticker) ticker, {col} AS value
                FROM {table}
                WHERE publish_date <= %s
                ORDER BY ticker, publish_date DESC
            )
            INSERT INTO monthly_factors(ticker, factor_date, factor_name, factor_value)
            SELECT ticker, %s, %s, value FROM latest_val
            ON CONFLICT (ticker, factor_date, factor_name) DO UPDATE SET factor_value = EXCLUDED.factor_value;
        """
        params = (factor_date, factor_date, factor_name)

    elif indicator['statistic'] == 'average':
        sql = f"""
            WITH ranked_vals AS (
                SELECT ticker, {col}, ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY publish_date DESC) AS rn
                FROM {table}
                WHERE publish_date <= %s
            ), avg_vals AS (
                SELECT ticker, AVG({col}) AS avg_value
                FROM ranked_vals
                WHERE rn <= 4
                GROUP BY ticker
                HAVING COUNT(*) = 4
            )
            INSERT INTO monthly_factors(ticker, factor_date, factor_name, factor_value)
            SELECT ticker, %s, %s, avg_value FROM avg_vals
            ON CONFLICT (ticker, factor_date, factor_name) DO UPDATE SET factor_value = EXCLUDED.factor_value;
        """
        params = (factor_date, factor_date, factor_name)

    else:
        raise ValueError(f"Unknown statistic type for stock indicator: {indicator['statistic']}")

    with conn.cursor() as cur:
        cur.execute(sql, params)

def build_factor_generic(conn, factor_date, indicator):
    if indicator['type'] == 'flow':
        build_flow_factor(conn, factor_date, indicator)
    elif indicator['type'] == 'stock':
        build_stock_factor(conn, factor_date, indicator)
    else:
        raise ValueError(f"Unknown indicator type: {indicator['type']}")

def main():
    monthly_dates = get_monthly_eom_dates(datetime(2010, 1, 1), datetime.today() - relativedelta(months=1))
    print(f"Building monthly factors from {monthly_dates[0].date()} to {monthly_dates[-1].date()}")

    conn = psycopg2.connect(
        dbname=params['dbname'],
        user=params['user'],
        password=params['password'],
        host=params['host'],
        port=params['port']
    )
    conn.set_client_encoding('UTF8')

    try:
        for factor_date in monthly_dates:
            print(f"Processing factors for {factor_date.strftime('%Y-%m-%d')}")
            for indicator in indicators:
                build_factor_generic(conn, factor_date, indicator)
    finally:
        conn.close()
        print("Monthly factor population completed.")

if __name__ == "__main__":
    main()
