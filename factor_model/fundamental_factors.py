import os
import yaml
import psycopg2
import psycopg2.extras
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta

# Indicators list as you provided
indicators = [
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
    {
        'name': 'pretax_income_loss',
        'sector': 'nonfinancials',
        'type': 'flow',
        'income_table': 'income_qtr_nonfinancials',
        'factor_name': 'ttm_pbt'
    },
    {
        'name': 'pretax_income_loss',
        'sector': 'banks',
        'type': 'flow',
        'income_table': 'income_qtr_banks',
        'factor_name': 'ttm_pbt'
    },
    {
        'name': 'pretax_income_loss',
        'sector': 'insurance',
        'type': 'flow',
        'income_table': 'income_qtr_insurance',
        'factor_name': 'ttm_pbt'
    },
    {
        'name': 'income_tax_expense_benefit_net',
        'sector': 'nonfinancials',
        'type': 'flow',
        'income_table': 'income_qtr_nonfinancials',
        'factor_name': 'ttm_tax'
    },
    {
        'name': 'income_tax_expense_benefit_net',
        'sector': 'banks',
        'type': 'flow',
        'income_table': 'income_qtr_banks',
        'factor_name': 'ttm_tax'
    },
    {
        'name': 'income_tax_expense_benefit_net',
        'sector': 'insurance',
        'type': 'flow',
        'income_table': 'income_qtr_insurance',
        'factor_name': 'ttm_tax'
    },
    {
        'name': 'dividends_paid',
        'sector': 'nonfinancials',
        'type': 'flow',
        'income_table': 'cashflow_qtr_nonfinancials',
        'factor_name': 'ttm_dvd'
    },
    {
        'name': 'dividends_paid',
        'sector': 'banks',
        'type': 'flow',
        'income_table': 'cashflow_qtr_banks',
        'factor_name': 'ttm_dvd'
    },
    {
        'name': 'dividends_paid',
        'sector': 'insurance',
        'type': 'flow',
        'income_table': 'cashflow_qtr_insurance',
        'factor_name': 'ttm_dvd'
    },
    {
        'name': 'net_cash_from_operating_activities',
        'sector': 'nonfinancials',
        'type': 'flow',
        'income_table': 'cashflow_qtr_nonfinancials',
        'factor_name': 'ttm_cash_flows'
    },
    {
        'name': 'net_cash_from_investing_activities',
        'sector': 'nonfinancials',
        'type': 'flow',
        'income_table': 'cashflow_qtr_nonfinancials',
        'factor_name': 'ttm_capex'
    },
    {
        'name': 'revenue',
        'sector': 'nonfinancials',
        'type': 'flow',
        'income_table': 'income_qtr_nonfinancials',
        'factor_name': 'ttm_sales'
    },
    {
        'name': 'revenue',
        'sector': 'banks',
        'type': 'flow',
        'income_table': 'income_qtr_banks',
        'factor_name': 'ttm_sales'
    },
    {
        'name': 'revenue',
        'sector': 'insurance',
        'type': 'flow',
        'income_table': 'income_qtr_insurance',
        'factor_name': 'ttm_sales'
    },
    {
        'name': 'operating_income_loss',
        'sector': 'nonfinancials',
        'type': 'flow',
        'income_table': 'income_qtr_nonfinancials',
        'factor_name': 'ttm_ebit'
    },
    {
        'name': 'operating_income_loss',
        'sector': 'banks',
        'type': 'flow',
        'income_table': 'income_qtr_banks',
        'factor_name': 'ttm_ebit'
    },
    {
        'name': 'operating_income_loss',
        'sector': 'insurance',
        'type': 'flow',
        'income_table': 'income_qtr_insurance',
        'factor_name': 'ttm_ebit'
    },
    {
        'name': 'cost_of_revenue',
        'sector': 'nonfinancials',
        'type': 'flow',
        'income_table': 'income_qtr_nonfinancials',
        'factor_name': 'ttm_cos'
    },
    {
        'name': 'shares_diluted',
        'sector': 'nonfinancials',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_nonfinancials',
        'statistic': 'last',
        'factor_name': 'shares_diluted'
    },
    {
        'name': 'shares_diluted',
        'sector': 'banks',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_banks',
        'statistic': 'last',
        'factor_name': 'shares_diluted'
    },
    {
        'name': 'shares_diluted',
        'sector': 'insurance',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_insurance',
        'statistic': 'last',
        'factor_name': 'shares_diluted'
    },
    {
        'name': 'total_liabilities',
        'sector': 'nonfinancials',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_nonfinancials',
        'statistic': 'last',
        'factor_name': 'total_liabilities'
    },
    {
        'name': 'total_liabilities',
        'sector': 'banks',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_banks',
        'statistic': 'last',
        'factor_name': 'total_liabilities'
    },
    {
        'name': 'total_liabilities',
        'sector': 'insurance',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_insurance',
        'statistic': 'last',
        'factor_name': 'total_liabilities'
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
        'name': 'total_equity',
        'sector': 'banks',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_banks',
        'statistic': 'average',
        'factor_name': 'total_equity_avg'
    },
    {
        'name': 'total_equity',
        'sector': 'insurance',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_insurance',
        'statistic': 'average',
        'factor_name': 'total_equity_avg'
    },
    {
        'name': 'total_equity',
        'sector': 'nonfinancials',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_nonfinancials',
        'statistic': 'last',
        'factor_name': 'total_equity_last'
    },
    {
        'name': 'total_equity',
        'sector': 'banks',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_banks',
        'statistic': 'last',
        'factor_name': 'total_equity_last'
    },
    {
        'name': 'total_equity',
        'sector': 'insurance',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_insurance',
        'statistic': 'last',
        'factor_name': 'total_equity_last'
    },
    {
        'name': 'preferred_equity',
        'sector': 'banks',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_banks',
        'statistic': 'last',
        'factor_name': 'preferred_equity_last'
    },
    {
        'name': 'preferred_equity',
        'sector': 'insurance',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_insurance',
        'statistic': 'last',
        'factor_name': 'preferred_equity_last'
    },
    {
        'name': 'total_deposits',
        'sector': 'banks',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_banks',
        'statistic': 'last',
        'factor_name': 'total_deposits'
    },
    {
        'name': 'total_assets',
        'sector': 'nonfinancials',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_nonfinancials',
        'statistic': 'average',
        'factor_name': 'total_assets_avg'
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
        'name': 'total_assets',
        'sector': 'insurance',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_insurance',
        'statistic': 'average',
        'factor_name': 'total_assets_avg'
    },
    {
        'name': 'insurance_reserves',
        'sector': 'insurance',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_insurance',
        'statistic': 'last',
        'factor_name': 'insurance_reserves'
    },
    {
        'name': 'total_current_assets',
        'sector': 'nonfinancials',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_nonfinancials',
        'statistic': 'last',
        'factor_name': 'curr_assets'
    },
    {
        'name': 'total_current_liabilities',
        'sector': 'nonfinancials',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_nonfinancials',
        'statistic': 'last',
        'factor_name': 'curr_liab'
    },
    {
        'name': 'property_plant_equipment_net',
        'sector': 'nonfinancials',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_nonfinancials',
        'statistic': 'last',
        'factor_name': 'net_fixed_assets'
    },
    {
        'name': 'cash_cash_equivalents_short_term_investments',
        'sector': 'nonfinancials',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_nonfinancials',
        'statistic': 'last',
        'factor_name': 'cash_and_equiv'
    },
    {
        'name': 'cash_cash_equivalents_short_term_investments',
        'sector': 'banks',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_banks',
        'statistic': 'last',
        'factor_name': 'cash_and_equiv'
    },
    {
        'name': 'cash_cash_equivalents_short_term_investments',
        'sector': 'insurance',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_insurance',
        'statistic': 'last',
        'factor_name': 'cash_and_equiv'
    },
    {
        'name': 'long_term_debt',
        'sector': 'nonfinancials',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_nonfinancials',
        'statistic': 'last',
        'factor_name': 'long_term_debt'
    },
    {
        'name': 'short_term_debt',
        'sector': 'nonfinancials',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_nonfinancials',
        'statistic': 'last',
        'factor_name': 'short_term_debt'
    },
    {
        'name': 'long_term_debt',
        'sector': 'banks',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_banks',
        'statistic': 'last',
        'factor_name': 'long_term_debt'
    },
    {
        'name': 'short_term_debt',
        'sector': 'banks',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_banks',
        'statistic': 'last',
        'factor_name': 'short_term_debt'
    },
    {
        'name': 'long_term_debt',
        'sector': 'insurance',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_insurance',
        'statistic': 'last',
        'factor_name': 'long_term_debt'
    },
    {
        'name': 'short_term_debt',
        'sector': 'insurance',
        'type': 'stock',
        'balance_sheet_table': 'balancesheets_qtr_insurance',
        'statistic': 'last',
        'factor_name': 'short_term_debt'
    }
]

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

def safe_div(a, b):
    if b in (None, 0) or a in (None, 0):
        return None
    return a / b

def safe_mult(a, b):
    if a in (None, 0) or b in (None, 0):
        return None
    return a * b

def get_monthly_eom_dates(start, end):
    dates = []
    current = start.replace(day=1)
    while current <= end:
        nxt = current + relativedelta(months=1)
        eom = nxt - timedelta(days=1)
        dates.append(eom)
        current = nxt
    return dates

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

def compute_and_upsert_extended_factors(conn, factor_date, engine, indicators):
    sql_fetch = """
        SELECT ticker,
               MAX(CASE WHEN factor_name = 'adj_close' THEN factor_value END) AS adj_close,
               MAX(CASE WHEN factor_name = 'shares_diluted' THEN factor_value END) AS shares_diluted,
               MAX(CASE WHEN factor_name = 'curr_assets' THEN factor_value END) AS curr_assets,
               MAX(CASE WHEN factor_name = 'curr_liab' THEN factor_value END) AS curr_liab,
               MAX(CASE WHEN factor_name = 'net_fixed_assets' THEN factor_value END) AS net_fixed_assets,
               MAX(CASE WHEN factor_name = 'cash_and_equiv' THEN factor_value END) AS cash_and_equiv,
               MAX(CASE WHEN factor_name = 'total_equity_last' THEN factor_value END) AS total_equity_last,
               MAX(CASE WHEN factor_name = 'total_equity_avg' THEN factor_value END) AS total_equity_avg,
               MAX(CASE WHEN factor_name = 'total_assets_avg' THEN factor_value END) AS total_assets_avg,
               MAX(CASE WHEN factor_name = 'preferred_equity_last' THEN factor_value END) AS preferred_equity_last,
               MAX(CASE WHEN factor_name = 'long_term_debt' THEN factor_value END) AS long_term_debt,
               MAX(CASE WHEN factor_name = 'short_term_debt' THEN factor_value END) AS short_term_debt,
               MAX(CASE WHEN factor_name = 'ttm_net_income' THEN factor_value END) AS ttm_net_income,
               MAX(CASE WHEN factor_name = 'ttm_tax' THEN factor_value END) AS ttm_tax,
               MAX(CASE WHEN factor_name = 'ttm_pbt' THEN factor_value END) AS ttm_pbt,
               MAX(CASE WHEN factor_name = 'ttm_cash_flows' THEN factor_value END) AS ttm_cash_flows,
               MAX(CASE WHEN factor_name = 'ttm_capex' THEN factor_value END) AS ttm_capex,
               MAX(CASE WHEN factor_name = 'ttm_ebit' THEN factor_value END) AS ttm_ebit,
               MAX(CASE WHEN factor_name = 'ttm_sales' THEN factor_value END) AS ttm_sales,
               MAX(CASE WHEN factor_name = 'ttm_cos' THEN factor_value END) AS ttm_cos,
               MAX(CASE WHEN factor_name = 'ttm_div' THEN factor_value END) AS ttm_div
        FROM monthly_factors
        WHERE factor_date = %s
        GROUP BY ticker
    """

    sql_market_cap = """
        SELECT ticker, factor_value AS market_cap
        FROM monthly_factors
        WHERE factor_date = %s AND factor_name = 'market_cap'
    """

    with conn.cursor() as cur:
        cur.execute(sql_fetch, (factor_date,))
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]

        cur.execute(sql_market_cap, (factor_date,))
        market_cap_rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=columns)
    market_cap_dict = {r[0]: r[1] for r in market_cap_rows}

    extended_rows = []

    for _, row in df.iterrows():
        ticker = row['ticker']
        market_cap = market_cap_dict.get(ticker)

        working_capital = None
        if row['curr_assets'] is not None and row['curr_liab'] is not None:
            working_capital = row['curr_assets'] - row['curr_liab']

        ttm_eps = None
        if row['ttm_net_income'] is not None and row['shares_diluted'] not in (None, 0):
            ttm_eps = safe_div(row['ttm_net_income'], row['shares_diluted'])

        post_tax = None
        if row['ttm_tax'] is not None and row['ttm_pbt'] not in (None, 0):
            post_tax = 1 - (row['ttm_tax'] / row['ttm_pbt'])

        ttm_fcf = None
        if row['ttm_cash_flows'] is not None and row['ttm_capex'] is not None:
            ttm_fcf = row['ttm_cash_flows'] - row['ttm_capex']

        invested_capital_nf = None
        if None not in (working_capital, row['cash_and_equiv'], row['net_fixed_assets']):
            invested_capital_nf = working_capital - row['cash_and_equiv'] + row['net_fixed_assets']

        invested_capital_f = None
        if None not in (row['total_equity_last'], row['preferred_equity_last']):
            invested_capital_f = row['total_equity_last'] - row['preferred_equity_last']

        invested_capital = invested_capital_nf or invested_capital_f

        enterprise_value = None
        if market_cap is not None:
            debt_long = row['long_term_debt'] or 0
            debt_short = row['short_term_debt'] or 0
            cash = row['cash_and_equiv'] or 0
            enterprise_value = market_cap + debt_long + debt_short - cash

        ebit_yield = None
        if row['ttm_ebit'] is not None and row['ttm_ebit'] > 0 and enterprise_value not in (None, 0):
            ebit_yield = safe_div(row['ttm_ebit'], enterprise_value)

        pe = None
        if market_cap is not None and row['ttm_net_income'] is not None and row['ttm_net_income'] > 0:
            pe = safe_div(market_cap, row['ttm_net_income'])

        pb = None
        if market_cap is not None and row['total_equity_last'] not in (None, 0) and row['total_equity_last'] > 0:
            pb = safe_div(market_cap, row['total_equity_last'])

        roe = None
        if row['ttm_net_income'] is not None and row['total_equity_avg'] not in (None, 0):
            roe = safe_div(row['ttm_net_income'], row['total_equity_avg'])

        roa = None
        if row['ttm_net_income'] is not None and row['total_assets_avg'] not in (None, 0):
            roa = safe_div(row['ttm_net_income'], row['total_assets_avg'])

        div_yield = None
        if row['ttm_div'] is not None and market_cap not in (None, 0):
            div_yield = safe_div(row['ttm_div'], market_cap)

        debt_to_equity = None
        long_term = row['long_term_debt'] or 0
        short_term = row['short_term_debt'] or 0
        total_equity_last = row['total_equity_last']
        if total_equity_last not in (None, 0):
            debt_to_equity = safe_div(long_term + short_term, total_equity_last)

        roic = None
        if row['ttm_ebit'] is not None and row['ttm_ebit'] > 0 and post_tax is not None and invested_capital not in (None, 0):
            roic = safe_div(safe_mult(row['ttm_ebit'], post_tax), invested_capital)

        fcf_yield = None
        if ttm_fcf is not None and market_cap not in (None, 0):
            fcf_yield = safe_div(ttm_fcf, market_cap)

        gp_margin = None
        if row['ttm_sales'] not in (None, 0) and row['ttm_cos'] is not None:
            gp_margin = safe_div(row['ttm_sales'] - row['ttm_cos'], row['ttm_sales'])

        np_margin = None
        if row['ttm_net_income'] is not None and row['ttm_sales'] not in (None, 0):
            np_margin = safe_div(row['ttm_net_income'], row['ttm_sales'])

        yrs_to_cash = None
        if row['long_term_debt'] is not None and row['short_term_debt'] is not None and row['ttm_cash_flows'] not in (None, 0) and row['ttm_capex'] not in (None, 0 ): 
            yrs_to_cash = safe_div(row['long_term_debt'] + row['short_term_debt'] , (row['ttm_cash_flows']-row['ttm_capex']))

        div_cover = None
        div_cover = safe_div(row['ttm_div'], row['ttm_net_income'])

        factors = {
            'working_capital': working_capital,
            'ttm_eps': ttm_eps,
            'post_tax': post_tax,
            'ttm_fcf': ttm_fcf,
            'invested_capital': invested_capital,
            'enterprise_value': enterprise_value,
            'ebit_yield': ebit_yield,
            'pe': pe,
            'pb': pb,
            'roe': roe,
            'roa': roa,
            'div_yield': div_yield,
            'div_cover': div_cover,
            'debt_to_equity': debt_to_equity,
            'roic': roic,
            'fcf_yield': fcf_yield,
            'gp_margin': gp_margin,
            'np_margin': np_margin,
            'yrs_to_cash': yrs_to_cash
        }

        for factor_name, factor_value in factors.items():
            extended_rows.append((ticker, factor_date, factor_name, factor_value))

    insert_sql = """
        INSERT INTO monthly_factors (ticker, factor_date, factor_name, factor_value)
        VALUES %s
        ON CONFLICT (ticker, factor_date, factor_name) DO UPDATE
        SET factor_value = EXCLUDED.factor_value;
    """

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, insert_sql, extended_rows)
        conn.commit()


def compute_and_upsert_growth_factors(conn, factor_date, engine, lookbacks=[12, 60]):
    factors_to_grow = ['ttm_sales', 'ttm_fcf', 'ttm_net_income', 'ttm_eps', 'ttm_div', 'shares_diluted']
    start_date = factor_date - relativedelta(months=max(lookbacks) * 2)

    sql = """
        SELECT ticker, factor_date, factor_name, factor_value
        FROM monthly_factors
        WHERE factor_name = ANY(%s)
          AND factor_date BETWEEN %s AND %s
    """

    with engine.connect() as connection:
        df = pd.read_sql(sql, connection, params=(factors_to_grow, start_date, factor_date))
    df['factor_date'] = pd.to_datetime(df['factor_date'])

    results = []

    for factor in factors_to_grow:
        df_factor = df[df['factor_name'] == factor].copy()
        df_factor.sort_values(by=['ticker', 'factor_date'], inplace=True)

        for lookback in lookbacks:
            # Calculate lagged factor_value by lookback months per ticker
            df_factor[f'lag_{lookback}'] = df_factor.groupby('ticker')['factor_value'].shift(lookback)

            # Filter rows for current factor_date to compute growth
            current_rows = df_factor[df_factor['factor_date'] == pd.Timestamp(factor_date)]

            for _, row in current_rows.iterrows():
                current_val = row['factor_value']
                lagged_val = row[f'lag_{lookback}']

                if pd.notna(current_val) and pd.notna(lagged_val) and current_val > 0 and lagged_val > 0:
                    growth_val = (current_val - lagged_val) / lagged_val
                    results.append((row['ticker'], factor_date, f"{factor}_growth_{lookback}m", growth_val))

    if not results:
        print(f"No growth factors computed for {factor_date}")
        return

    insert_sql = """
        INSERT INTO monthly_factors (ticker, factor_date, factor_name, factor_value)
        VALUES %s
        ON CONFLICT (ticker, factor_date, factor_name) DO UPDATE
        SET factor_value = EXCLUDED.factor_value
    """

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, insert_sql, results)
        conn.commit()


def count_valid_factors_per_date(conn, factor_date):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT factor_name, COUNT(factor_value) AS valid_count
            FROM monthly_factors
            WHERE factor_date = %s AND factor_value IS NOT NULL
            GROUP BY factor_name
            ORDER BY valid_count DESC
        """, (factor_date,))
        rows = cur.fetchall()
        print(f"Valid factor counts for {factor_date}:")
        for factor_name, count in rows:
            print(f"  {factor_name}: {count}")

def run_full_rebuild(conn, engine):
    start_date = date(2000,1,1)
    today = datetime.today()
    end_date = (today.replace(day=1) - timedelta(days=1)).date()
    dates = get_monthly_eom_dates(start_date, end_date)
    for factor_date in dates:
        print(f"Processing fundamental factors for {factor_date}")
        for indicator in indicators:
            build_factor_generic(conn, factor_date, indicator)
        compute_and_upsert_extended_factors(conn, factor_date, engine, indicators)
        compute_and_upsert_growth_factors(conn, factor_date, engine)
        count_valid_factors_per_date(conn, factor_date)
        conn.commit()

def run_incremental_update(conn, engine, since_date):
    today = datetime.today()
    last_month_end = (today.replace(day=1) - timedelta(days=1)).date()
    dates = get_monthly_eom_dates(since_date, last_month_end)
    for factor_date in dates:
        print(f"Processing fundamental factors for {factor_date}")
        for indicator in indicators:
            build_factor_generic(conn, factor_date, indicator)
        compute_and_upsert_extended_factors(conn, factor_date, engine, indicators)
        compute_and_upsert_growth_factors(conn, factor_date, engine)
        count_valid_factors_per_date(conn, factor_date)
        conn.commit()

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
            run_full_rebuild(conn, engine)
        elif mode == 'incremental':
            since_date = datetime.today().date() - timedelta(days=180)
            run_incremental_update(conn, engine, since_date)
        else:
            print("Not scheduled to run today")
    finally:
        conn.close()
        print("DB connection closed.")
