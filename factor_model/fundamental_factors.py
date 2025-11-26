import os
import yaml
import psycopg2
import psycopg2.extras
from psycopg2 import pool
import io
from sqlalchemy import create_engine
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import concurrent.futures
from tqdm import tqdm
import numpy as np
import csv
from functools import partial
import multiprocessing as mp 

# --- Global Variables and Constants

connection_pool = None

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
        'factor_name': 'ttm_div'
    },
    {
        'name': 'dividends_paid',
        'sector': 'banks',
        'type': 'flow',
        'income_table': 'cashflow_qtr_banks',
        'factor_name': 'ttm_div'
    },
    {
        'name': 'dividends_paid',
        'sector': 'insurance',
        'type': 'flow',
        'income_table': 'cashflow_qtr_insurance',
        'factor_name': 'ttm_div'
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
    if (month_change_day.month == 1 or month_change_day.month == 7) and month_change_day.weekday() == 4 and 1 <= month_change_day.day <= 7:
        return "full"
    return "incremental"

def init_connection_pool(conn_params):
    global connection_pool
    if connection_pool is None:
        connection_pool = pool.ThreadedConnectionPool(
            minconn=2, maxconn=10,
            dbname=conn_params['dbname'],
            user=conn_params['user'],
            password=conn_params['password'],
            host=conn_params['host'],
            port=conn_params['port']
        )

def get_conn_from_pool():
    return connection_pool.getconn()

def put_conn_back(conn):
    connection_pool.putconn(conn)

def close_connection_pool():
    global connection_pool
    if connection_pool:
        try:
            connection_pool.closeall()
        except psycopg2.pool.PoolError:
            # Pool already closed, ignore
            pass
        connection_pool = None

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


def process_indicator_for_date(conn_params, factor_date, indicator):
    conn = get_conn_from_pool()
    try:
        build_factor_generic(conn, factor_date, indicator)
        conn.commit()
    finally:
        put_conn_back(conn)


def compute_and_upsert_extended_factors(conn, factor_date, extended_start):
    print(f" Computing extended factors for {factor_date.strftime('%Y-%m-%d')}...")

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
        WHERE factor_date = %s AND factor_date >= %s
        GROUP BY ticker
    """

    sql_market_cap = """
        SELECT ticker, factor_value AS market_cap
        FROM monthly_factors
        WHERE factor_date = %s AND factor_name = 'market_cap'
    """

    with conn.cursor() as cur:
        cur.execute(sql_fetch, (factor_date, extended_start))
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
        if row['total_equity_last'] is not None:
            preferred = row['preferred_equity_last'] or 0
            invested_capital_f = row['total_equity_last'] - preferred

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

        long_term = row['long_term_debt'] or 0
        short_term = row['short_term_debt'] or 0
        total_equity_last = row['total_equity_last'] or 0
        cash = row['cash_and_equiv'] or 0
        if total_equity_last not in (None, 0):
            debt_to_equity = safe_div(long_term + short_term - cash, total_equity_last)
        else:
            debt_to_equity = None

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
        if row['long_term_debt'] is not None and row['short_term_debt'] is not None and row['ttm_cash_flows'] not in (None, 0) and row['ttm_capex'] not in (None, 0) and row['cash_and_equiv'] not in (None, 0):
            yrs_to_cash = safe_div(row['long_term_debt'] + row['short_term_debt'] - row['cash_and_equiv'], (row['ttm_cash_flows'] - row['ttm_capex']))

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

    print(f"Upserting {len(extended_rows)} extended factor records for date {factor_date}...")

    insert_sql = """
        INSERT INTO monthly_factors (ticker, factor_date, factor_name, factor_value)
        VALUES %s
        ON CONFLICT (ticker, factor_date, factor_name) DO UPDATE
        SET factor_value = EXCLUDED.factor_value;
    """

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, insert_sql, extended_rows)
        conn.commit()


def compute_and_upsert_extended_factors_batch(conn_params, engine_params, indicators, factor_dates, extended_start):
    engine = create_engine(
        f"postgresql+psycopg2://{engine_params['user']}:{engine_params['password']}@"
        f"{engine_params['host']}:{engine_params['port']}/{engine_params['dbname']}"
    )
    conn = psycopg2.connect(**conn_params)
    try:
        for factor_date in factor_dates:
            for indicator in indicators:
                build_factor_generic(conn, factor_date, indicator)
            compute_and_upsert_extended_factors(conn, factor_date, extended_start)
            conn.commit()
    finally:
        conn.close()


def compute_and_upsert_growth_factors(conn, engine, factor_date, max_lookback_months=84):
    print(f"Computing growth factors for {factor_date.strftime('%Y-%m-%d')}...")
    factors_to_grow = ['ttm_sales', 'ttm_fcf', 'ttm_net_income', 'ttm_eps', 'ttm_div', 'shares_diluted']

    start_date = factor_date - relativedelta(months=max_lookback_months)

    sql_fetch = """
        SELECT ticker, factor_date, factor_name, factor_value
        FROM monthly_factors
        WHERE factor_name = ANY(%s)
          AND factor_date BETWEEN %s AND %s
    """
    df = pd.read_sql(sql_fetch, con=engine, params=(factors_to_grow, start_date, factor_date))
    df['factor_date'] = pd.to_datetime(df['factor_date'])

    if df.empty:
        print(f"No factor data found between {start_date} and {factor_date} for growth calculation.")
        return

    def resample_forward_fill(group):
        group = group.set_index('factor_date').resample('ME').ffill()
        group['ticker'] = group['ticker'].iloc[0]
        group['factor_name'] = group['factor_name'].iloc[0]
        return group

    df_monthly = df.groupby(['ticker', 'factor_name'], group_keys=False).apply(resample_forward_fill, include_groups=False).reset_index()

    lookbacks = [12, 60]
    results = []

    for factor in factors_to_grow:
        df_factor = df_monthly[df_monthly['factor_name'] == factor].copy()
        df_factor.sort_values(['ticker', 'factor_date'], inplace=True)

        for lookback in lookbacks:
            lag_col = f'lag_{lookback}'
            df_factor[lag_col] = df_factor.groupby('ticker')['factor_value'].shift(lookback)

        valid_mask = pd.Series(False, index=df_factor.index)
        for lookback in lookbacks:
            lag_col = f'lag_{lookback}'
            valid_mask |= (df_factor[lag_col].notna() & df_factor['factor_value'].notna() & (df_factor[lag_col] != 0))

        df_valid = df_factor[valid_mask]

        for lookback in lookbacks:
            lag_col = f'lag_{lookback}'
            mask = df_valid[lag_col].notna() & (df_valid[lag_col] != 0)
            df_tmp = df_valid[mask]

            growth_values = (df_tmp['factor_value'] - df_tmp[lag_col]) / df_tmp[lag_col]
            growth_values = growth_values.replace([np.inf, -np.inf], np.nan).dropna()

            if growth_values.empty:
                continue

            factor_results = list(
                zip(
                    df_tmp.loc[growth_values.index, 'ticker'],
                    df_tmp.loc[growth_values.index, 'factor_date'].dt.strftime('%Y-%m-%d'),
                    [f"{factor}_growth_{lookback}m"] * len(growth_values),
                    growth_values
                )
            )
            results.extend(factor_results)

    if not results:
        print(f"No growth factors computed for factor_date {factor_date}.")
        return

    print(f"Bulk copying {len(results)} growth factor records for {factor_date}...")

    # prepare CSV in memory
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerows(results)
    csv_buffer.seek(0)

    with conn.cursor() as cur:

        # Create temporary table for bulk insert
        cur.execute("""
            CREATE TEMP TABLE tmp_growth_factors (
                ticker TEXT,
                factor_date DATE,
                factor_name TEXT,
                factor_value FLOAT8
            ) ON COMMIT DROP
        """)
        conn.commit()

        # Bulk copy from CSV buffer into temp table
        cur.copy_expert("COPY tmp_growth_factors FROM STDIN WITH CSV", csv_buffer)
        conn.commit()

        # Upsert from temp table into main table
        upsert_sql = """
            INSERT INTO monthly_factors (ticker, factor_date, factor_name, factor_value)
            SELECT ticker, factor_date, factor_name, factor_value FROM tmp_growth_factors
            ON CONFLICT (ticker, factor_date, factor_name) DO UPDATE
            SET factor_value = EXCLUDED.factor_value
        """
        cur.execute(upsert_sql)
        conn.commit()

    print(f"Successfully bulk upserted {len(results)} growth factor records for {factor_date}.")


def compute_and_upsert_growth_factors_single_batch(conn_params, engine_params, max_lookback_months, batch_dates):
    batch_start = batch_dates[0]
    batch_end = batch_dates[-1]
    engine = create_engine(
        f"postgresql+psycopg2://{engine_params['user']}:{engine_params['password']}@"
        f"{engine_params['host']}:{engine_params['port']}/{engine_params['dbname']}"
    )
    conn = psycopg2.connect(**conn_params)
    print(f"Processing growth factors for batch {batch_start} to {batch_end}...")
    extended_start = batch_start - relativedelta(months=max_lookback_months)

    factors_to_grow = ['ttm_sales', 'ttm_fcf', 'ttm_net_income',
                      'ttm_eps', 'ttm_div', 'shares_diluted']

    sql_fetch = """
        SELECT ticker, factor_date, factor_name, factor_value
        FROM monthly_factors
        WHERE factor_name = ANY(%s)
          AND factor_date BETWEEN %s AND %s
        ORDER BY ticker, factor_name, factor_date
    """
    df_raw = pd.read_sql(sql_fetch, con=engine,
                         params=(factors_to_grow, extended_start, batch_end))
    df_raw['factor_date'] = pd.to_datetime(df_raw['factor_date'])

    if df_raw.empty:
        print(f"No data between {extended_start} and {batch_end}")
        conn.close()
        return 0

    def resample_forward_fill(group):
        if isinstance(group.name, tuple):
            ticker, factor_name = group.name
        else:
            ticker = group.name
            factor_name = None
        group = group.set_index('factor_date').resample('ME').ffill()
        group['ticker'] = ticker
        if factor_name is not None:
            group['factor_name'] = factor_name
        return group.reset_index()

    df_monthly = (df_raw.groupby(['ticker', 'factor_name'], group_keys=False)
                  .apply(resample_forward_fill, include_groups=False)
                  .reset_index(drop=True))

    lookbacks = [12, 60]
    results = []
    print(f"Calculating growth factors...")
    for factor in factors_to_grow:
        df_factor = df_monthly[df_monthly['factor_name'] == factor].copy()
        df_factor.sort_values(['ticker', 'factor_date'], inplace=True)

        for lb in lookbacks:
            lag_col = f'lag_{lb}'
            df_factor[lag_col] = df_factor.groupby('ticker')['factor_value'].shift(lb)

            valid_mask = (df_factor[lag_col].notna() &
                          df_factor['factor_value'].notna() &
                          (df_factor[lag_col] != 0))
            df_valid = df_factor[valid_mask].copy()

            growth_vals = ((df_valid['factor_value'] - df_valid[lag_col]) /
                           df_valid[lag_col])
            growth_vals = growth_vals.replace([float('inf'), float('-inf')], pd.NA).dropna()
            df_valid = df_valid.loc[growth_vals.index]

            if growth_vals.empty:
                continue

            factor_records = list(zip(
                df_valid['ticker'],
                df_valid['factor_date'].dt.strftime('%Y-%m-%d'),
                [f"{factor}_growth_{lb}m"] * len(growth_vals),
                growth_vals,
            ))
            results.extend(factor_records)
    print(f"Preparing to upsert {len(results)} growth factor records for batch {batch_start} to {batch_end}...")
    if results:
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerows(results)
        csv_buffer.seek(0)

        with conn.cursor() as cur:
            cur.execute("""
                CREATE TEMP TABLE tmp_growth_factors (
                    ticker TEXT,
                    factor_date DATE,
                    factor_name TEXT,
                    factor_value FLOAT8
                ) ON COMMIT DROP
            """)

            cur.copy_expert("COPY tmp_growth_factors FROM STDIN WITH CSV", csv_buffer)

            upsert_sql = """
                INSERT INTO monthly_factors (ticker, factor_date, factor_name, factor_value)
                SELECT ticker, factor_date, factor_name, factor_value FROM tmp_growth_factors
                ON CONFLICT (ticker, factor_date, factor_name) DO UPDATE
                SET factor_value = EXCLUDED.factor_value
            """
            cur.execute(upsert_sql)

        conn.commit()
        print(f"Batch {batch_start} to {batch_end} upserted {len(results)} records.")
    else:
        print(f"Batch {batch_start} to {batch_end} no growth factors to upsert.")

    conn.close()
    return len(results)


def compute_and_upsert_growth_factors_batch_vectorized(
        conn_params, engine_params, factor_dates, batch_period_months=4,
        max_lookback_months=84, max_workers=2):
    batches = [factor_dates[i:i + batch_period_months] for i in range(0, len(factor_dates), batch_period_months)]
    print(f"Processing growth factors in {len(batches)} batches with up to {max_workers} workers...")
    worker = partial(compute_and_upsert_growth_factors_single_batch,
                     conn_params, engine_params, max_lookback_months)

    with mp.Pool(max_workers) as pool:
        results = pool.map(worker, batches)

    print(f"Total upserted records: {sum(results)}")


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
        if not rows:
            print(f"No valid factors found for date {factor_date}")
        else:
            print(f"Valid data points count for factors on {factor_date}:")
            for factor_name, count in rows:
                print(f"  {factor_name}: {count}")
    return rows


def get_date_chunks(start, end, chunk_size_months):
    current_start = start
    while current_start <= end:
        current_end = min(current_start + relativedelta(months=chunk_size_months) - timedelta(days=1), end)
        yield current_start, current_end
        current_start = current_end + timedelta(days=1)


def run_full_rebuild(conn_params, engine_params, indicators, batch_period_months=24, max_lookback_months=84):
    init_connection_pool(conn_params)
    start_date = date(2000, 1, 1)
    today = datetime.today()
    end_date = (today.replace(day=1) - timedelta(days=1)).date()

    all_chunks = list(get_date_chunks(start_date, end_date, batch_period_months))
    all_factor_dates = []

    # Step 1: Compute extended factors for all batches sequentially
    for chunk_start, chunk_end in tqdm(all_chunks, desc="Extended factor rebuild batches"):
        extended_start = max(pd.Timestamp('2000-01-01'), pd.Timestamp(chunk_start) - relativedelta(months=max_lookback_months))
        factor_dates = get_monthly_eom_dates(chunk_start, chunk_end)
        all_factor_dates.extend(factor_dates)

        compute_and_upsert_extended_factors_batch(conn_params, engine_params, indicators, factor_dates, extended_start)

    # Step 2: Compute growth factors vectorized after all extended factors are done
    compute_and_upsert_growth_factors_batch_vectorized(conn_params, engine_params, all_factor_dates, batch_period_months, max_lookback_months)

    close_connection_pool()

def run_incremental_update(conn_params, engine_params, indicators, since_date, batch_period_months=24, max_lookback_months=84):
    init_connection_pool(conn_params)
    today = datetime.today()
    last_month_end = (today.replace(day=1) - timedelta(days=1)).date()
    all_chunks = list(get_date_chunks(since_date, last_month_end, batch_period_months))
    all_factor_dates = []

    # Step 1: Compute extended factors for all incremental batches sequentially
    for chunk_start, chunk_end in tqdm(all_chunks, desc="Extended factor incremental batches"):
        # Adjust extended_start to be relative to chunk_start rather than fixed 2018 date
        extended_start = max(pd.Timestamp(chunk_start) - relativedelta(months=max_lookback_months), pd.Timestamp(chunk_start))
        factor_dates = get_monthly_eom_dates(chunk_start, chunk_end)
        all_factor_dates.extend(factor_dates)

        compute_and_upsert_extended_factors_batch(conn_params, engine_params, indicators, factor_dates, extended_start)

    # Step 2: Compute growth factors vectorized after all extended factors are done
    # Reduce batch_period_months to 6 months for growth factor computation batches
    compute_and_upsert_growth_factors_batch_vectorized(
        conn_params, engine_params, all_factor_dates,
        batch_period_months=2,  # smaller chunks to improve processing efficiency
        max_lookback_months=max_lookback_months
    )

    close_connection_pool()


def rebuild_specific_factors(conn_params, engine_params, target_factors, start_date=None, end_date=None,
                            batch_period_months=24, max_lookback_months=84):
    init_connection_pool(conn_params)

    if start_date is None:
        start_date = date(2000, 1, 1)
    if end_date is None:
        today = datetime.today()
        end_date = (today.replace(day=1) - timedelta(days=1)).date()

    print(f"Running rebuild for specified factors {target_factors} from {start_date} to {end_date}")

    dependency_map = {
        'div_yield': ['ttm_div', 'market_cap'],
        'div_cover': ['ttm_div', 'ttm_net_income'],
        'debt_to_equity': ['long_term_debt', 'short_term_debt', 'cash_and_equiv', 'total_equity_last'],
        'ttm_eps': ['ttm_net_income', 'shares_diluted'],
        'enterprise_value': ['market_cap', 'long_term_debt', 'short_term_debt', 'cash_and_equiv'],
        'yrs_to_cash': ['ttm_cash_flows', 'ttm_capex', 'ttm_fcf', 'cash_and_equiv', 'long_term_debt', 'short_term_debt', 'cash_and_equiv'],
        'post_tax': ['ttm_tax', 'ttm_pbt'],
        'working_capital': ['curr_assets', 'curr_liab'],
        'invested_capital_nf': ['working_capital', 'cash_and_equiv', 'net_fixed_assets'],
        'invested_capital_f': ['total_equity_last', 'preferred_equity_last'],
        'invested_capital': ['invested_capital_nf', 'invested_capital_f'],
        'roic': ['ttm_ebit', 'post_tax', 'invested_capital'],
        'np_margin': ['ttm_net_income', 'ttm_sales'],
        'ebit_yield': ['ttm_ebit', 'enterprise_value'],
        'pe': ['ttm_net_income', 'market_cap'],
    }

    needed_raw_factors = set()
    for f in target_factors:
        needed_raw_factors.update(dependency_map.get(f, []))

    indicators_to_build = [ind for ind in indicators if ind['factor_name'] in needed_raw_factors]

    extended_factors = {
        'working_capital', 'ttm_eps', 'post_tax', 'ttm_fcf', 'invested_capital',
        'enterprise_value', 'ebit_yield', 'pe', 'pb', 'roe', 'roa', 'div_yield',
        'div_cover', 'debt_to_equity', 'roic', 'fcf_yield', 'gp_margin',
        'np_margin', 'yrs_to_cash',
    }

    growth_factors = ['ttm_sales', 'ttm_fcf', 'ttm_net_income', 'ttm_eps', 'ttm_div', 'shares_diluted']

    all_chunks = list(get_date_chunks(start_date, end_date, batch_period_months))
    all_factor_dates = []

    if any(f in extended_factors for f in target_factors):
        # Step 1: Compute extended factors in batches sequentially
        for chunk_start, chunk_end in tqdm(all_chunks, desc="Extended factor rebuild batches"):
            extended_start = max(pd.Timestamp('2000-01-01'), pd.Timestamp(chunk_start) - relativedelta(months=max_lookback_months))
            factor_dates = get_monthly_eom_dates(chunk_start, chunk_end)
            all_factor_dates.extend(factor_dates)

            compute_and_upsert_extended_factors_batch(conn_params, engine_params, indicators_to_build, factor_dates, extended_start)
    else:
        # Just accumulate factor_dates if no extended factors requested
        for chunk_start, chunk_end in all_chunks:
            factor_dates = get_monthly_eom_dates(chunk_start, chunk_end)
            all_factor_dates.extend(factor_dates)

    # Step 2: Compute growth factors vectorized once over entire relevant date range
    if any(f in growth_factors for f in target_factors):
        compute_and_upsert_growth_factors_batch_vectorized(
            conn_params, engine_params, all_factor_dates, batch_period_months, max_lookback_months
        )

    # Step 3: Count valid factors per date for logging/debug
    with psycopg2.connect(**conn_params) as conn:
        for factor_date in all_factor_dates:
            count_valid_factors_per_date(conn, factor_date)
        conn.commit()

    close_connection_pool()


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(project_root, 'config.yml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    params = config['database']

    conn_params = dict(
        dbname=params['dbname'],
        user=params['user'],
        password=params['password'],
        host=params['host'],
        port=params['port']
    )

    engine_params = {
        'dbname': params['dbname'],
        'user': params['user'],
        'password': params['password'],
        'host': params['host'],
        'port': params['port']
    }

    try:
        mode = decide_mode()
        # mode = 'full'
        if mode == 'full':
            run_full_rebuild(conn_params, engine_params, indicators)
        elif mode == 'incremental':
            since_date = datetime.today().date() - relativedelta(months=24)
            run_incremental_update(
                conn_params, engine_params, indicators, since_date,
                batch_period_months=2,  # smaller update chunk size
                max_lookback_months=84  # keep large history window
            )
        else:
            print("Not scheduled to run today")
    finally:
        close_connection_pool()
        print("DB connection pool closed.")
