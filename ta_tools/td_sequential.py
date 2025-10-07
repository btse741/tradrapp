import pandas as pd
import yaml
import os
import sys
from datetime import datetime
from sqlalchemy import create_engine
import psycopg2.extras
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# === Load config and DB connection helpers ===

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
config_path = os.path.join(project_root, 'config.yml')
data_folder = os.path.join(project_root, 'data', 'sf_data')

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

params = config['database']


def create_engine_db():
    connection_string = (
        f"postgresql+psycopg2://{params['user']}:{params['password']}"
        f"@{params['host']}:{params['port']}/{params['dbname']}"
    )
    engine = create_engine(connection_string)
    return engine


def connect_db():
    import psycopg2
    conn = psycopg2.connect(
        dbname=params['dbname'],
        user=params['user'],
        password=params['password'],
        host=params['host'],
        port=params['port'],
    )
    conn.set_client_encoding('UTF8')
    return conn


def upsert_df_to_db(df, table_name, conn, primary_keys):
    """
    Upsert pandas DataFrame into PostgreSQL table.
    """
    if df.empty:
        print("DataFrame empty, skipping upsert.")
        return

    cols = list(df.columns)
    values_placeholders = ", ".join(["%s"] * len(cols))
    update_assignments = ", ".join([
        f"{col} = EXCLUDED.{col}" for col in cols if col not in primary_keys
    ])
    conflict_keys = ", ".join(primary_keys)

    insert_sql = f"""
    INSERT INTO {table_name} ({', '.join(cols)})
    VALUES ({values_placeholders})
    ON CONFLICT ({conflict_keys}) DO UPDATE
    SET {update_assignments};
    """

    data_tuples = [tuple(x) for x in df[cols].to_numpy()]

    try:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, insert_sql, data_tuples)
        conn.commit()
        print(f"Upserted {len(df)} rows into {table_name}.")
    except Exception as e:
        conn.rollback()
        print(f"Error upserting into {table_name}: {e}")
        raise


def calculate_td_setups(prices):
    buy_setup = [0] * len(prices)
    sell_setup = [0] * len(prices)
    setup_completed = False

    for i in range(4, len(prices)):
        if setup_completed:
            buy_setup[i] = 0
            sell_setup[i] = 0
            if (buy_setup[i - 1] == 9 and prices[i] >= prices[i - 4]) or \
               (sell_setup[i - 1] == 9 and prices[i] <= prices[i - 4]):
                setup_completed = False
            continue

        if prices[i] < prices[i - 4]:
            buy_setup[i] = buy_setup[i - 1] + 1 if buy_setup[i - 1] > 0 else 1
            sell_setup[i] = 0
            if buy_setup[i] == 9:
                setup_completed = True
        elif prices[i] > prices[i - 4]:
            sell_setup[i] = sell_setup[i - 1] + 1 if sell_setup[i - 1] > 0 else 1
            buy_setup[i] = 0
            if sell_setup[i] == 9:
                setup_completed = True
        else:
            buy_setup[i] = 0
            sell_setup[i] = 0

    return buy_setup, sell_setup


def calculate_td_countdowns(prices, buy_setup, sell_setup):
    buy_countdown = [0] * len(prices)
    sell_countdown = [0] * len(prices)
    buy_countdown_bars = []
    sell_countdown_bars = []
    buy_signals = []
    sell_signals = []

    active_countdown = None
    countdown_started = False

    for i in range(len(prices)):
        if buy_setup[i] == 9:
            if i >= 2 and prices[i] < prices[i - 2]:
                active_countdown = 'buy'
                countdown_started = True
                buy_countdown_bars.clear()
                sell_countdown_bars.clear()
            else:
                countdown_started = False

        elif sell_setup[i] == 9:
            if i >= 2 and prices[i] > prices[i - 2]:
                active_countdown = 'sell'
                countdown_started = True
                sell_countdown_bars.clear()
                buy_countdown_bars.clear()
            else:
                countdown_started = False

        if active_countdown == 'buy' and countdown_started and i >= 2:
            if prices[i] <= prices[i - 2]:
                if not buy_countdown_bars or i > buy_countdown_bars[-1]:
                    buy_countdown_bars.append(i)
                    buy_countdown[i] = len(buy_countdown_bars)
                    if len(buy_countdown_bars) == 13:
                        bar_13 = prices[i]
                        bar_8 = prices[buy_countdown_bars[7]]
                        low_2_before = prices[i - 2]
                        if bar_13 <= bar_8 and bar_13 <= low_2_before:
                            buy_signals.append(i)
                            active_countdown = None
                            countdown_started = False
                            buy_countdown_bars.clear()
                        else:
                            buy_countdown_bars.pop()
                            buy_countdown[i] = len(buy_countdown_bars)
            else:
                buy_countdown[i] = 0

        elif active_countdown == 'sell' and countdown_started and i >= 2:
            if prices[i] >= prices[i - 2]:
                if not sell_countdown_bars or i > sell_countdown_bars[-1]:
                    sell_countdown_bars.append(i)
                    sell_countdown[i] = len(sell_countdown_bars)
                    if len(sell_countdown_bars) == 13:
                        bar_13 = prices[i]
                        bar_8 = prices[sell_countdown_bars[7]]
                        high_2_before = prices[i - 2]
                        if bar_13 >= bar_8 and bar_13 >= high_2_before:
                            sell_signals.append(i)
                            active_countdown = None
                            countdown_started = False
                            sell_countdown_bars.clear()
                        else:
                            sell_countdown_bars.pop()
                            sell_countdown[i] = len(sell_countdown_bars)
            else:
                sell_countdown[i] = 0

        else:
            buy_countdown[i] = 0
            sell_countdown[i] = 0

        if active_countdown == 'buy' and sell_setup[i] >= 1:
            active_countdown = None
            countdown_started = False
            buy_countdown_bars.clear()

        if active_countdown == 'sell' and buy_setup[i] >= 1:
            active_countdown = None
            countdown_started = False
            sell_countdown_bars.clear()

    return buy_countdown, sell_countdown, buy_signals, sell_signals


def get_closing_prices(engine, ticker, years_back=5):
    query = """
        SELECT date, close
        FROM daily_prices
        WHERE ticker = %s AND date >= %s
        ORDER BY date ASC
    """
    cutoff_date = pd.Timestamp.today() - pd.DateOffset(years=years_back)
    df = pd.read_sql(query, engine, params=(ticker, cutoff_date))
    if df.empty:
        raise ValueError(f"No price data found for ticker '{ticker}'")
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df


def plot_td_sequential_interactive(df, buy_setup, sell_setup, buy_countdown, sell_countdown, buy_signals, sell_signals, ticker):
    cutoff_display = pd.Timestamp.today() - pd.DateOffset(months=12)
    df_plot = df[df.index >= cutoff_display]
    index_map = {date: i for i, date in enumerate(df.index)}
    filtered_indices = [index_map[date] for date in df_plot.index if date in index_map]

    fig = make_subplots(specs=[[{"secondary_y": False}]])

    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot['close'],
        mode='lines',
        name='Close Price',
        line=dict(color='black')
    ))

    for i in filtered_indices:
        val = buy_setup[i]
        if val == 9:
            fig.add_annotation(
                x=df.index[i], y=df['close'].iloc[i] * 0.995,
                text='9',
                showarrow=False,
                font=dict(color='darkgreen', size=12)
            )

    for i in filtered_indices:
        val = sell_setup[i]
        if val == 9:
            fig.add_annotation(
                x=df.index[i], y=df['close'].iloc[i] * 1.005,
                text='9',
                showarrow=False,
                font=dict(color='darkred', size=12)
            )

    fig.add_trace(go.Scatter(
        x=[df.index[i] for i in filtered_indices if buy_countdown[i] > 0],
        y=[df['close'].iloc[i] * 0.98 for i in filtered_indices if buy_countdown[i] > 0],
        mode='markers',
        marker=dict(color='lightgreen', size=8, symbol='circle'),
        name='Buy Countdown'
    ))

    fig.add_trace(go.Scatter(
        x=[df.index[i] for i in filtered_indices if sell_countdown[i] > 0],
        y=[df['close'].iloc[i] * 1.02 for i in filtered_indices if sell_countdown[i] > 0],
        mode='markers',
        marker=dict(color='lightcoral', size=8, symbol='circle'),
        name='Sell Countdown'
    ))

    for idx in buy_signals:
        if df.index[idx] >= cutoff_display:
            fig.add_annotation(
                x=df.index[idx], y=df['close'].iloc[idx], text='BUY',
                showarrow=True, arrowhead=2, arrowsize=1, arrowcolor='green', ay=30,
                font=dict(color='green', size=12)
            )

    for idx in sell_signals:
        if df.index[idx] >= cutoff_display:
            fig.add_annotation(
                x=df.index[idx], y=df['close'].iloc[idx], text='SELL',
                showarrow=True, arrowhead=2, arrowsize=1, arrowcolor='red', ay=-30,
                font=dict(color='red', size=12)
            )

    fig.update_layout(
        title=f"TD Sequential Indicator for {ticker} (Last 12 Months)",
        xaxis_title="Date", yaxis_title="Price",
        xaxis=dict(rangeslider=dict(visible=True)),
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Commented out to prevent pop-up during batch runs
    # fig.show()


def build_td_sequential_history(ticker_list):
    conn = None
    try:
        conn = connect_db()
        engine = create_engine_db()
        cur = conn.cursor()

        cur.execute("TRUNCATE TABLE td_sequential;")
        conn.commit()

        for ticker in ticker_list:
            print(f"Building history for ticker: {ticker}")
            df_price = get_closing_prices(engine, ticker, years_back=20)
            closing_prices = df_price['close'].tolist()

            buy_setup, sell_setup = calculate_td_setups(closing_prices)
            buy_countdown, sell_countdown, buy_signals, sell_signals = calculate_td_countdowns(closing_prices, buy_setup, sell_setup)

            df_td_seq = pd.DataFrame({
                'ticker': ticker,
                'date': df_price.index,
                'buy_setup': buy_setup,
                'sell_setup': sell_setup,
                'buy_countdown': buy_countdown,
                'sell_countdown': sell_countdown,
                'buy_signal': [i in buy_signals for i in range(len(buy_setup))],
                'sell_signal': [i in sell_signals for i in range(len(sell_setup))],
            })

            upsert_df_to_db(df_td_seq, 'td_sequential', conn, primary_keys=['ticker', 'date'])
            print(f"TD Sequential history built for {ticker}")

        print("History build complete.")
    except Exception as e:
        print(f"Error building history: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


def update_td_sequential_incremental(ticker_list):
    conn = None
    try:
        conn = connect_db()
        engine = create_engine_db()
        cur = conn.cursor()

        for ticker in ticker_list:
            print(f"Updating ticker: {ticker}")
            cur.execute("SELECT MAX(date) FROM td_sequential WHERE ticker = %s;", (ticker,))
            row = cur.fetchone()
            last_date = row[0] if row else None

            if not last_date:
                print(f"No existing data for {ticker}, consider running full build.")
                continue

            start_date = last_date - pd.Timedelta(days=30)
            query = """
                SELECT date, close FROM daily_prices 
                WHERE ticker = %s AND date >= %s
                ORDER BY date ASC
            """
            df_price = pd.read_sql(query, engine, params=(ticker, start_date))
            if df_price.empty:
                print(f"No new price data for {ticker} since {start_date}.")
                continue

            df_price['date'] = pd.to_datetime(df_price['date'])
            df_price.set_index('date', inplace=True)
            closing_prices = df_price['close'].tolist()

            buy_setup, sell_setup = calculate_td_setups(closing_prices)
            buy_countdown, sell_countdown, buy_signals, sell_signals = calculate_td_countdowns(closing_prices, buy_setup, sell_setup)

            df_td_seq = pd.DataFrame({
                'ticker': ticker,
                'date': df_price.index,
                'buy_setup': buy_setup,
                'sell_setup': sell_setup,
                'buy_countdown': buy_countdown,
                'sell_countdown': sell_countdown,
                'buy_signal': [i in buy_signals for i in range(len(buy_setup))],
                'sell_signal': [i in sell_signals for i in range(len(sell_setup))],
            })

            upsert_df_to_db(df_td_seq, 'td_sequential', conn, primary_keys=['ticker', 'date'])
            print(f"TD Sequential updated for {ticker} through {df_price.index[-1]}.")

        print("Incremental update complete.")
    except Exception as e:
        print(f"Error updating data: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    engine = create_engine_db()
    conn = connect_db()

    with conn.cursor() as cur:
        cur.execute("SELECT MAX(date) FROM daily_prices;")
        last_date = cur.fetchone()[0]

    query_top100 = """
        SELECT ticker FROM daily_prices
        WHERE date = %s AND adj_close IS NOT NULL AND shares_outstanding IS NOT NULL
        ORDER BY (adj_close * shares_outstanding) DESC
        LIMIT 100;
    """
    df_top100 = pd.read_sql(query_top100, engine, params=(last_date,))
    tickers_top100 = df_top100['ticker'].tolist()
    print(f"Top 100 tickers by market cap on {last_date}: {tickers_top100}")

    build_td_sequential_history(tickers_top100)

    conn.close()

