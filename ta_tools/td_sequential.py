import pandas as pd
import yaml
import os
from datetime import datetime
from sqlalchemy import create_engine
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.graph_objs import Candlestick


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


# === TD Sequential Setup with Price Flip ===


def calculate_td_setups(close_prices):
    buy_setup = [0] * len(close_prices)
    sell_setup = [0] * len(close_prices)

    last_flip = None  # 'bearish' or 'bullish'

    for i in range(4, len(close_prices)):
        if close_prices[i] < close_prices[i - 4]:
            if last_flip != 'bearish':
                last_flip = 'bearish'
            if last_flip == 'bearish':
                buy_setup[i] = buy_setup[i - 1] + 1 if buy_setup[i - 1] > 0 else 1
            sell_setup[i] = 0

        elif close_prices[i] > close_prices[i - 4]:
            if last_flip != 'bullish':
                last_flip = 'bullish'
            if last_flip == 'bullish':
                sell_setup[i] = sell_setup[i - 1] + 1 if sell_setup[i - 1] > 0 else 1
            buy_setup[i] = 0

        else:
            buy_setup[i] = 0
            sell_setup[i] = 0

        if buy_setup[i] > 0 and close_prices[i] >= close_prices[i - 4]:
            buy_setup[i] = 0
        if sell_setup[i] > 0 and close_prices[i] <= close_prices[i - 4]:
            sell_setup[i] = 0

        buy_setup[i] = min(buy_setup[i], 9)
        sell_setup[i] = min(sell_setup[i], 9)

    return buy_setup, sell_setup


# === TD Sequential Countdown with deferred start and low/high comparisons ===


def calculate_td_countdowns(close, high, low, buy_setup, sell_setup):
    buy_countdown = [0] * len(close)
    sell_countdown = [0] * len(close)
    buy_countdown_bars = []
    sell_countdown_bars = []
    buy_signals = []
    sell_signals = []

    active_countdown = None
    countdown_started = False

    for i in range(len(close)):

        # Reset countdown flags at setup 9 bars, so countdown start is deferred
        if buy_setup[i] == 9:
            countdown_started = False
            active_countdown = None

        if sell_setup[i] == 9:
            countdown_started = False
            active_countdown = None

        # Deferred countdown start waiting for bar that meets countdown start condition
        if not countdown_started:
            if buy_setup[i] == 9 or active_countdown == 'buy':
                if i >= 2 and close[i] <= low[i - 2]:
                    active_countdown = 'buy'
                    countdown_started = True
                    buy_countdown_bars.clear()
                    sell_countdown_bars.clear()

            if sell_setup[i] == 9 or active_countdown == 'sell':
                if i >= 2 and close[i] >= high[i - 2]:
                    active_countdown = 'sell'
                    countdown_started = True
                    sell_countdown_bars.clear()
                    buy_countdown_bars.clear()

        # Countdown counting bars that meet conditions, non-consecutively
        if active_countdown == 'buy' and countdown_started and i >= 2:
            if close[i] <= low[i - 2]:
                if not buy_countdown_bars or i > buy_countdown_bars[-1]:
                    buy_countdown_bars.append(i)
                    buy_countdown[i] = len(buy_countdown_bars)

                    if len(buy_countdown_bars) == 13:
                        bar_13_low = low[i]
                        bar_8_close = close[buy_countdown_bars[7]]
                        low_2_before = low[i - 2]

                        # Final countdown bar 13 validation
                        if bar_13_low <= bar_8_close and bar_13_low <= low_2_before:
                            buy_signals.append(i)
                            active_countdown = None
                            countdown_started = False
                            buy_countdown_bars.clear()
                        else:
                            # Defer countdown bar 13 completion; remove count 13
                            buy_countdown_bars.pop()
                            buy_countdown[i] = len(buy_countdown_bars)
            else:
                buy_countdown[i] = 0

        elif active_countdown == 'sell' and countdown_started and i >= 2:
            if close[i] >= high[i - 2]:
                if not sell_countdown_bars or i > sell_countdown_bars[-1]:
                    sell_countdown_bars.append(i)
                    sell_countdown[i] = len(sell_countdown_bars)

                    if len(sell_countdown_bars) == 13:
                        bar_13_high = high[i]
                        bar_8_close = close[sell_countdown_bars[7]]
                        high_2_before = high[i - 2]

                        # Final countdown bar 13 validation
                        if bar_13_high >= bar_8_close and bar_13_high >= high_2_before:
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
            if not countdown_started:
                buy_countdown[i] = 0
                sell_countdown[i] = 0

        # Cancel countdown if opposite setup starts
        if active_countdown == 'buy' and sell_setup[i] >= 1:
            active_countdown = None
            countdown_started = False
            buy_countdown_bars.clear()
        if active_countdown == 'sell' and buy_setup[i] >= 1:
            active_countdown = None
            countdown_started = False
            sell_countdown_bars.clear()

    return buy_countdown, sell_countdown, buy_signals, sell_signals


# === Fetch Adjusted OHLC Prices from DB ===


def get_adjusted_ohlc(engine, ticker, years_back=5):
    query = """
        SELECT date, open, high, low, close, adj_close
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

    df['factor'] = df['adj_close'] / df['close']
    df['open'] = df['open'] * df['factor']
    df['high'] = df['high'] * df['factor']
    df['low'] = df['low'] * df['factor']
    df['close'] = df['adj_close']

    return df[['open', 'high', 'low', 'close']]


# === Interactive Plotly Candlestick Visualization ===


def plot_td_sequential_candlestick(df, buy_setup, sell_setup, buy_countdown, sell_countdown, buy_signals, sell_signals, ticker):
    cutoff_display = pd.Timestamp.today() - pd.DateOffset(months=12)
    df_plot = df[df.index >= cutoff_display]
    index_map = {date: i for i, date in enumerate(df.index)}
    filtered_indices = [index_map[date] for date in df_plot.index if date in index_map]

    fig = make_subplots(specs=[[{"secondary_y": False}]])

    fig.add_trace(Candlestick(
        x=df_plot.index,
        open=df_plot['open'],
        high=df_plot['high'],
        low=df_plot['low'],
        close=df_plot['close'],
        name='Price'
    ))

    # Darker green for buy setup numbers
    for i in filtered_indices:
        val = buy_setup[i]
        if val > 0:
            fig.add_annotation(
                x=df.index[i], y=df['low'].iloc[i] * 0.995,
                text=str(val),
                showarrow=False,
                font=dict(color='darkgreen', size=10)
            )

    # Darker red for sell setup numbers
    for i in filtered_indices:
        val = sell_setup[i]
        if val > 0:
            fig.add_annotation(
                x=df.index[i], y=df['high'].iloc[i] * 1.005,
                text=str(val),
                showarrow=False,
                font=dict(color='darkred', size=10)
            )

    # Lighter green for buy countdown markers
    fig.add_trace(go.Scatter(
        x=[df.index[i] for i in filtered_indices if buy_countdown[i] > 0],
        y=[df['low'].iloc[i] * 0.98 for i in filtered_indices if buy_countdown[i] > 0],
        mode='markers',
        marker=dict(color='lightgreen', size=8, symbol='circle'),
        name='Buy Countdown'
    ))

    # Lighter red for sell countdown markers
    fig.add_trace(go.Scatter(
        x=[df.index[i] for i in filtered_indices if sell_countdown[i] > 0],
        y=[df['high'].iloc[i] * 1.02 for i in filtered_indices if sell_countdown[i] > 0],
        mode='markers',
        marker=dict(color='lightcoral', size=8, symbol='circle'),
        name='Sell Countdown'
    ))

    # Buy signals in green - only at countdown 13 complettion bars
    for idx in buy_signals:
        if df.index[idx] >= cutoff_display:
            fig.add_annotation(
                x=df.index[idx], y=df['low'].iloc[idx], text='BUY',
                showarrow=True, arrowhead=2, arrowsize=1, arrowcolor='green', ay=30,
                font=dict(color='green', size=12)
            )

    # Sell signals in red - only at countdown 13 completion bars
    for idx in sell_signals:
        if df.index[idx] >= cutoff_display:
            fig.add_annotation(
                x=df.index[idx], y=df['high'].iloc[idx], text='SELL',
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

    fig.show()


# === Main script ===


if __name__ == "__main__":
    ticker = "AAPL"  # Replace with your ticker symbol

    engine = create_engine_db()
    df = get_adjusted_ohlc(engine, ticker, years_back=5)

    close_prices = df['close'].tolist()
    high_prices = df['high'].tolist()
    low_prices = df['low'].tolist()

    buy_setup, sell_setup = calculate_td_setups(close_prices)
    buy_countdown, sell_countdown, buy_signals, sell_signals = calculate_td_countdowns(close_prices, high_prices, low_prices, buy_setup, sell_setup)

    plot_td_sequential_candlestick(df, buy_setup, sell_setup, buy_countdown, sell_countdown, buy_signals, sell_signals, ticker)
