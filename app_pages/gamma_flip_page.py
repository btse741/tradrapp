import streamlit as st
import os
import requests
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime, timedelta
import yaml
from streamlit_autorefresh import st_autorefresh

st_autorefresh(interval=60000, key="datarefresh")


# Set project root and data/output folders
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_FOLDER = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, 'output')


def load_config():
    config_path = os.path.join(PROJECT_ROOT, 'config.yml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


config = load_config()
api_key = config.get('av', {}).get('api_key')


os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@st.cache_data(ttl=300)
def load_latest_etf_candles(ticker, days=5):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": ticker,
        "interval": "15min",
        "apikey": api_key,
        "outputsize": "full"
    }
    response = requests.get(url, params=params)
    data = response.json()

    time_series_key = "Time Series (15min)"
    if time_series_key not in data:
        raise ValueError(f"Alpha Vantage API error or limit reached: {data.get('Note', data)}")

    ts_data = data[time_series_key]
    df = pd.DataFrame.from_dict(ts_data, orient='index')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    start_date = datetime.now() - timedelta(days=days)
    df = df[df.index >= start_date]

    df = df.rename(columns={
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "5. volume": "volume"
    })
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])

    df.reset_index(inplace=True)
    df.rename(columns={"index": "datetime"}, inplace=True)

    df_clean = clean_intraday_candles_mad(df, threshold=3)
    return df_clean


def clean_intraday_candles_mad(df, threshold=3.0):
    """
    Clean intraday candles using 3*MAD rule on 'close' prices to detect outliers.
    When an outlier close is detected, replace open, high, low, close with median candle values to preserve candle integrity.
    """
    df = df.copy()

    prices = df['close'].values
    median = np.median(prices)
    abs_dev = np.abs(prices - median)
    mad = np.median(abs_dev)
    if mad == 0:
        return df

    modified_z_score = 0.6745 * abs_dev / mad
    outliers = modified_z_score > threshold

    median_open = df['open'].median()
    median_high = df['high'].median()
    median_low = df['low'].median()
    median_close = median

    df.loc[outliers, ['open', 'high', 'low', 'close']] = [median_open, median_high, median_low, median_close]

    return df


def load_option_metrics(ticker):
    try:
        df_metrics = pd.read_csv(os.path.join(OUTPUT_FOLDER, 'option_metrics_data.csv'))
        df_ticker = df_metrics[df_metrics['symbol'] == ticker].sort_values('date', ascending=False)
        if df_ticker.empty:
            return None
        latest = df_ticker.iloc[0]
        return latest
    except Exception as e:
        st.error(f"Failed to load option metrics: {e}")
        return None
def plot_etf_with_metrics(df_etf, metrics):
    fig = go.Figure()

    # Plot candlesticks
    fig.add_trace(go.Candlestick(
        x=df_etf['datetime'],
        open=df_etf['open'],
        high=df_etf['high'],
        low=df_etf['low'],
        close=df_etf['close'],
        name='15min Candles',
        showlegend=False
    ))

    # Gamma flip line (yellow horizontal line + annotation)
    if metrics is not None and not pd.isna(metrics['gamma_flip_line']):
        gamma_level = round(metrics['gamma_flip_line'], 2)
        fig.add_shape(type="line",
            x0=df_etf['datetime'].min(), y0=gamma_level,
            x1=df_etf['datetime'].max() + timedelta(days=1), y1=gamma_level,
            line=dict(color="yellow", dash="dash"),
            xref='x', yref='y'
        )
        fig.add_annotation(
            x=df_etf['datetime'].max() + timedelta(hours=6),  # slightly to the right of last candle
            y=gamma_level,
            showarrow=True,
            arrowhead=3,
            ax=40,
            ay=0,
            font=dict(color="yellow"),
            bgcolor="black",
            bordercolor="yellow"
        )

    # Add expected ranges horizontal lines and print levels
    if metrics is not None:
        x0 = df_etf['datetime'].min()
        x1 = df_etf['datetime'].max()

        def add_expected_range_line(high_label, low_label, color):
            high_val = round(metrics.get(high_label), 2)
            low_val = round(metrics.get(low_label), 2)
            if pd.notna(high_val):
                print(f"{high_label}: {high_val}")  # Print level
                fig.add_shape(type="line",
                    x0=x0, y0=high_val,
                    x1=x1, y1=high_val,
                    line=dict(color=color, width=3),
                    xref='x', yref='y'
                )
            if pd.notna(low_val):
                print(f"{low_label}: {low_val}")  # Print level
                fig.add_shape(type="line",
                    x0=x0, y0=low_val,
                    x1=x1, y1=low_val,
                    line=dict(color=color, width=3),
                    xref='x', yref='y'
                )

        add_expected_range_line('1_day_ahead_high', '1_day_ahead_low', 'red')
        add_expected_range_line('1_wk_ahead_high', '1_wk_ahead_low', 'blue')
        add_expected_range_line('1_mth_ahead_high', '1_mth_ahead_low', 'green')

    # Compute y-axis range (for example, 20 EMA ± 4 std)
    ema_period = 60
    df_etf['ema20'] = df_etf['close'].ewm(span=ema_period, adjust=False).mean()
    df_etf['std20'] = df_etf['close'].rolling(window=ema_period).std()

    last_ema = df_etf['ema20'].dropna().iloc[-1]
    last_std = df_etf['std20'].dropna().iloc[-1]

    y_min = last_ema - 8 * last_std
    y_max = last_ema + 8 * last_std

    # Extend x-axis by one day
    x_min = df_etf['datetime'].min()
    x_max_extended = df_etf['datetime'].max() + timedelta(days=1)

    trading_start = 9.5   # 9:30 AM
    trading_end = 16.0    # 4:00 PM

    fig.update_layout(
        title=f"ETF Price & Option Metrics ({metrics['symbol'] if metrics is not None else ''})",
        xaxis_title="DateTime",
        yaxis_title="Price",
        legend=dict(visible=False),
        yaxis=dict(range=[y_min, y_max], autorange=False),
        xaxis=dict(
            range=[x_min, x_max_extended],
            rangebreaks=[
                dict(bounds=["sat", "mon"]),
                dict(bounds=[trading_end, trading_start], pattern="hour"),
            ],
            showspikes=True,
            spikemode='across+marker',
            spikesnap='cursor',
            showline=True,
        ),
        hovermode='x unified',
        margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig

def run():
    st.title("ETF Option Metrics Visualization")

    ticker = st.selectbox('Choose ETF', options=[
        "SPY", "QQQ", "IWM", "DIA", "HYG", "TLT", "SLV", "GLD"
    ])

    df_etf = load_latest_etf_candles(ticker)
    metrics = load_option_metrics(ticker)

    fig = plot_etf_with_metrics(df_etf, metrics)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
