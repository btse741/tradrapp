import streamlit as st
import os
import asyncio
import datetime
import time
import pandas as pd
import plotly.graph_objects as go
from models.etf_momentum import data_update, computations, visualization, telegram_notify

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FILTERED_TRADES_CSV = os.path.join(PROJECT_ROOT, 'output', 'filtered_trades_action_table.csv')
FILTERED_TRADES_IMG = os.path.join(PROJECT_ROOT, 'output', 'filtered_trades_action_table.png')
cumulative_12m_png = os.path.join(PROJECT_ROOT, 'output', 'cumulative_12m_performance.png')
strategy_performance_table = os.path.join(PROJECT_ROOT, 'output', 'strategy_performance_table.jpg')

def get_latest_file_mtime(filepaths):
    mtimes = [os.path.getmtime(fp) for fp in filepaths if os.path.exists(fp)]
    return max(mtimes) if mtimes else None


def data_is_fresh():
    last_update_file = os.path.join(PROJECT_ROOT, "data","etf_momentum_last_update.txt")
    if not os.path.exists(last_update_file):
        return False
    
    with open(last_update_file, "r") as f:
        last_update_str = f.read().strip()
    try:
        last_update_date = datetime.strptime(last_update_str, "%Y-%m-%d").date()
    except Exception:
        return False

    today = datetime.date.today()
    weekday = today.weekday()
    if weekday >= 5:
        friday = today - datetime.timedelta(days=(weekday - 4))
        return last_update_date >= friday
    else:
        return last_update_date >= today


@st.cache_data(ttl=3600)
def cached_computations():
    return computations.run_all()

async def send_telegram_images():
    if os.path.exists(FILTERED_TRADES_IMG):
        await telegram_notify.send_image_async(FILTERED_TRADES_IMG)
    if os.path.exists(cumulative_12m_png):
        await telegram_notify.send_image_async(cumulative_12m_png)
    if os.path.exists(strategy_performance_table):
        await telegram_notify.send_image_async(strategy_performance_table)


def run():
    st.title("ETF Momentum Model")

    if 'data_fresh' not in st.session_state:
        st.session_state.data_fresh = data_is_fresh()

    col1, col2, col3 = st.columns([1, 1, 1])

    if st.session_state.data_fresh:
        placeholder = st.empty()
        placeholder.success("Data is up to date.")
        time.sleep(3)
        placeholder.empty()
        result = cached_computations()

        returns_dict = result['returns_dict']
        rebalance_dates = result['rebalance_dates']

        df_perf, _ = visualization.generate_performance_table(returns_dict, rebalance_dates)
        st.subheader("Performance Table")
        st.dataframe(df_perf, use_container_width=True)

        fig = visualization.plot_interactive_cumulative_returns(returns_dict, rebalance_dates)
        st.subheader("Cumulative Returns Chart (Last 12 Months)")
        st.plotly_chart(fig, use_container_width=True)
        # Trades and Actions Table
        strategy_files = {
            'Blended': 'quantile_trade_log_blended_momentum_equal_weight.csv',
            'Clenow': 'quantile_trade_log_clenow_momentum_equal_weight.csv',
            'Carver': 'quantile_trade_log_carver_momentum_equal_weight.csv',
            'EWAC': 'quantile_trade_log_ewac_momentum_equal_weight.csv',
            'Composite': 'quantile_trade_log_composite_momentum_equal_weight.csv',
        }
        all_trade_logs = []
        for strat, filename in strategy_files.items():
            df = pd.read_csv(os.path.join(visualization.OUTPUT_FOLDER, filename))
            df['Strategy'] = strat
            all_trade_logs.append(df)
        all_trade_logs_df = pd.concat(all_trade_logs, ignore_index=True)
        trades_table, _ = visualization.generate_trades_action_table(all_trade_logs_df, strategy_files)
        st.subheader("Trades and Actions Table")
        st.dataframe(trades_table, use_container_width=True)



        # visualization.show_all(precomputed=result)
    else:
        cached_computations.clear()
        st.cache_data.clear()
        st.cache_resource.clear()

        data_progress = st.progress(0, text="Downloading Data")
        comp_progress = st.progress(0, text="Running Computations")

        def data_cb(pct):
            data_progress.progress(pct)

        def comp_cb(pct):
            comp_progress.progress(pct)

        try:
            cached_computations.clear()
            data_update.run_data_update(progress_callback=data_cb)
            computations.run_all(progress_callback=comp_cb)
            st.session_state.data_fresh = True
            placeholder = st.empty()
            placeholder.success("Data updated and computations complete.")
            time.sleep(3)
            placeholder.empty()

            result = cached_computations()
            visualization.show_all(precomputed=result)
        except Exception as e:
            st.error(f"Error during update: {e}")

        data_progress.empty()
        comp_progress.empty()

    if col3.button("Send Telegram"):
        placeholder = st.empty()
        placeholder.success("Telegram filtered trades table image sent!")
        time.sleep(3)
        placeholder.empty()
        try:
            asyncio.run(send_telegram_images())

        except Exception as e:
            st.error(f"Telegram sending failed: {e}")
