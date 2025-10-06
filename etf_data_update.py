from models.etf_momentum.data_update import run_data_update
from models.etf_momentum.computations import run_all
from models.etf_momentum.visualization import (
    generate_performance_table,
    plot_cumulative_returns,
    generate_trades_action_table,
    create_trades_action_table,
)
import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FILTERED_TRADES_CSV = os.path.join(PROJECT_ROOT, 'output', 'filtered_trades_action_table.csv')
FILTERED_TRADES_IMG = os.path.join(PROJECT_ROOT, 'output', 'filtered_trades_action_table.png')
cumulative_12m_png = os.path.join(PROJECT_ROOT, 'output', 'cumulative_12m_performance.png')
strategy_performance_table = os.path.join(PROJECT_ROOT, 'output', 'strategy_performance_table.jpg')

# Trades and Actions Table
strategy_files = {
    'Blended': 'quantile_trade_log_blended_momentum_equal_weight.csv',
    'Clenow': 'quantile_trade_log_clenow_momentum_equal_weight.csv',
    'Carver': 'quantile_trade_log_carver_momentum_equal_weight.csv',
    'EWAC': 'quantile_trade_log_ewac_momentum_equal_weight.csv',
    'Composite': 'quantile_trade_log_composite_momentum_equal_weight.csv',
}

def run():
        try:
        
            run_data_update(progress_callback=None)
            result = run_all(progress_callback=None)
        
        
        
            show_all(precomputed=result)
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
