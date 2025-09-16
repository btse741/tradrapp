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

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, 'output')

if __name__ == "__main__":
    # Step 1: Update raw data
    run_data_update()

    # Step 2: Run all computations
    result = run_all()

    returns_dict = result['returns_dict']
    rebalance_dates = result['rebalance_dates']

    # Step 3: Generate and save performance table and chart images
    generate_performance_table(returns_dict, rebalance_dates)
    plot_cumulative_returns(returns_dict, rebalance_dates)

    # Step 4: Prepare trade logs for all strategies
    strategy_files = {
        'Blended': 'quantile_trade_log_blended_momentum_equal_weight.csv',
        'Clenow': 'quantile_trade_log_clenow_momentum_equal_weight.csv',
        'Carver': 'quantile_trade_log_carver_momentum_equal_weight.csv',
        'EWAC': 'quantile_trade_log_ewac_momentum_equal_weight.csv',
        'Composite': 'quantile_trade_log_composite_momentum_equal_weight.csv',
    }
    all_trade_logs = []
    for strat, filename in strategy_files.items():
        df = pd.read_csv(os.path.join(OUTPUT_FOLDER, filename))
        df['Strategy'] = strat
        all_trade_logs.append(df)
    all_trade_logs_df = pd.concat(all_trade_logs, ignore_index=True)

    # Step 5: Generate and save trades action table image and filtered trades CSV
    generate_trades_action_table(all_trade_logs_df, strategy_files)
    create_trades_action_table(all_trade_logs_df, strategy_files)