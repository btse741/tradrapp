import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import dataframe_image as dfi
import plotly.graph_objects as go

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_FOLDER = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, 'output')

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def calc_period_return(series, end_date, days):
    start_date = end_date - pd.Timedelta(days=days)
    idx_start = series.index.get_indexer([start_date], method='nearest')[0]
    idx_end = series.index.get_indexer([end_date], method='nearest')[0]
    period_returns = series.iloc[idx_start:idx_end + 1]
    cum_ret = (1 + period_returns).prod() - 1
    return cum_ret

def generate_performance_table(returns_dict, rebalance_dates):
    end_date = rebalance_dates[-1] if len(rebalance_dates) > 0 else None
    last_rebalance_date = rebalance_dates[-1] if len(rebalance_dates) > 0 else None

    df_perf = pd.DataFrame(index=returns_dict.keys(),
                           columns=['Since Last Rebalance', '3 Months', '6 Months', '1 Year'])

    for strat, ret_series in returns_dict.items():
        cum_since_rebal = (1 + ret_series.loc[last_rebalance_date: end_date]).prod() - 1
        df_perf.loc[strat, 'Since Last Rebalance'] = cum_since_rebal
        df_perf.loc[strat, '3 Months'] = calc_period_return(ret_series, end_date, 63)
        df_perf.loc[strat, '6 Months'] = calc_period_return(ret_series, end_date, 126)
        df_perf.loc[strat, '1 Year'] = calc_period_return(ret_series, end_date, 252)

    df_perf = df_perf.apply(lambda col: col.map(lambda x: f"{float(x):.2%}"))

    perf_table_image_path = os.path.join(OUTPUT_FOLDER, 'strategy_performance_table.jpg')
    dfi.export(df_perf, perf_table_image_path)

    return df_perf, perf_table_image_path

def plot_interactive_cumulative_returns(returns_dict, rebalance_dates):
    end_date = rebalance_dates[-1] if len(rebalance_dates) > 0 else None
    if end_date is None:
        return go.Figure()  # empty figure

    start_date_12m_ago = end_date - pd.Timedelta(days=365)
    fig = go.Figure()

    for name, ret_series in returns_dict.items():
        # Filter returns for the last 12 months till end_date
        ret_sub = ret_series.loc[start_date_12m_ago:end_date]
        # Compute cumulative returns
        cum_ret = (1 + ret_sub).cumprod() - 1
        # Add line plot to Figure
        fig.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret.values, mode='lines', name=name, line=dict(width=2)))

    fig.update_layout(
        title='Cumulative Returns of Momentum Strategies vs SPY (Last 12 Months)',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        xaxis=dict(tickformat='%b %Y'),
        hovermode='x unified',
        template='plotly_white'
    )

    return fig

def plot_cumulative_returns(returns_dict, rebalance_dates):
    end_date = rebalance_dates[-1] if len(rebalance_dates) > 0 else None
    start_date_12m_ago = end_date - pd.Timedelta(days=365)

    plt.figure(figsize=(12, 6))

    for name, ret_series in returns_dict.items():
        ret_sub = ret_series.loc[start_date_12m_ago:end_date]
        cum_ret = (1 + ret_sub).cumprod() - 1
        plt.plot(cum_ret.index, cum_ret.values, label=name, linewidth=2)

    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title('Cumulative Returns of Momentum Strategies vs SPY (Last 12 Months)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)

    cumulative_performance_chart_path = os.path.join(OUTPUT_FOLDER, 'cumulative_12m_performance.png')
    plt.savefig(cumulative_performance_chart_path)
    plt.close()

    return cumulative_performance_chart_path


def generate_trades_action_table(all_trade_logs_df, strategy_files):
    rebalance_dates = sorted(all_trade_logs_df['Date'].unique())
    if len(rebalance_dates) < 2:
        raise Exception("Not enough rebalance dates in combined trade logs to determine trades.")
    last_date = rebalance_dates[-1]
    prev_date = rebalance_dates[-2]

    actions = []
    for strategy, group_df in all_trade_logs_df.groupby('Strategy'):
        weights_last = group_df.loc[group_df['Date'] == last_date].set_index('ETF')['Weight']
        weights_prev = group_df.loc[group_df['Date'] == prev_date].set_index('ETF')['Weight']
        all_etfs = set(weights_last.index).union(weights_prev.index)

        for etf in all_etfs:
            w_prev = weights_prev.get(etf, 0)
            w_last = weights_last.get(etf, 0)
            if w_prev <= 0 and w_last > 0:
                action = 'Buy'
            elif w_prev > 0 and w_last <= 0:
                action = 'Sell'
            elif w_prev >= 0 and w_last < 0:
                action = 'Sell Short'
            elif w_prev < 0 and w_last >= 0:
                action = 'Cover'
            else:
                action = 'Hold/Adjust'

            actions.append({
                'Strategy': strategy,
                'ETF': etf,
                'Action': action,
                'Weight Change': w_last - w_prev,
                'Prev Weight': w_prev
            })

    trades_to_take_df_all = pd.DataFrame(actions)
    trades_to_take_df_all['Current Position'] = trades_to_take_df_all['Prev Weight']
    trades_to_take_df_all.drop(columns=['Prev Weight'], inplace=True)

    strategy_order = list(strategy_files.keys())
    trades_to_take_df_all['Strategy'] = pd.Categorical(
        trades_to_take_df_all['Strategy'], categories=strategy_order, ordered=True
    )

    def trade_type(action):
        if action in ['Buy', 'Hold/Adjust']:
            return 'Long'
        elif action in ['Sell', 'Sell Short', 'Cover']:
            return 'Short'
        else:
            return 'Other'

    trades_to_take_df_all['TradeType'] = trades_to_take_df_all['Action'].map(trade_type)

    trades_sorted = trades_to_take_df_all.sort_values(
        ['Strategy', 'TradeType', 'ETF'],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    trades_sorted = trades_sorted.drop(columns=['TradeType'])

    final_trades_table = trades_sorted[['Strategy', 'ETF', 'Action', 'Weight Change', 'Current Position']]

    trades_table_image_path = os.path.join(OUTPUT_FOLDER, 'trades_action_table_all_strategies_sorted.jpg')
    dfi.export(final_trades_table, trades_table_image_path)

    return final_trades_table, trades_table_image_path


def create_trades_action_table(all_trade_logs_df, strategy_files):
    """
    Generate full trades action table using existing function, then filter for trades with non-zero Weight Change,
    save filtered trades CSV for inspection, and return the filtered DataFrame.
    """
    full_trades_df, _ = generate_trades_action_table(all_trade_logs_df, strategy_files)

    filtered_trades_df = full_trades_df[full_trades_df['Weight Change'] != 0].copy()

    csv_path = os.path.join(OUTPUT_FOLDER, 'filtered_trades_action_table.csv')
    filtered_trades_df.to_csv(csv_path, index=False)

    return filtered_trades_df

def csv_to_image(csv_path, image_path):
    df = pd.read_csv(csv_path)
    # Export the DataFrame as an image
    dfi.export(df, image_path)


def show_all(precomputed=None):
    import streamlit as st
    from models.etf_momentum.computations import run_all

    # Use precomputed results if provided to avoid redundant calculations
    if precomputed is None:
        result = run_all()
    else:
        result = precomputed

    returns_dict = result['returns_dict']
    prices = result['prices']
    rebalance_dates = result['rebalance_dates']

    df_perf, perf_img_path = generate_performance_table(returns_dict, rebalance_dates)
    st.subheader("Performance Table")
    st.dataframe(df_perf)

    chart_path = plot_cumulative_returns(returns_dict, rebalance_dates)
    st.subheader("Cumulative Returns Chart (Last 12 Months)")
    st.image(chart_path)

    fig = plot_interactive_cumulative_returns(returns_dict, rebalance_dates)
    st.subheader("Cumulative Returns Chart (Last 12 Months)")
    st.plotly_chart(fig, use_container_width=True)


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

    trades_table, trades_img_path = generate_trades_action_table(all_trade_logs_df, strategy_files)
    st.subheader("Trades and Actions Table")
    st.dataframe(trades_table)

    # Optionally show image of trade action table
    # st.image(trades_img_path, caption="Trades Action Table Image")

    # Create fully prepared trades DataFrame for telegram from all_trade_logs_df
    final_trades_df = create_trades_action_table(all_trade_logs_df, strategy_files)

    # Add all_trade_logs_df and final_trades_df to result dict for downstream use
    result['all_trade_logs_df'] = all_trade_logs_df
    result['final_trades_df'] = final_trades_df

    return result

