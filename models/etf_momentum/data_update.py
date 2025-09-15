import os
import pandas as pd
import yfinance as yf
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_FOLDER = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, 'output')


os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


strat1_etf = ["SPY", "EFA", "AGG", "VNQ", "GSG",
              "FXE", "EEM", "GLD", "TIP", "LQD",
              "USO", "UNG", "HYG", "IWM", "QQQ"]
strat2_etf = ["DGRW", "GGRW", "DON", "DES", "DEM", "DTH", "DHS"]
strat3_etf = ["SPY", "VOO", "IVV", "EEM", "VTI", "TDVG", "VT", "FBND", "AOA", "AOR"]
strat4_etf = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLU", "XLI", "XLB", "XLRE"]
strat5_etf = list(set(strat1_etf + strat3_etf + strat4_etf))


DL_list = {
    "strat1_etf": strat1_etf,
    "strat2_etf": strat2_etf,
    "strat3_etf": list(set(strat3_etf + strat4_etf)),
    "strat4_etf": strat4_etf,
    "strat5_etf": strat5_etf,
}

from_date = "2000-01-01"
to_date = datetime.today().strftime("%Y-%m-%d")


def download_adjusted_prices(symbols, filename, progress_callback=None, progress_start=0, progress_end=100):
    data = yf.download(symbols, start=from_date, end=to_date, group_by='ticker', auto_adjust=True)
    if len(symbols) == 1:
        df = data['Close'].to_frame()
        df.columns = symbols
    else:
        close_data = pd.concat([data[ticker]['Close'] for ticker in symbols], axis=1)
        close_data.columns = symbols
        df = close_data
    df.reset_index(inplace=True)
    df.to_csv(os.path.join(DATA_FOLDER, filename), index=False)
    if progress_callback:
        progress_callback(progress_end)
    print(f"Exported adjusted prices to {filename}")


def download_unadjusted_close(symbols, filename, progress_callback=None, progress_start=0, progress_end=100):
    data = yf.download(symbols, start=from_date, end=to_date, group_by='ticker', auto_adjust=False)
    if len(symbols) == 1:
        df = data['Close'].to_frame()
        df.columns = symbols
    else:
        close_data = pd.concat([data[ticker]['Close'] for ticker in symbols], axis=1)
        close_data.columns = symbols
        df = close_data
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    df.to_csv(os.path.join(DATA_FOLDER, filename), index=False)
    if progress_callback:
        progress_callback(progress_end)
    print(f"Exported unadjusted close prices to {filename}")


def download_dividends(symbols, filename, progress_callback=None, progress_start=0, progress_end=100):
    all_divs = []
    for i, ticker in enumerate(symbols):
        ticker_obj = yf.Ticker(ticker)
        divs = ticker_obj.dividends
        if divs.empty:
            continue
        divs_monthly = divs.resample('ME').sum()
        divs_monthly.name = ticker
        all_divs.append(divs_monthly)

        # Update progress incrementally
        if progress_callback:
            progress = progress_start + (i + 1) * (progress_end - progress_start) / len(symbols)
            progress_callback(min(100, int(progress)))

    if all_divs:
        df_divs = pd.concat(all_divs, axis=1).fillna(0)
        df_divs.index.name = 'Date'
        df_divs.reset_index(inplace=True)
        df_divs.to_csv(os.path.join(DATA_FOLDER, filename), index=False)
        print(f"Exported dividends to {filename}")
    else:
        print("No dividend data to export.")


def run_data_update(progress_callback=None):
    total_tasks = len(DL_list) + 2  # total download tasks: adjusted prices + unadj close + dividends
    task_num = 0

    latest_dates = []

    for name, symbols in DL_list.items():
        start_pct = int(task_num / total_tasks * 100)
        end_pct = int((task_num + 1) / total_tasks * 100)
        download_adjusted_prices(symbols, f"{name}.csv",
                                progress_callback=progress_callback,
                                progress_start=start_pct,
                                progress_end=end_pct)
        # After download, get latest date from CSV file
        csv_path = os.path.join(DATA_FOLDER, f"{name}.csv")
        df = pd.read_csv(csv_path, parse_dates=['Date'])
        latest_dates.append(df['Date'].max())
        task_num += 1

    # strat2 unadjusted close prices
    start_pct = int(task_num / total_tasks * 100)
    end_pct = int((task_num + 1) / total_tasks * 100)
    download_unadjusted_close(strat2_etf, "strat2_unadj_close.csv",
                              progress_callback=progress_callback,
                              progress_start=start_pct,
                              progress_end=end_pct)
    task_num += 1

    # strat2 dividends
    start_pct = int(task_num / total_tasks * 100)
    end_pct = int((task_num + 1) / total_tasks * 100)
    download_dividends(strat2_etf, "strat2_dividends.csv",
                       progress_callback=progress_callback,
                       progress_start=start_pct,
                       progress_end=end_pct)

    # Determine the latest overall date from adjusted prices downloaded
    overall_latest_date = max(latest_dates).strftime('%Y-%m-%d')

    # Write/update the last update date file
    timestamp_file = os.path.join(DATA_FOLDER, "etf_momentum_last_update.txt")
    with open(timestamp_file, "w") as f:
        f.write(overall_latest_date)

    print(f"Data update completed. Last data date: {overall_latest_date}")
