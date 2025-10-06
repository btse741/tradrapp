import os
import pandas as pd
from datetime import datetime


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
DATA_FOLDER = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, 'output')

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

etf_list = ["SPY", "QQQ", "IWM", "DIA", "HYG", "TLT", "SLV", "GLD"]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)  # Make sure directory exists
output_file = os.path.join(OUTPUT_FOLDER, "option_metrics_data.csv")

def run_all():
    # Move import here to avoid circular import error
    from models.options.option_metrics import process_ticker_wrapper
    
    today = datetime.now()
    all_results = []

    for etf in etf_list:
        print(f"Processing {etf} on {today.strftime('%Y-%m-%d')}")
        res = process_ticker_wrapper(etf, today)
        if res:
            all_results.append(res)

    if all_results:
        df_new = pd.DataFrame(all_results)

        if os.path.exists(output_file):
            df_existing = pd.read_csv(output_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(output_file, index=False)
        else:
            df_new.to_csv(output_file, index=False)

        print(f"Appended {len(all_results)} records to {output_file}")
    else:
        print("No data processed today.")

if __name__ == "__main__":
    run_all()
