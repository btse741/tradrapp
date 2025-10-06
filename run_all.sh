#!/bin/bash

ROOT_DIR=$(pwd)

run_python_script() {
  script_path=$1
  echo "Running Python script ${script_path} ..."
  python3 "$script_path"
  if [ $? -ne 0 ]; then
    echo "Warning: Python script ${script_path} failed but continuing with next scripts." >&2
  else
    echo "Python script ${script_path} completed successfully."
  fi
}

run_r_script() {
  script_path=$1
  echo "Running R script ${script_path} ..."
  Rscript "$script_path"
  if [ $? -ne 0 ]; then
    echo "Warning: R script ${script_path} failed but continuing with next scripts." >&2
  else
    echo "R script ${script_path} completed successfully."
  fi
}

# Run Python scripts
run_python_script "$ROOT_DIR/update_etf_data.py"
run_python_script "$ROOT_DIR/option_metrics_new.py"
run_python_script "$ROOT_DIR/stockdata/db_price_update_r.py"
run_python_script "$ROOT_DIR/stockdata/sp500_update.py"

# Run R script for corporate action update
run_r_script "$ROOT_DIR/r_scripts/update_corporate_action.R"
# Run R script for fundamentals
run_r_script "$ROOT_DIR/r_scripts/update_fundamentals.R"

echo "All scripts finished running."
