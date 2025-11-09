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

# Change working dir to where config.yml is located
cd /app || exit 1

# Run Python scripts
run_python_script "$ROOT_DIR/update_etf_data.py"
run_python_script "$ROOT_DIR/option_metrics_new.py"
run_python_script "$ROOT_DIR/stockdata/db_price_update_r.py"
run_python_script "$ROOT_DIR/stockdata/sp500_update.py"

# Run R scripts (paths relative to current dir now)
run_r_script "r_scripts/update_corporate_action.R"
run_r_script "r_scripts/update_fundamentals.R"

# Update raw indicators into monthly
run_python_script "$ROOT_DIR/factor_model/price_factors.py"
run_python_script "$ROOT_DIR/factor_model/fundamental_factors.py"
run_python_script "$ROOT_DIR/factor_model/factor_ranks.py"

echo "All scripts finished running."
