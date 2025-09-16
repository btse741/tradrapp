#!/bin/bash

# Run the existing etf_momentum data update script
python update_etf_data.py

# Run your new option metrics script
python run_option_metrics.py
