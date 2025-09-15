#!/bin/bash

# Run the existing etf_momentum data update script
python models/etf_momentum/data_update.py

# Run your new option metrics script
python run_option_metrics.py
