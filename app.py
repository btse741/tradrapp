import streamlit as st
from importlib import import_module
import asyncio
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Define pages dictionary as module path strings
pages = {
    "Home": "app_pages.home",
    "ETF Momentum": "app_pages.etf_momentum",
    "Earnings Position": "app_pages.earnings_positions",  # new page added here
}


st.sidebar.title("Navigation")
page_choice = st.sidebar.selectbox("Go to", list(pages.keys()))

# Dynamically import selected page module and run its `run()` function
page_module = import_module(pages[page_choice])
page_module.run()
