import streamlit as st
from importlib import import_module
import asyncio
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Define pages dictionary as module path strings
pages = {
    "Home": "app_pages.home",
    "ETF Rotations": "app_pages.etf_momentum",
    "Preannouncement Options Strategy": "app_pages.earnings_positions", 
     "Intraday Triggers": "app_pages.gamma_flip_page" # new page added here
}

st.sidebar.title("Models")
# Use radio buttons to show all pages in sidebar for quick access
page_choice = st.sidebar.radio("Go to", list(pages.keys()))

# Dynamically import selected page module and run its `run()` function
page_module = import_module(pages[page_choice])
page_module.run()
