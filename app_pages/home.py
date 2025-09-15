import streamlit as st
from datetime import datetime, timedelta
from streamlit_calendar import calendar
from models.calendar_data.earnings_calendar import get_upcoming_earnings_finnhub

@st.cache_data(ttl=3600)
def cached_earnings(days_ahead=7):
    return get_upcoming_earnings_finnhub(days_ahead)

def format_earnings_events(df):
    events = []
    for _, row in df.iterrows():
        date_str = row.get('date', None) or row.get('date_time', None)
        if not date_str:
            continue
        try:
            date_obj = datetime.strptime(date_str.split(" ")[0], "%Y-%m-%d")
        except:
            continue
        symbol = row.get('symbol', 'Earnings')
        events.append({
            "title": f"Earnings: {symbol}",
            "start": date_obj.strftime("%Y-%m-%dT09:00:00"),
            "end": date_obj.strftime("%Y-%m-%dT10:00:00"),
        })
    return events

def run():
    st.title("Upcoming Earnings")

    earnings_df = cached_earnings(7)

    if earnings_df.empty:
        st.info("No earnings events in the next 7 days.")
    else:
        earnings_events = format_earnings_events(earnings_df)
        calendar(
            events=earnings_events,
            options={
                "initialView": "listWeek",
                "headerToolbar": {
                    "left": "prev,next today",
                    "center": "title",
                    "right": "listWeek,dayGridMonth"
                }
            },
            # height=600
        )
