import os
import yaml
import streamlit as st

from models.earnings_positions.compute import compute_recommendation
from models.earnings_positions.telegram_notify import send_telegram_message

# Cache config loading for performance and reusability
@st.cache_resource
def load_config():
    # Define project root relative to this file's location (2-levels up)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(project_root, 'config.yml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()
BOT_TOKEN = config.get('telegram', {}).get('bot_token')
CHAT_ID = config.get('telegram', {}).get('chat_id')

def run():
    st.title("Preannouncement Imp vol Opportunity Scan")

    # Shrink input width for better UI
    st.markdown("""
    <style>
    input#stock_input {
        max-width: 150px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    stock_input = st.text_input("Enter Stock Symbol:", max_chars=10, key="stock_input")

    if st.button("Submit"):
        if not stock_input.strip():
            st.error("No stock symbol provided.")
        else:
            with st.spinner("Loading..."):
                try:
                    result = compute_recommendation(stock_input.strip())

                    if isinstance(result, str) and result.startswith("Error"):
                        st.error(result)
                        return

                    avg_volume_bool = result['avg_volume_bool']
                    avg_volume = result.get('avg_volume', None)
                    iv30_rv30_bool = result['iv30_rv30_bool']
                    iv30_rv30 = result.get('iv30_rv30', None)
                    ts_slope_bool = result['ts_slope_bool']
                    ts_slope = result.get('ts_slope_0_45', None)

                    expected_move_straddle = result.get('expected_move_straddle', 'N/A')
                    straddle_expiry = result.get('straddle_expiry', 'N/A')
                    straddle_strike = result.get('straddle_strike', 'N/A')

                    calendar_short_expiry = result.get('calendar_short_expiry', 'N/A')
                    calendar_long_expiry = result.get('calendar_long_expiry', 'N/A')
                    calendar_strike = result.get('calendar_strike', 'N/A')
                    expected_move_calendar = result.get('expected_move_calendar', 'N/A')

                    # Determine recommendation and color
                    if avg_volume_bool and iv30_rv30_bool and ts_slope_bool:
                        title = "Recommended"
                        title_color = "#006600"
                    elif ts_slope_bool and ((avg_volume_bool and not iv30_rv30_bool) or (iv30_rv30_bool and not avg_volume_bool)):
                        title = "Consider"
                        title_color = "#ff9900"
                    else:
                        title = "Avoid"
                        title_color = "#800000"

                    st.markdown(f"<h1 style='color: {title_color};'>{title}</h1>", unsafe_allow_html=True)

                    # Display criteria in columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("### Average Volume")
                        st.write(f"Status: {'PASS' if avg_volume_bool else 'FAIL'}")
                        st.write(f"Value: {avg_volume:.0f}" if avg_volume is not None else "Value: N/A")
                        st.write("Threshold: ≥ 1,500,000")

                    with col2:
                        st.markdown("### IV30 / RV30 Ratio")
                        st.write(f"Status: {'PASS' if iv30_rv30_bool else 'FAIL'}")
                        st.write(f"Value: {iv30_rv30:.2f}" if iv30_rv30 is not None else "Value: N/A")
                        st.write("Threshold: ≥ 1.25")

                    with col3:
                        st.markdown("### Term Slope 0-45")
                        st.write(f"Status: {'PASS' if ts_slope_bool else 'FAIL'}")
                        st.write(f"Value: {ts_slope:.5f}" if ts_slope is not None else "Value: N/A")
                        st.write("Threshold: ≤ -0.00406")

                    st.markdown("---")

                    # Straddle and Calendar Spread info side by side
                    col_s, col_c = st.columns(2)
                    with col_s:
                        st.markdown("## Straddle Trade")
                        st.write(f"**Strike:** {straddle_strike}")
                        st.write(f"**Expiry:** {straddle_expiry}")
                        st.write(f"**Expected Move:** {expected_move_straddle}")

                    with col_c:
                        st.markdown("## Calendar Spread Trade")
                        st.write(f"**Common Strike:** {calendar_strike}")
                        st.write(f"**Short Leg Expiry:** {calendar_short_expiry}")
                        st.write(f"**Long Leg Expiry:** {calendar_long_expiry}")
                        st.write(f"**Expected Move / Edge:** {expected_move_calendar}")

                    # Send Telegram if recommended
                    if title == "Recommended":
                        if BOT_TOKEN and CHAT_ID:
                            message = (
                                f"📢 Earnings Position Recommendation\n"
                                f"Straddle:\nExpiry: {straddle_expiry}\nStrike: {straddle_strike}\nExpected Move: {expected_move_straddle}\n\n"
                                f"Calendar Spread:\nShort Leg Expiry: {calendar_short_expiry}\nLong Leg Expiry: {calendar_long_expiry}\nStrike: {calendar_strike}\nExpected Move / Edge: {expected_move_calendar}"
                            )
                            send_telegram_message(BOT_TOKEN, CHAT_ID, message)
                        else:
                            st.warning("Telegram bot token or chat ID missing. Cannot send notification.")

                except Exception as e:
                    st.error(f"Error occurred: {e}")
