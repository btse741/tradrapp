import streamlit as st

# Page configuration
st.set_page_config(page_title="Portfolio Dashboard", layout="wide")

# Sidebar for page navigation only
page = st.sidebar.selectbox("Select a page", ["Overview", "Stock Analysis", "Risk Metrics"])

if page == "Overview":
    st.title("Portfolio Overview")
    # Inputs on the page itself
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")
    # Display sections
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Summary Table")
        st.write("Table will be shown here")
    with col2:
        st.subheader("Portfolio Chart")
        st.write("Chart will be shown here")

elif page == "Stock Analysis":
    st.title("Stock Analysis")
    # Inputs embedded in this page
    stock_symbol = st.text_input("Stock Symbol", "AAPL")
    analysis_date = st.date_input("Analysis Date")
    # Display stock analysis results
    st.write(f"Showing analysis for {stock_symbol} on {analysis_date}")
    st.write("Stock analysis results ...")

else:  # Risk Metrics page
    st.title("Risk Metrics")
    # Inputs specific to risk metrics page
    risk_level = st.slider("Risk level", 1, 10, 5)
    # Display risk metric results
    st.write(f"Showing risk metrics at level {risk_level}")
    st.write("Risk metrics results ...")
