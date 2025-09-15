import streamlit as st
from datetime import date
from models.gamma_flip import compute_analysis

def run():
    st.title("Gamma Flip Line / Expected Ranges Analysis")

    # Input widget: date selector
    selected_date = st.date_input("Select Date for Analysis", value=date.today())

    # Run the analysis logic from models
    result = compute_analysis(selected_date)

    # Display results — you can expand this with charts or tables
    st.subheader("Analysis Result")
    st.write(result)

    # Example: You might add chart visualizations here later
    # st.line_chart(...)
