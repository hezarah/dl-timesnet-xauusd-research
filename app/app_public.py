"""
Public (Research-only) Edition — DL App (TimesNet, XAUUSD M5)

This app is intentionally released as a research demonstration.
Live trading, broker connectivity, proprietary datasets, and commercial
execution rules are excluded for IP protection.
"""

import streamlit as st

st.set_page_config(page_title="DL Research Demo – TimesNet (XAUUSD)", layout="wide")

st.title("Deep Learning Research Demo – TimesNet on XAUUSD (5-Minute)")
st.markdown(
    """
This is a **research-only public demo**.
- No MetaTrader connectivity
- No live trading
- No proprietary execution rules
- Dataset excluded (schema only)
"""
)

st.subheader("Input Data")
st.info("Upload your own CSV dataset matching the schema in `data/sample_schema.csv`.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

st.subheader("Inference (Research Stub)")
if uploaded is None:
    st.warning("Upload a CSV to run the demo (schema-only).")
else:
    st.success("File uploaded. (In public version, inference is a stub.)")
    st.write("In a private commercial edition, this step runs TimesNet inference and produces signals.")

st.subheader("Figures (from Research Report)")
st.info("See figures in the `figures/` folder for DL evaluation examples.")
