import json
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
examples = json.load(open("example_requests.json"))

st.title("Music Genre Classifier")
genre = st.selectbox("Load example values for genre", list(examples))
values = {
    f: st.number_input(f, value=float(v), format="%.6f", key=f"{genre}-{f}")
    for f, v in examples[genre].items()
}

if st.button("Predict"):
    try:
        res = requests.post(f"{API_URL}/predict", json=values, timeout=10).json()
        st.success(f"Predicted genre: {res['prediction']}")
        st.bar_chart(pd.Series(res["probabilities"]))
    except Exception as e:
        st.error(f"API error: {e}")
