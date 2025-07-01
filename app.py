import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('house_price_model.pkl')
top_features = joblib.load('top_features.pkl')

st.set_page_config(page_title="🏠 House Price Predictor", layout="wide")
st.title("🏠 House Price Prediction App")
st.markdown("Enter the house details below and click **Predict** to see the estimated sale price.")

def user_input_features():
    input_data = {}
    for feature in top_features:
        if "Area" in feature or "SF" in feature or "GrLivArea" in feature:
            input_data[feature] = st.slider(f"{feature}", 0, 5000, 1500)
        elif "Year" in feature or "Yr" in feature:
            input_data[feature] = st.slider(f"{feature}", 1900, 2025, 2000)
        elif "Qual" in feature or "Cond" in feature or "Overall" in feature:
            input_data[feature] = st.selectbox(f"{feature}", list(range(1, 11)))
        else:
            input_data[feature] = st.number_input(f"{feature}", value=0)
    return pd.DataFrame([input_data])

input_df = user_input_features()

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated House Sale Price: **${prediction:,.0f}**")
    st.markdown("### 🔍 Your Inputs")
    st.dataframe(input_df.T, use_container_width=True)

st.sidebar.markdown("## ℹ️ About")
st.sidebar.write("""
This app uses a machine learning regression model trained on the Ames Housing dataset to predict house sale prices.
It uses your input values for the top selected features.
""")
