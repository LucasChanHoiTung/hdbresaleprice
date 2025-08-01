import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load('rfrmodel.pkl')

st.title("HDB Resale Price Predictor (2017–2025)")

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://image.shutterstock.com/image-photo/hdb-estate-bendemeer-singapore-canal-250nw-2467679863.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

towns = [
    'ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
    'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
    'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
    'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
    'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
    'TOA PAYOH', 'WOODLANDS', 'YISHUN'
]

flat_types = [
    '1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION'
]

# User input widgets
town = st.selectbox("Select Town", towns)
flat_type = st.selectbox("Select Flat Type", flat_types)
floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=20, max_value=300, value=70)
flat_age = st.number_input("Flat Age (years)", min_value=0, max_value=100, value=30)

# Predict button
if st.button("Predict Resale Price"):
    # Create DataFrame from input
    input_df = pd.DataFrame([[town, flat_type, floor_area_sqm, flat_age]],
                            columns=['town', 'flat_type', 'floor_area_sqm', 'flat_age'])

    # One-hot encode town and flat_type (drop_first to avoid dummy columns)
    input_df = pd.get_dummies(input_df, columns=['town', 'flat_type'], drop_first=True)

    # Align columns with model's expected features (add missing with 0)
    model_columns = model.feature_names_in_
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model_columns]

    # Predict resale price
    prediction = model.predict(input_df)[0]
    formatted_price = f"${prediction:,.0f}"

    st.success(f"Estimated Resale Price: {formatted_price}")
