import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load trained CatBoost model
# ----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("california_lgbm_model.pkl")
    return model

model = load_model()

# Streamlit UI
st.title("🏡 California Housing Price Predictor")
st.write("Move the sliders to provide housing features and predict **Median House Value**")

# Collect user inputs (matching training feature order)
MedInc = st.slider("Median Income (in $10,000)", 0.5, 15.0, 3.0)
HouseAge = st.slider("House Age (years)", 1, 52, 20)
AveRooms = st.slider("Average Rooms per Household", 1.0, 15.0, 5.0)
Population = st.slider("Population in Block", 3, 35000, 1500)
AveOccup = st.slider("Average Occupants per Household", 1.0, 6.0, 3.0)

# Special engineered feature Lat*Long
Latitude = st.slider("Latitude", 32.0, 42.0, 34.0)
Longitude = st.slider("Longitude", -125.0, -114.0, -118.0)
LatLong = Latitude * Longitude

# Maintain SAME order as training
input_df = pd.DataFrame([[
    MedInc, HouseAge, AveRooms,
    Population, AveOccup, LatLong
]], columns=["MedInc", "HouseAge", "AveRooms", "Population", "AveOccup", "Lat*Long"])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success(f"🏠 Estimated Median House Value: **${prediction * 100000:.2f}**")
