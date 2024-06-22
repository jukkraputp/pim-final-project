import streamlit as st
import pandas as pd
import joblib

def predictor(pipeline, selected_features):
    st.title("Predictor")
    st.write("Enter new data to make a prediction:")
    best_model = st.checkbox("Use our best model")

    if "AQI" in selected_features or best_model:
        AQI = st.slider(
            "AQI (Air Quality Index, a measure of how polluted the air currently is or how polluted it is forecast to become)",
            value=0.0,
            min_value=0.0,
            max_value=600.0,
        )
    if "PM10" in selected_features or best_model:
        PM10 = st.slider(
            "PM10 (Concentration of particulate matter less than 10 micrometers in diameter (μg/m³))",
            value=0.0,
            min_value=0.0,
            max_value=400.0,
        )
    if "PM2_5" in selected_features or best_model:
        PM2_5 = st.slider(
            "PM2_5 (Concentration of particulate matter less than 2.5 micrometers in diameter (μg/m³))",
            value=0.0,
            min_value=0.0,
            max_value=300.0,
        )
    if "NO2" in selected_features or best_model:
        NO2 = st.slider(
            "NO2 (Concentration of nitrogen dioxide (ppb))",
            value=0.0,
            min_value=0.0,
            max_value=300.0,
        )
    if "SO2" in selected_features or best_model:
        SO2 = st.slider(
            "SO2 (Concentration of sulfur dioxide (ppb))",
            value=0.0,
            min_value=0.0,
            max_value=200.0,
        )
    if "O3" in selected_features or best_model:
        O3 = st.slider(
            "O3 (Concentration of ozone (ppb))",
            value=0.0,
            min_value=0.0,
            max_value=400.0,
        )
    if "Temperature" in selected_features or best_model:
        Temperature = st.slider(
            "Temperature (Temperature in degrees Celsius (°C))",
            value=0.0,
            min_value=0.0,
            max_value=60.0,
        )
    if "Humidity" in selected_features or best_model:
        Humidity = st.slider(
            "Humidity (Humidity percentage (%))",
            value=0.0,
            min_value=0.0,
            max_value=200.0,
        )
    if "WindSpeed" in selected_features or best_model:
        WindSpeed = st.slider(
            "WindSpeed (Wind speed in meters per second (m/s))",
            value=0.0,
            min_value=0.0,
            max_value=40.0,
        )

    if st.button("Predict"):
        params = {
            "AQI": [0],
            "PM10": [0],
            "PM2_5": [0],
            "NO2": [0],
            "SO2": [0],
            "O3": [0],
            "Temperature": [0],
            "Humidity": [0],
            "WindSpeed": [0],
        }
        if "AQI" in selected_features or best_model:
            params["AQI"] = [AQI]
        if "PM10" in selected_features or best_model:
            params["PM10"] = [PM10]
        if "PM2_5" in selected_features or best_model:
            params["PM2_5"] = [PM2_5]
        if "NO2" in selected_features or best_model:
            params["NO2"] = [NO2]
        if "SO2" in selected_features or best_model:
            params["SO2"] = [SO2]
        if "O3" in selected_features or best_model:
            params["O3"] = [O3]
        if "Temperature" in selected_features or best_model:
            params["Temperature"] = [Temperature]
        if "Humidity" in selected_features or best_model:
            params["Humidity"] = [Humidity]
        if "WindSpeed" in selected_features or best_model:
            params["WindSpeed"] = [WindSpeed]
        new_data = pd.DataFrame(params)
        if best_model:
            model = joblib.load("best_air_quality_health_impact_model.pkl")
            prediction = model.predict(new_data)
        else:
            prediction = pipeline.predict(new_data)
        mapper = {0: "Very High", 1: "High", 2: "Moderate", 3: "Low", 4: "Very Low"}
        st.write("Predicted Health Impact Class:", mapper[prediction[0]])
