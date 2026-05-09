import streamlit as st
import pandas as pd
import joblib

model = joblib.load('weather_model.pkl')
st.title("🌦️ Weather Prediction System")

temp = st.slider(
    "Temperature",
    -20.0,50.0,25.0
)

pressure = st.number_input(
    "Pressure",
    value=1013.0
)

humidity = st.slider(
    "Humidity",
    0.0,
    100.0,
    50.0
)

wind = st.slider(
    "Wind Speed",
    0.0,
    30.0,
    5.0
)

cloudiness = st.slider(
    "Cloudiness",
    0.0,
    100.0,
    20.0
)

country = st.text_input(
    "Country",
    "Unknown"
)

input_data = pd.DataFrame([{

    'Temp': temp,

    'Pressure': pressure,

    'Humidity': humidity,

    'Wind_speed': wind,

    'Cloudiness': cloudiness,

    'Country': country
}])

if st.button("Predict Weather"):

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Weather: {prediction[0]}"
    )

