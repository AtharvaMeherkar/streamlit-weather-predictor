import streamlit as st
import pandas as pd
from sklearn.linear_model import Ridge
import numpy as np

# --- Caching the model ---
# This decorator ensures the model is trained only once and cached for performance.
@st.cache_data
def train_and_get_model():
    """
    Loads data, cleans it, engineers features, and trains the final Ridge model.
    Returns the trained model and the last row of data for feature calculation.
    """
    weather = pd.read_csv("weather.csv", index_col="DATE")

    # Data Cleaning and Preparation
    core_weather = weather[["PRCP", "SNOW", "SNWD", "TMAX", "TMIN"]].copy()
    core_weather.columns = ["precip", "snow", "snow_depth", "temp_max", "temp_min"]
    core_weather["snow"] = core_weather["snow"].fillna(0)
    core_weather["snow_depth"] = core_weather["snow_depth"].fillna(0)
    core_weather = core_weather.ffill()
    core_weather.index = pd.to_datetime(core_weather.index)

    # Feature Engineering
    core_weather["month_max"] = core_weather["temp_max"].rolling(30).mean()
    core_weather["month_day_max"] = core_weather["month_max"] / core_weather["temp_max"]
    core_weather["max_min_ratio"] = core_weather["temp_max"] / core_weather["temp_min"]
    core_weather = core_weather.replace([np.inf, -np.inf], 0).ffill()

    # Target Definition
    core_weather["target"] = core_weather.shift(-1)["temp_max"]
    core_weather = core_weather.iloc[30:-1,:].copy()

    # Model Training
    reg = Ridge(alpha=.1)
    predictors = ["precip", "snow", "snow_depth", "temp_max", "temp_min", "month_max", "month_day_max", "max_min_ratio"]
    reg.fit(core_weather[predictors], core_weather["target"])
    
    return reg, core_weather.iloc[-1]

# --- Streamlit App UI ---

# Set page title and icon
st.set_page_config(page_title="AI Weather Predictor", page_icon="🌦️")

# App Title
st.title("AI Weather Predictor 🌦️")
st.write("Enter today's weather conditions to predict tomorrow's maximum temperature.")

# Load the trained model and latest data
model, latest_data = train_and_get_model()

# Create input widgets in the sidebar
st.sidebar.header("Today's Weather Inputs")

temp_max = st.sidebar.number_input("Today's Max Temperature (°F)", min_value=-20.0, max_value=120.0, value=75.0, step=1.0)
temp_min = st.sidebar.number_input("Today's Min Temperature (°F)", min_value=-30.0, max_value=100.0, value=55.0, step=1.0)
precip = st.sidebar.slider("Precipitation (inches)", 0.0, 5.0, 0.0, 0.01)
snow = st.sidebar.slider("Snowfall (inches)", 0.0, 24.0, 0.0, 0.1)
snow_depth = st.sidebar.slider("Current Snow Depth (inches)", 0.0, 48.0, 0.0, 0.1)

# Predict button
if st.sidebar.button("Predict", type="primary"):
    # Prepare input for prediction
    today_data = {
        "precip": precip,
        "snow": snow,
        "snow_depth": snow_depth,
        "temp_max": temp_max,
        "temp_min": temp_min,
        "month_max": latest_data["month_max"], # Use recent averages for these features
        "month_day_max": latest_data["month_max"] / temp_max if temp_max != 0 else 0,
        "max_min_ratio": temp_max / temp_min if temp_min != 0 else 0
    }
    
    # Convert to DataFrame and handle potential infinity values
    today_df = pd.DataFrame([today_data])
    today_df = today_df.replace([np.inf, -np.inf], 0)
    
    # Define predictors
    predictors = ["precip", "snow", "snow_depth", "temp_max", "temp_min", "month_max", "month_day_max", "max_min_ratio"]

    # Make the prediction
    predicted_temp = model.predict(today_df[predictors])[0]
    
    # Display the output
    st.header("Prediction Result")
    st.metric(label="Predicted Max Temperature for Tomorrow", value=f"{predicted_temp:.1f} °F")

    # Add a simple chart for context
    st.write("### Temperature Context")
    chart_data = pd.DataFrame({
        "Temperature": [temp_max, predicted_temp],
    }, index=["Today", "Tomorrow (Predicted)"])
    st.bar_chart(chart_data)

else:
    st.info("Enter today's weather conditions in the sidebar and click 'Predict'.")