import streamlit as st
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

no_show_model = joblib.load(os.path.join(BASE_DIR, "models/no_show_model.pkl"))
feature_names = joblib.load(os.path.join(BASE_DIR, "models/feature_names.pkl"))
forecast_model = joblib.load(os.path.join(BASE_DIR, "models/demand_forecast_model.pkl"))


st.title("🏥 Medical Appointment Analytics")

tab1, tab2 = st.tabs(["No-Show Prediction", "Demand Forecast"])


with tab1:
    st.subheader("No-Show Risk Prediction")

    age = st.number_input("Age", min_value=0, max_value=100, value=24)
    sms = st.selectbox("SMS Received", [0, 1])

    if st.button("Predict No-Show Risk"):
        # Create input with ALL features
        input_data = pd.DataFrame(
            [[0] * len(feature_names)],
            columns=feature_names
        )

        # Fill known inputs
        if "age" in input_data.columns:
            input_data.loc[0, "age"] = age

        if "sms_received" in input_data.columns:
            input_data.loc[0, "sms_received"] = sms

        # Predict
        risk = no_show_model.predict_proba(input_data)[0][1]

        st.success(f"📊 No-Show Risk: {risk:.2%}")

with tab2:
    st.subheader("📈 Appointment Demand Forecast")

    st.write(
        "Forecast future daily appointment demand using historical trends."
    )

    # User input
    forecast_days = st.slider(
        "Select number of days to forecast",
        min_value=7,
        max_value=60,
        value=30
    )

    if st.button("Generate Demand Forecast"):
        # Generate forecast
        forecast = forecast_model.forecast(forecast_days)

        # Convert to DataFrame
        forecast_df = forecast.reset_index(drop=True)
        forecast_df.name = "Predicted Appointments"

        st.success("Forecast generated successfully")

        # Plot forecast
        st.line_chart(forecast_df)

        # Optional: show data table
        with st.expander("View forecast data"):
            st.dataframe(forecast_df)
