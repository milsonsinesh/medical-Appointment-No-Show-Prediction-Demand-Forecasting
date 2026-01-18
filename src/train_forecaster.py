import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import os

df = pd.read_csv("data/raw/medical_appointment_data.csv")
df["appointment_date"] = pd.to_datetime(df["appointment_date_continuous"])

daily_demand = df.groupby("appointment_date").size()

model = SARIMAX(
    daily_demand,
    order=(1,1,1),
    seasonal_order=(1,1,1,7)
)

model_fit = model.fit(disp=False)

os.makedirs("models", exist_ok=True)
joblib.dump(model_fit, "models/demand_forecast_model.pkl")

print("Demand forecast model saved")
