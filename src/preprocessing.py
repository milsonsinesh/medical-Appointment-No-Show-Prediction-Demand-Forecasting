import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    # Target encoding
    df["no_show"] = df["no_show"].map({"No": 0, "Yes": 1})

    # Missing value handling
    df["age"].fillna(df["age"].median(), inplace=True)
    df["specialty"].fillna("Unknown", inplace=True)
    df["disability"].fillna(0, inplace=True)
    df["place"].fillna("Unknown", inplace=True)

    # Date features
    df["appointment_date"] = pd.to_datetime(df["appointment_date_continuous"])
    df["day"] = df["appointment_date"].dt.day
    df["month"] = df["appointment_date"].dt.month
    df["weekday"] = df["appointment_date"].dt.weekday
    df["is_weekend"] = df["weekday"].isin([5,6]).astype(int)

    # Binary flags
    df["under_12"] = (df["age"] < 12).astype(int)
    df["over_60"] = (df["age"] > 60).astype(int)

    # Encode categoricals
    encoders = {}
    cat_cols = ["gender", "specialty", "place", "appointment_shift"]

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders
