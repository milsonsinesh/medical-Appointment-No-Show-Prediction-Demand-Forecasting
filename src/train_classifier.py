import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier
from preprocessing import preprocess_data

df = pd.read_csv("data/raw/medical_appointments.csv")
df, encoders = preprocess_data(df)

features = [col for col in df.columns if col not in ["no_show", "appointment_date"]]
X = df[features]
y = df["no_show"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = XGBClassifier(
    scale_pos_weight=2,
    eval_metric="logloss",
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

joblib.dump(model, "models/no_show_model.pkl")
joblib.dump(encoders, "models/encoders.pkl")
