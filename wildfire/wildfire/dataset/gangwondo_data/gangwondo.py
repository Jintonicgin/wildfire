import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import json

df = pd.read_csv("./gangwondo_data/Export_NDVI_Climate.csv")
df = df.drop(columns=["system:index", ".geo"], errors="ignore")

X = df.drop(columns=["NDVI"])
y = df["NDVI"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE: {rmse:.4f}")

importance = model.get_booster().get_score(importance_type='gain')
total_gain = sum(importance.values())
importance_pct = {k: round((v / total_gain) * 100, 2) for k, v in importance.items()}

with open("./machine_learning/importance.json", "w", encoding="utf-8") as f:
    json.dump(importance_pct, f, ensure_ascii=False, indent=2)