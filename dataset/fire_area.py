import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ========== [1] 데이터 불러오기 ==========
df = pd.read_csv("gangwon_fire_data_with_direction.csv")

# ========== [2] 데이터 전처리 ==========
# datetime 형식 제거
df = df.drop(columns=["ndvi_date"], errors="ignore")

# object → numeric (숫자형으로 바뀌는 것만 유지)
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            continue  # 변환 불가하면 패스

# 결측치 제거
df = df.dropna(subset=['damage_area'])

# ========== [3] Feature 선택 (중요변수 수동 선택 또는 자동 중요도 기반) ==========
selected_features = [
    'ndvi_pre_fire_latest', 'rainfall_cumulative_30d', 'start_month',
    'fire_duration_hours', 'elevation', 'ws2m_std', 'wd2m_mean',
    'rainfall_cumulative_14d', 'forest_cover_5km_percent', 'latitude',
    'ws2m_1', 'ws10m_mean', 'dry_days_7d', 'ws2m_max', 'ps_min'
]

X = df[selected_features]
y = df['damage_area']

# ========== [4] 학습/테스트 데이터 분할 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========== [5] 모델 학습 ==========
model = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# ========== [6] 예측 및 평가 ==========
y_pred = model.predict(X_test)

mae = round(mean_absolute_error(y_test, y_pred), 4)
rmse = round(mean_squared_error(y_test, y_pred) ** 0.5, 4)
r2 = round(r2_score(y_test, y_pred), 4)

print(f"📊 피해면적 예측 결과:")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# ========== [7] 모델 저장 ==========
joblib.dump(model, "xgb_damage_area_model.joblib")
print("✅ 모델이 'xgb_damage_area_model.joblib'으로 저장되었습니다.")

# ========== [8] 중요도 시각화 ==========
importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by="Importance", ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df, color="chocolate")
plt.title("Top 15 Feature Importances for Damage Area Prediction")
plt.tight_layout()
plt.show()