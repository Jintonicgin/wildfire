import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

# ========== [1] 데이터 불러오기 ==========
df = pd.read_csv("gangwon_fire_data_with_direction.csv")

# ========== [2] 파생 변수 생성 ==========
# datetime 파싱
df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce')
df['end_datetime'] = pd.to_datetime(df['end_datetime'], errors='coerce')

# spread_speed 계산
df['duration_hours'] = (df['end_datetime'] - df['start_datetime']).dt.total_seconds() / 3600
df['spread_speed'] = df['damage_area'] / df['duration_hours']

# 무한대, NaN 제거
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# ========== [3] 결측치 제거 ==========
df = df.dropna(subset=[
    'damage_area', 'spread_direction_class_8', 'spread_speed'
])

# ========== [4] 인코더 및 feature 목록 불러오기 ==========
direction_features = joblib.load("direction_feature_list.joblib")
damage_features = joblib.load("damage_feature_list.joblib")
le_direction = joblib.load("direction_label_encoder.joblib")

# ========== [5] 모델 불러오기 ==========
clf_dir = joblib.load("direction_classifier_rf.joblib")
reg_speed = joblib.load("spread_speed_regressor_xgb.joblib")
reg_dmg = joblib.load("xgb_damage_area_model.joblib")

# ========== [6] X, y 준비 ==========
df['direction_encoded'] = le_direction.transform(df['spread_direction_class_8'])

X_dir_speed = df[direction_features]
X_dmg = df[damage_features]

y_dir = df['direction_encoded']
y_speed = df['spread_speed']
y_dmg = df['damage_area']

# ========== [7] 예측 ==========
y_pred_dir = clf_dir.predict(X_dir_speed)
y_pred_speed = reg_speed.predict(X_dir_speed)
y_pred_dmg = reg_dmg.predict(X_dmg)

# ========== [8] 방향 분류 결과 ==========
print("\n📌 [방향 예측 성능]")
print(classification_report(y_dir, y_pred_dir, target_names=le_direction.classes_, zero_division=0))

# 혼동 행렬 시각화
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_dir, y_pred_dir)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_direction.classes_,
            yticklabels=le_direction.classes_)
plt.title("📊 Spread Direction Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ========== [9] 속도 회귀 결과 ==========
mae_s = round(mean_absolute_error(y_speed, y_pred_speed), 4)
rmse_s = round(np.sqrt(mean_squared_error(y_speed, y_pred_speed)), 4)
r2_s = round(r2_score(y_speed, y_pred_speed), 4)

print("\n📌 [속도 예측 성능]")
print(f"MAE: {mae_s}")
print(f"RMSE: {rmse_s}")
print(f"R²: {r2_s}")

# 시각화
plt.figure(figsize=(6, 6))
plt.scatter(y_speed, y_pred_speed, alpha=0.3, color='green')
plt.plot([y_speed.min(), y_speed.max()], [y_speed.min(), y_speed.max()], 'k--')
plt.xlabel("Actual Spread Speed")
plt.ylabel("Predicted Spread Speed")
plt.title("🔥 Spread Speed Prediction")
plt.tight_layout()
plt.show()

# ========== [10] 피해면적 회귀 결과 ==========
mae_d = round(mean_absolute_error(y_dmg, y_pred_dmg), 4)
rmse_d = round(np.sqrt(mean_squared_error(y_dmg, y_pred_dmg)), 4)
r2_d = round(r2_score(y_dmg, y_pred_dmg), 4)

print("\n📌 [피해면적 예측 성능]")
print(f"MAE: {mae_d}")
print(f"RMSE: {rmse_d}")
print(f"R²: {r2_d}")

# 시각화
plt.figure(figsize=(6, 6))
plt.scatter(y_dmg, y_pred_dmg, alpha=0.3, color='chocolate')
plt.plot([y_dmg.min(), y_dmg.max()], [y_dmg.min(), y_dmg.max()], 'k--')
plt.xlabel("Actual Damage Area")
plt.ylabel("Predicted Damage Area")
plt.title("🔥 Damage Area Prediction")
plt.tight_layout()
plt.show()