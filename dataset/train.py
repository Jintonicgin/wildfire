# train_models.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Load
df = pd.read_csv("gangwon_fire_data_with_direction.csv")

# Spread Speed 계산
df['duration_hours'] = (
    pd.to_datetime(df['end_datetime']) - pd.to_datetime(df['start_datetime'])
).dt.total_seconds() / 3600
df['spread_speed'] = df['damage_area'] / df['duration_hours']
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 결측치 제거
df = df.drop(columns=df.columns[df.isnull().mean() > 0.3])
df = df.drop(columns=['fire_id', 'fire_start_date', 'start_time', 'fire_end_date',
                      'end_time', 'start_datetime', 'end_datetime', 'address'], errors='ignore')
df.dropna(subset=['damage_area', 'spread_direction_class_8', 'spread_speed'], inplace=True)

# Label Encoding
df['spread_direction_class_8'] = df['spread_direction_class_8'].astype(str)
le = LabelEncoder()
df['direction_encoded'] = le.fit_transform(df['spread_direction_class_8'])

# Feature 선택
X = df.select_dtypes(include=[np.number]).drop(columns=['damage_area', 'spread_speed', 'direction_encoded'], errors='ignore')
y_dir = df['direction_encoded']
y_speed = df['spread_speed']
y_damage = df['damage_area']

# 중요 변수 선택용
xgb_temp = XGBRegressor(n_estimators=50, max_depth=4, random_state=42)
xgb_temp.fit(X, y_speed)
top_features = pd.Series(xgb_temp.feature_importances_, index=X.columns).nlargest(15).index.tolist()
X_top = X[top_features]

# [1] 확산 방향 모델
X_train_dir, X_test_dir, y_train_dir, y_test_dir = train_test_split(X_top, y_dir, test_size=0.2, random_state=42)
clf_dir = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf_dir.fit(X_train_dir, y_train_dir)

# [2] 확산 속도 모델
X_train_sp, X_test_sp, y_train_sp, y_test_sp = train_test_split(X_top, y_speed, test_size=0.2, random_state=42)
reg_sp = XGBRegressor(n_estimators=50, max_depth=4, random_state=42)
reg_sp.fit(X_train_sp, y_train_sp)

# [3] 피해 면적 모델 (별도 feature set 사용)
selected_damage_features = [
    'ndvi_pre_fire_latest', 'rainfall_cumulative_30d', 'start_month',
    'fire_duration_hours', 'elevation', 'ws2m_std', 'wd2m_mean',
    'rainfall_cumulative_14d', 'forest_cover_5km_percent', 'latitude',
    'ws2m_1', 'ws10m_mean', 'dry_days_7d', 'ws2m_max', 'ps_min'
]
X_dmg = df[selected_damage_features]
y_dmg = df['damage_area']
X_train_dmg, X_test_dmg, y_train_dmg, y_test_dmg = train_test_split(X_dmg, y_dmg, test_size=0.2, random_state=42)
reg_dmg = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42)
reg_dmg.fit(X_train_dmg, y_train_dmg)

# 모델 저장
joblib.dump(clf_dir, "direction_classifier_rf.joblib")
joblib.dump(reg_sp, "spread_speed_regressor_xgb.joblib")
joblib.dump(reg_dmg, "xgb_damage_area_model.joblib")
joblib.dump(le, "direction_label_encoder.joblib")
joblib.dump(top_features, "direction_feature_list.joblib")
joblib.dump(selected_damage_features, "damage_feature_list.joblib")

print("✅ 모든 모델 및 feature 리스트 저장 완료")