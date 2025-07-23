import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
import shap

# --- 1. Load Data ---
df = pd.read_csv("gangwon_fire_data_with_direction.csv")

df['duration_hours'] = (
    pd.to_datetime(df['end_datetime']) - pd.to_datetime(df['start_datetime'])
).dt.total_seconds() / 3600
df['spread_speed'] = df['damage_area'] / df['duration_hours']
df.replace([np.inf, -np.inf], np.nan, inplace=True)

df = df.drop(columns=df.columns[df.isnull().mean() > 0.3])
df = df.drop(columns=['fire_id', 'fire_start_date', 'start_time', 'fire_end_date',
                      'end_time', 'start_datetime', 'end_datetime', 'address'], errors='ignore')
df.dropna(subset=['damage_area', 'spread_direction_class_8', 'spread_speed'], inplace=True)

df['spread_direction_class_8'] = df['spread_direction_class_8'].astype(str)
le = LabelEncoder()
df['direction_encoded'] = le.fit_transform(df['spread_direction_class_8'])

X = df.select_dtypes(include=[np.number]).drop(columns=['damage_area', 'spread_speed', 'direction_encoded'], errors='ignore')
y_speed = df['spread_speed']
y_dir = df['direction_encoded']

# --- 2. Train-Test Split ---
X_train_sp, X_test_sp, y_train_sp, y_test_sp = train_test_split(X, y_speed, test_size=0.2, random_state=42)
X_train_dir, X_test_dir, y_train_dir, y_test_dir = train_test_split(X, y_dir, test_size=0.2, random_state=42)

# --- 3. Train Models ---
reg_sp = XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
reg_sp.fit(X_train_sp, y_train_sp)

clf_dir = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf_dir.fit(X_train_dir, y_train_dir)

# --- 4. Basic Feature Importance (XGBoost 기준) ---
feat_imp = pd.Series(reg_sp.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n🎯 [기본 Feature Importance - XGBoost]")
print(feat_imp.head(10))

# --- 5. Permutation Importance ---
print("\n🎯 [Permutation Importance]")
perm_result = permutation_importance(reg_sp, X_test_sp, y_test_sp, n_repeats=10, random_state=42)
sorted_idx = perm_result.importances_mean.argsort()[::-1]
for idx in sorted_idx[:10]:
    print(f"{X.columns[idx]}: {perm_result.importances_mean[idx]:.4f}")

# --- 6. SHAP 값 분석 ---
print("\n🎯 [SHAP 분석]")
explainer = shap.Explainer(reg_sp)
shap_values = explainer(X_test_sp)
shap.summary_plot(shap_values, X_test_sp, show=False)  # show=False: interactive 환경 아니면 error 방지
print("✅ SHAP 분석 완료")

# --- 7. Top-N 피처로 다시 모델 학습하여 비교 ---
top5 = feat_imp.head(5).index
top10 = feat_imp.head(10).index

for topN, features in [('Top 5', top5), ('Top 10', top10)]:
    X_train_N, X_test_N = X_train_sp[features], X_test_sp[features]
    model = XGBRegressor(n_estimators=100, random_state=42).fit(X_train_N, y_train_sp)
    preds = model.predict(X_test_N)
    rmse = np.sqrt(mean_squared_error(y_test_sp, preds))
    r2 = r2_score(y_test_sp, preds)
    print(f"\n🎯 [{topN} 피처 성능]")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")

print("\n✅ 모든 중요도 분석 및 비교 완료")