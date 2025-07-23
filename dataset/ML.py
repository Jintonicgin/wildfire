import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1. Load Data
df = pd.read_csv("gangwon_fire_data_with_direction.csv")

# 2. Spread Duration & Speed
df['duration_hours'] = (
    pd.to_datetime(df['end_datetime']) - pd.to_datetime(df['start_datetime'])
).dt.total_seconds() / 3600
df['spread_speed'] = df['damage_area'] / df['duration_hours']
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 3. Drop Columns with >30% NaN
df = df.drop(columns=df.columns[df.isnull().mean() > 0.3])

# 4. Drop unnecessary string columns
drop_cols = ['fire_id', 'fire_start_date', 'start_time', 'fire_end_date',
             'end_time', 'start_datetime', 'end_datetime', 'address']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# 5. Label encode direction
df = df[df['spread_direction_class_8'].notna()].copy()
df['spread_direction_class_8'] = df['spread_direction_class_8'].astype(str)
label_encoder = LabelEncoder()
df['direction_class_encoded'] = label_encoder.fit_transform(df['spread_direction_class_8'])

# 6. Final NaN drop
df.dropna(inplace=True)

# 7. Feature / Target
features = df.drop(columns=['damage_area', 'spread_speed', 'spread_direction_label',
                            'spread_direction_class_8', 'direction_class_encoded'])
X = features.select_dtypes(include=['number', 'bool'])
y_direction = df['direction_class_encoded']
y_speed = df['spread_speed']

# 8. Feature Importance (XGBoost for speed)
xgb_temp = XGBRegressor(n_estimators=50, max_depth=4, random_state=42)
xgb_temp.fit(X, y_speed)
top_features = pd.Series(xgb_temp.feature_importances_, index=X.columns).nlargest(15).index.tolist()
X_top = X[top_features]

# 9. Direction Classification (RandomForest + balanced)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_top, y_direction, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf.fit(X_train_c, y_train_c)
y_pred_c = clf.predict(X_test_c)

print("\n[Spread Direction Classification Report]")
print(classification_report(y_test_c, y_pred_c, target_names=label_encoder.classes_, zero_division=0))

# 10. Confusion Matrix Plot
cm = confusion_matrix(y_test_c, y_pred_c)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix (Spread Direction)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# 11. Feature Importance Plot
importances = clf.feature_importances_
importance_df = pd.DataFrame({"Feature": X_top.columns, "Importance": importances}).sort_values(by="Importance")

plt.figure(figsize=(8, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="Greens_r")
plt.title("Top 15 Feature Importances (Direction Classifier)")
plt.tight_layout()
plt.show()

# 12. Spread Speed Regression (XGB)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_top, y_speed, test_size=0.2, random_state=42)
reg_model = XGBRegressor(n_estimators=50, max_depth=4, random_state=42)
reg_model.fit(X_train_r, y_train_r)
y_pred_r = reg_model.predict(X_test_r)

print("\n[Spread Speed Regression Metrics]")
print({
    "MAE": round(mean_absolute_error(y_test_r, y_pred_r), 4),
    "RMSE": round(np.sqrt(mean_squared_error(y_test_r, y_pred_r)), 4),
    "R2": round(r2_score(y_test_r, y_pred_r), 4)
})

# 13. Save Models
joblib.dump(clf, "direction_classifier_rf.joblib")
joblib.dump(reg_model, "spread_speed_regressor_xgb.joblib")