import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

# --- 설정 ---
SOURCE_DATA_PATH = "gangwon_fire_data_augmented_parallel.csv"
MODEL_OUTPUT_PATH = "area_regressor_model_v2.joblib"
COLUMNS_OUTPUT_PATH = "area_model_columns_v2.json"
SCALER_OUTPUT_PATH = "area_model_scaler_v2.joblib"

# --- 앙상블 클래스 정의 (전역 범위로 옮김) ---
class EnsembleRegressor:
    """여러 개의 회귀 모델을 평균하여 예측하는 간단한 앙상블 클래스."""
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        preds = [model.predict(X) for model in self.models]
        return np.mean(np.column_stack(preds), axis=1)

# --- 메인 실행 로직 ---
def main():
    print(f"1. 증강된 데이터 로딩: {SOURCE_DATA_PATH}")
    df = pd.read_csv(SOURCE_DATA_PATH)

    print("2. 피처 엔지니어링 (duration 관련 피처 포함)")
    features = [
        'lat', 'lng',
        'duration_hours', 'total_duration_hours',
        'T2M', 'RH2M', 'WS10M', 'WD10M', 'PRECTOTCORR',
        'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI'
    ]
    target = 'estimated_damage_area'

    df.dropna(subset=features + [target], inplace=True)
    df = df[(df[target] > 0) & (df[target] < df[target].quantile(0.99))]

    X = df[features]
    y = np.log1p(df[target])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("3. 모델 훈련 시작")
    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse')
    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    gb_model = GradientBoostingRegressor(random_state=42)

    param_grid = {
        'n_estimators': [200, 500],
        'learning_rate': [0.01, 0.1],
        'max_depth': [5, 7],
        'subsample': [0.7],
        'colsample_bytree': [0.8]
    }

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        verbose=2,
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    best_xgb = grid_search.best_estimator_

    rf_model.fit(X_train_scaled, y_train)
    gb_model.fit(X_train_scaled, y_train)

    y_pred_xgb = best_xgb.predict(X_test_scaled)
    y_pred_rf = rf_model.predict(X_test_scaled)
    y_pred_gb = gb_model.predict(X_test_scaled)

    y_pred_xgb_orig = np.expm1(y_pred_xgb)
    y_pred_rf_orig = np.expm1(y_pred_rf)
    y_pred_gb_orig = np.expm1(y_pred_gb)
    y_test_orig = np.expm1(y_test)

    ensemble_pred_orig = (y_pred_xgb_orig + y_pred_rf_orig + y_pred_gb_orig) / 3

    print("4. 모델 성능 평가")
    print(f"  - XGB RMSE: {np.sqrt(mean_squared_error(y_test_orig, y_pred_xgb_orig)):.4f}, R^2: {r2_score(y_test_orig, y_pred_xgb_orig):.4f}")
    print(f"  - RF   RMSE: {np.sqrt(mean_squared_error(y_test_orig, y_pred_rf_orig)):.4f}, R^2: {r2_score(y_test_orig, y_pred_rf_orig):.4f}")
    print(f"  - GB   RMSE: {np.sqrt(mean_squared_error(y_test_orig, y_pred_gb_orig)):.4f}, R^2: {r2_score(y_test_orig, y_pred_gb_orig):.4f}")
    print(f"  - Ensemble RMSE: {np.sqrt(mean_squared_error(y_test_orig, ensemble_pred_orig)):.4f}, R^2: {r2_score(y_test_orig, ensemble_pred_orig):.4f}")

    print("5. 모델 및 스케일러 저장")
    ensemble_model = EnsembleRegressor([best_xgb, rf_model, gb_model])
    joblib.dump(ensemble_model, MODEL_OUTPUT_PATH)
    joblib.dump(scaler, SCALER_OUTPUT_PATH)
    with open(COLUMNS_OUTPUT_PATH, 'w') as f:
        json.dump(features, f)

    print("6. 시각화")
    plt.figure(figsize=(10, 10))
    plt.scatter(y_test_orig, ensemble_pred_orig, alpha=0.3)
    plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], '--r')
    plt.xlabel('Actual')
    plt.ylabel('Predicted (Ensemble)')
    plt.title('Actual vs Predicted Damage Area (Ensemble)')
    plt.savefig('actual_vs_predicted_damage_area.png')

    feature_importances = pd.DataFrame(best_xgb.feature_importances_, index=features, columns=['importance'])
    feature_importances = feature_importances.sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=feature_importances.importance, y=feature_importances.index)
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance_v2.png')

if __name__ == "__main__":
    main()
