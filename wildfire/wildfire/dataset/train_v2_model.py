import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
import joblib
import json
from model_definitions import EnsembleRegressor, EnsembleClassifier

SOURCE_DATA_PATH = "gangwon_fire_data_augmented_parallel.csv"
MODEL_OUTPUT_PATH = "area_regressor_model_v2.joblib"
COLUMNS_OUTPUT_PATH = "area_model_columns_v2.json"
SCALER_OUTPUT_PATH = "area_model_scaler_v2.joblib"

def main():
    print(f"1. 데이터 로딩: {SOURCE_DATA_PATH}")
    df = pd.read_csv(SOURCE_DATA_PATH)

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

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    print("2. K-Fold 교차 검증 시작 (5 folds)")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    rmse_xgb, r2_xgb = [], []
    rmse_rf, r2_rf = [], []
    rmse_gb, r2_gb = [], []
    rmse_ens, r2_ens = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
        print(f"\n📂 Fold {fold + 1}")
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse')
        rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        gb_model = GradientBoostingRegressor(random_state=42)

        xgb_model.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)
        gb_model.fit(X_train, y_train)

        y_val_orig = np.expm1(y_val)
        pred_xgb = np.expm1(xgb_model.predict(X_val))
        pred_rf = np.expm1(rf_model.predict(X_val))
        pred_gb = np.expm1(gb_model.predict(X_val))

        rmse_x = np.sqrt(mean_squared_error(y_val_orig, pred_xgb))
        r2_x = r2_score(y_val_orig, pred_xgb)
        print(f"    XGB       - RMSE: {rmse_x:.4f}, R²: {r2_x:.4f}")

        rmse_r = np.sqrt(mean_squared_error(y_val_orig, pred_rf))
        r2_r = r2_score(y_val_orig, pred_rf)
        print(f"    RandomForest - RMSE: {rmse_r:.4f}, R²: {r2_r:.4f}")

        rmse_g = np.sqrt(mean_squared_error(y_val_orig, pred_gb))
        r2_g = r2_score(y_val_orig, pred_gb)
        print(f"    GB        - RMSE: {rmse_g:.4f}, R²: {r2_g:.4f}")

        ensemble_pred = (pred_xgb + pred_rf + pred_gb) / 3
        rmse_e = np.sqrt(mean_squared_error(y_val_orig, ensemble_pred))
        r2_e = r2_score(y_val_orig, ensemble_pred)
        print(f"    Ensemble  - RMSE: {rmse_e:.4f}, R²: {r2_e:.4f}")

        rmse_xgb.append(rmse_x)
        r2_xgb.append(r2_x)
        rmse_rf.append(rmse_r)
        r2_rf.append(r2_r)
        rmse_gb.append(rmse_g)
        r2_gb.append(r2_g)
        rmse_ens.append(rmse_e)
        r2_ens.append(r2_e)

    print("\n📊 평균 성능 요약")
    print(f"  - XGB       평균 RMSE: {np.mean(rmse_xgb):.4f}, R²: {np.mean(r2_xgb):.4f}")
    print(f"  - RandomForest 평균 RMSE: {np.mean(rmse_rf):.4f}, R²: {np.mean(r2_rf):.4f}")
    print(f"  - GB        평균 RMSE: {np.mean(rmse_gb):.4f}, R²: {np.mean(r2_gb):.4f}")
    print(f"  - Ensemble  평균 RMSE: {np.mean(rmse_ens):.4f}, R²: {np.mean(r2_ens):.4f}")

    print("\n3. 전체 데이터로 최종 모델 재학습")
    final_xgb = xgb.XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse')
    final_rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    final_gb = GradientBoostingRegressor(random_state=42)

    final_xgb.fit(X_scaled, y)
    final_rf.fit(X_scaled, y)
    final_gb.fit(X_scaled, y)

    ensemble_model = EnsembleRegressor([final_xgb, final_rf, final_gb])

    joblib.dump(ensemble_model, MODEL_OUTPUT_PATH)
    joblib.dump(scaler, SCALER_OUTPUT_PATH)
    with open(COLUMNS_OUTPUT_PATH, 'w') as f:
        json.dump(features, f)

    print("✅ 최종 모델 및 스케일러 저장 완료")

if __name__ == "__main__":
    main()