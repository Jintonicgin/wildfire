import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score

SOURCE_DATA_PATH = "final_merged_feature_engineered.csv"
MODEL_OUTPUT_PATH = "area_regressor_model_v3_tuned.joblib"
COLUMNS_OUTPUT_PATH = "area_model_columns_v3_tuned.json"
SCALER_OUTPUT_PATH = "area_model_scaler_v3_tuned.joblib"
PERFORMANCE_OUTPUT_PATH = "area_model_performance.json"

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, SOURCE_DATA_PATH)

    print(f"1. 데이터 로딩: {data_path}")
    df = pd.read_csv(data_path)
    df.columns = [col.lower() for col in df.columns]

    # --- 필수 피처 목록 (새로운 단축 이름 사용) ---
    essential_features = [
        'elevation_mean', 'elevation_std', 'slope_mean', 'slope_std', 'aspect_mode', 'ndvi_before',
        'fwi_0h', 'ffmc_0h', 'dmc_0h', 'dc_0h', 'isi_0h', 'bui_0h'
    ]
    existing_essential_features = [f for f in essential_features if f in df.columns]
    print(f"데이터셋에 존재하는 필수 피처: {len(existing_essential_features)}개")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cols_to_exclude = ['fire_area', 'fire_duration_hours']
    all_features = [c for c in numeric_cols if c not in cols_to_exclude]

    target = 'fire_area'
    df_filtered = df[df[target].notna()].copy()
    df_filtered = df_filtered[(df_filtered[target] > 0) & (df_filtered[target] < df_filtered[target].quantile(0.99))]

    final_features = sorted(list(set(all_features + existing_essential_features)))
    print(f"최종 학습 피처 수: {len(final_features)}")

    X_selected = df_filtered[final_features].fillna(0)
    y_selected = np.log1p(df_filtered[target])

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_selected)

    print("\n2. 하이퍼파라미터 튜닝 (GridSearchCV) 시작")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [20, 30],
        'min_samples_leaf': [1, 3],
        'max_features': ['sqrt', 'log2']
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=kf,
                               scoring='neg_root_mean_squared_error', verbose=1)
    grid_search.fit(X_scaled, y_selected)

    print("\n3. 최적 모델 성능 평가")
    best_model = grid_search.best_estimator_
    best_rmse = -grid_search.best_score_
    y_pred = best_model.predict(X_scaled)
    r2 = r2_score(y_selected, y_pred)
    print(f"   - 최적 하이퍼파라미터: {grid_search.best_params_}")
    print(f"   - 교차 검증 최적 RMSE: {best_rmse:.4f}")
    print(f"   - 전체 데이터에 대한 R² 점수: {r2:.4f}")

    performance_metrics = {
        'model_type': 'RandomForestRegressor',
        'task': 'Area Prediction',
        'best_hyperparameters': grid_search.best_params_,
        'cv_rmse': best_rmse,
        'r2_score': r2
    }
    with open(os.path.join(script_dir, PERFORMANCE_OUTPUT_PATH), 'w') as f:
        json.dump(performance_metrics, f, indent=4)
    print(f"\n   ✅ 성능 지표가 '{PERFORMANCE_OUTPUT_PATH}' 파일에 저장되었습니다.")

    print("\n4. 최종 모델 및 관련 파일 저장")
    joblib.dump(best_model, os.path.join(script_dir, MODEL_OUTPUT_PATH))
    joblib.dump(scaler, os.path.join(script_dir, SCALER_OUTPUT_PATH))
    with open(os.path.join(script_dir, COLUMNS_OUTPUT_PATH), 'w') as f:
        json.dump(final_features, f, indent=4)
    print("\n최종 개선된 모델, 스케일러, 피처 목록 저장 완료")

if __name__ == "__main__":
    main()