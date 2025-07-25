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
MODEL_OUTPUT_PATH = "area_regressor_model_v3.joblib"
COLUMNS_OUTPUT_PATH = "area_model_columns_v3.json"
SCALER_OUTPUT_PATH = "area_model_scaler_v3.joblib"

# --- 메인 실행 로직 ---
def main():
    print(f"1. 증강된 데이터 로딩: {SOURCE_DATA_PATH}")
    try:
        df = pd.read_csv(SOURCE_DATA_PATH)
    except FileNotFoundError:
        print(f"오류: 증강된 데이터 파일 '{SOURCE_DATA_PATH}'을 찾을 수 없습니다.")
        return

    print("2. 피처 엔지니어링 (수정됨 - duration 관련 피처 제외)")

    # 모델에 사용할 피처 목록 정의 (duration 관련 피처 모두 제외)
    features = [
        'lat', 'lng',
        'T2M', 'RH2M', 'WS10M', 'WD10M', 'PRECTOTCORR',
        'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI'
    ]
    target = 'estimated_damage_area'

    print(f"  - 사용할 피처 개수: {len(features)}개")
    print(f"  - 예측 목표: {target}")

    df.dropna(subset=features + [target], inplace=True)
    df = df[(df[target] > 0) & (df[target] < df[target].quantile(0.99))]

    X = df[features]
    y = np.log1p(df[target])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("3. 모델 훈련 시작")

    # --- 앙상블 모델 정의 ---
    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse')
    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    gb_model = GradientBoostingRegressor(random_state=42)

    # --- 하이퍼파라미터 그리드 ---
    param_grid = {
        'n_estimators': [200, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [5, 7, 10],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8]
    }

    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, 
                               scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    print(f"최적 하이퍼파라미터: {grid_search.best_params_}")
    best_xgb = grid_search.best_estimator_

    # --- 단일 모델 예측 및 평가 ---
    y_pred = best_xgb.predict(X_test_scaled)
    y_pred_orig = np.expm1(y_pred)
    y_test_orig = np.expm1(y_test)

    print("4. 모델 성능 평가")
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    r2 = r2_score(y_test_orig, y_pred_orig)
    print(f"  - 최종 모델 RMSE: {rmse:.4f}")
    print(f"  - 최종 모델 R^2: {r2:.4f}")

    print("5. 훈련된 모델 및 스케일러 저장")
    joblib.dump(best_xgb, MODEL_OUTPUT_PATH)
    joblib.dump(scaler, SCALER_OUTPUT_PATH)
    with open(COLUMNS_OUTPUT_PATH, 'w') as f:
        json.dump(features, f)
    print(f"  - 모델, 스케일러, 피처 목록 저장 완료.")

    # --- 피처 중요도 시각화 ---
    feature_importances = pd.DataFrame(best_xgb.feature_importances_, index=features, columns=['importance'])
    feature_importances = feature_importances.sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=feature_importances.importance, y=feature_importances.index)
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance_v2.png')
    print("  - 피처 중요도 그래프 저장 완료: feature_importance_v2.png")

if __name__ == "__main__":
    main()