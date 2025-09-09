import pandas as pd
import numpy as np
import joblib
import json
import os
import math
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from pathlib import Path

# --- 파일 경로 설정 ---
SOURCE_DATA_PATH = "final_merged_feature_engineered.csv"

SPEED_MODEL_PATH = "speed_classifier_model_v2_tuned_cw.joblib"
SPEED_COLUMNS_PATH = "speed_model_columns_v2_tuned_cw.json"
SPEED_SCALER_PATH = "speed_scaler_v2_tuned_cw.joblib"
SPEED_PERFORMANCE_PATH = "speed_model_performance.json"

DIRECTION_MODEL_PATH = "direction_classifier_model_v2_tuned_cw.joblib"
DIRECTION_COLUMNS_PATH = "direction_model_columns_v2_tuned_cw.json"
DIRECTION_SCALER_PATH = "direction_scaler_v2_tuned_cw.joblib"
DIRECTION_PERFORMANCE_PATH = "direction_model_performance.json"

# --- Helper Functions ---
def classify_speed(speed: float, thresholds=(0.014, 0.11)) -> int:
    low, high = thresholds
    if speed <= low: return 0
    if speed <= high: return 1
    return 2

def convert_degree_to_direction(deg: float) -> int:
    if deg is None or pd.isna(deg) or deg == -999: return 0
    return int(math.floor(((float(deg) + 22.5) % 360) / 45))

def load_improved_features():
    """Load improved features from correlation analysis"""
    try:
        improved_dir = Path("improved_correlation_analysis")
        with open(improved_dir / "improved_direction_features.json", 'r') as f:
            direction_features = [col.lower() for col in json.load(f)]
        with open(improved_dir / "improved_speed_features.json", 'r') as f:
            speed_features = [col.lower() for col in json.load(f)]
        print(f"✅ Loaded improved features - Direction: {len(direction_features)}, Speed: {len(speed_features)}")
        return direction_features, speed_features
    except FileNotFoundError:
        print("⚠️ Improved features not found, using fallback method...")
        return None, None

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, SOURCE_DATA_PATH)

    print(f"1. 데이터 로딩: {data_path}")
    df = pd.read_csv(data_path)
    df.columns = [col.lower() for col in df.columns]

    df.dropna(subset=["fire_area", "fire_duration_hours", "wd10m_0h"], inplace=True)
    df = df[df["fire_duration_hours"] > 0].copy()
    df["spread_speed_class"] = (df["fire_area"] / df["fire_duration_hours"]).apply(classify_speed)
    df["spread_direction_class"] = df["wd10m_0h"].apply(convert_degree_to_direction)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    print("Loading improved features from correlation analysis...")
    improved_direction_features, improved_speed_features = load_improved_features()

    if improved_direction_features is None or improved_speed_features is None:
        print("Using fallback feature selection...")
        cols_to_exclude_base = ['fire_area', 'fire_duration_hours', 'spread_speed', 'spread_speed_class', 'spread_direction_class']
        all_features = [c for c in numeric_cols if c not in cols_to_exclude_base]
        leaky_patterns = ['_end', 'endday', 'endmonth', 'endyear', 'start_latitude', 'start_longitude']
        features_for_retraining = [f for f in all_features if not any(p in f for p in leaky_patterns)]

        improved_direction_features = [f for f in features_for_retraining if not f.startswith('wd')]
        improved_speed_features = [f for f in features_for_retraining if 'duration' not in f]
    else:
        improved_direction_features = [f for f in improved_direction_features if f in numeric_cols]
        improved_speed_features = [f for f in improved_speed_features if f in numeric_cols]

    # --- 데이터 유출 방지: 방향 모델 피처에서 과거 풍향(wd..._past) 관련 피처 모두 제거 ---
    original_dir_feature_count = len(improved_direction_features)
    improved_direction_features = [f for f in improved_direction_features if not (f.startswith('wd') and '_past' in f)]
    print(f"데이터 유출 방지를 위해 방향 모델에서 과거 풍향(wd..._past) 피처 {original_dir_feature_count - len(improved_direction_features)}개를 제거했습니다.")

    # --- 필수 피처 강제 포함 (새로운 단축 이름 사용) ---
    essential_features = [
        'elevation_mean', 'elevation_std', 'slope_mean', 'slope_std', 'aspect_mode', 'ndvi_before',
        'fwi_0h', 'ffmc_0h', 'dmc_0h', 'dc_0h', 'isi_0h', 'bui_0h'
    ]
    for f in essential_features:
        if f in df.columns and f not in improved_direction_features:
            improved_direction_features.append(f)
        if f in df.columns and f not in improved_speed_features:
            improved_speed_features.append(f)

    improved_direction_features = sorted(list(set(improved_direction_features)))
    improved_speed_features = sorted(list(set(improved_speed_features)))

    print(f"지형 및 FWI 피처 강제 포함 후, 방향 피처 수: {len(improved_direction_features)}, 속도 피처 수: {len(improved_speed_features)}")

    param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20], 'min_samples_leaf': [1, 3]}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # --- 확산 속도 모델 재학습 ---
    print("\n--- 🚀 개선된 확산 속도 모델 재학습 시작 ---")
    X_speed = df[improved_speed_features].fillna(0)
    y_speed = df["spread_speed_class"].astype(int)
    scaler_speed = RobustScaler()
    X_speed_scaled = scaler_speed.fit_transform(X_speed)
    grid_search_speed = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), param_grid, cv=kf, scoring='f1_weighted', n_jobs=-1, verbose=1)
    grid_search_speed.fit(X_speed_scaled, y_speed)
    best_speed_model = grid_search_speed.best_estimator_
    y_pred_speed = best_speed_model.predict(X_speed_scaled)
    print("\n[속도 모델 재학습 최종 성능]")
    print(classification_report(y_speed, y_pred_speed, zero_division=0))
    report_dict = classification_report(y_speed, y_pred_speed, zero_division=0, output_dict=True)
    with open(os.path.join(script_dir, SPEED_PERFORMANCE_PATH), 'w') as f: json.dump(report_dict, f, indent=4)
    print(f"✅ 속도 모델 성능 지표가 '{SPEED_PERFORMANCE_PATH}' 파일에 저장되었습니다.")
    joblib.dump(best_speed_model, os.path.join(script_dir, SPEED_MODEL_PATH))
    joblib.dump(scaler_speed, os.path.join(script_dir, SPEED_SCALER_PATH))
    with open(os.path.join(script_dir, SPEED_COLUMNS_PATH), 'w') as f: json.dump(improved_speed_features, f, indent=4)

    # --- 확산 방향 모델 재학습 ---
    print("\n--- 🚀 개선된 확산 방향 모델 재학습 시작 ---")
    X_dir = df[improved_direction_features].fillna(0)
    y_dir = df["spread_direction_class"].astype(int)
    scaler_dir = RobustScaler()
    X_dir_scaled = scaler_dir.fit_transform(X_dir)
    grid_search_dir = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), param_grid, cv=kf, scoring='f1_weighted', n_jobs=-1, verbose=1)
    grid_search_dir.fit(X_dir_scaled, y_dir)
    best_dir_model = grid_search_dir.best_estimator_
    y_pred_dir = best_dir_model.predict(X_dir_scaled)
    print("\n[방향 모델 재학습 최종 성능]")
    print(classification_report(y_dir, y_pred_dir, zero_division=0))
    report_dict_dir = classification_report(y_dir, y_pred_dir, zero_division=0, output_dict=True)
    with open(os.path.join(script_dir, DIRECTION_PERFORMANCE_PATH), 'w') as f: json.dump(report_dict_dir, f, indent=4)
    print(f"✅ 방향 모델 성능 지표가 '{DIRECTION_PERFORMANCE_PATH}' 파일에 저장되었습니다.")
    joblib.dump(best_dir_model, os.path.join(script_dir, DIRECTION_MODEL_PATH))
    joblib.dump(scaler_dir, os.path.join(script_dir, DIRECTION_SCALER_PATH))
    with open(os.path.join(script_dir, DIRECTION_COLUMNS_PATH), 'w') as f: json.dump(improved_direction_features, f, indent=4)

if __name__ == "__main__":
    main()
