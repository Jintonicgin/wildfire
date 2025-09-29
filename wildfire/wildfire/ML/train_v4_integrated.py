import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
import math
import datetime
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, classification_report, mean_squared_error
from pathlib import Path
import ee

# 경로 설정
sys.path.append(os.path.dirname(__file__))
try:
    from fetch_all_weather import fetch_all_weather_features
    from predict import get_gee_features
except ImportError as e:
    print(f"⚠️ 모듈 임포트 오류: {e}")

# 출력 파일 경로
AREA_MODEL_PATH = "area_regressor_model_v4.joblib"
AREA_COLUMNS_PATH = "area_model_columns_v4.json"
AREA_SCALER_PATH = "area_model_scaler_v4.joblib"

SPEED_MODEL_PATH = "speed_classifier_model_v4.joblib"
SPEED_COLUMNS_PATH = "speed_model_columns_v4.json"
SPEED_SCALER_PATH = "speed_scaler_v4.joblib"

DIRECTION_MODEL_PATH = "direction_classifier_model_v4.joblib"
DIRECTION_COLUMNS_PATH = "direction_model_columns_v4.json"
DIRECTION_SCALER_PATH = "direction_scaler_v4.joblib"

_initialized = False

def classify_speed(speed: float, thresholds=(0.014, 0.11)) -> int:
    """확산 속도 분류"""
    low, high = thresholds
    if speed <= low: return 0
    if speed <= high: return 1
    return 2

def convert_degree_to_direction(deg: float) -> int:
    """풍향을 8방향으로 분류"""
    if deg is None or pd.isna(deg) or deg == -999: return 0
    return int(math.floor(((float(deg) + 22.5) % 360) / 45))

def remove_temporal_leakage_columns(df):
    """시간적 누수 컬럼 제거"""
    print("🔧 시간적 누수 컬럼 제거 중...")
    
    original_cols = len(df.columns)
    
    # 1. 종료 시점 데이터 제거
    end_cols = [col for col in df.columns if 'end' in col.lower() and col not in ['endyear', 'endmonth', 'endday', 'endtime']]
    
    # 2. 미래 시점 데이터 제거 (15h 이상)
    future_cols = []
    for col in df.columns:
        import re
        matches = re.findall(r'_(\d+)h(?!_past)', col)
        for match in matches:
            hours = int(match)
            if hours >= 15:  # 15시간 이후는 미래 데이터
                future_cols.append(col)
                break
    
    # 3. 기타 제거할 컬럼들
    other_leakage_cols = ['endyear', 'endmonth', 'endday', 'endtime']
    
    # 4. 타겟 관련 파생변수 제거
    target_derived_cols = [col for col in df.columns if any(x in col.lower() for x in 
                          ['spread_speed', 'spread_direction', 'potential_spread'])]
    
    # 모든 누수 컬럼 합치기
    leakage_cols = set(end_cols + future_cols + other_leakage_cols + target_derived_cols)
    
    # 안전한 컬럼만 선택
    safe_cols = [col for col in df.columns if col not in leakage_cols]
    
    cleaned_df = df[safe_cols].copy()
    
    print(f"📊 컬럼 정리 결과:")
    print(f"   - 원본: {original_cols}개")
    print(f"   - 제거: {len(leakage_cols)}개 (종료시점: {len(end_cols)}, 미래시점: {len(future_cols)}, 파생변수: {len(target_derived_cols)}, 기타: {len(other_leakage_cols)})")
    print(f"   - 최종: {len(safe_cols)}개")
    
    return cleaned_df

def load_and_clean_existing_data():
    """기존 CSV 데이터 로드 및 정제"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_data_path = os.path.join(script_dir, "final_merged_feature_engineered.csv")
    
    if not os.path.exists(source_data_path):
        print(f"❌ 소스 데이터 파일을 찾을 수 없습니다: {source_data_path}")
        return None
    
    print(f"📁 기존 화재 데이터 로딩: {source_data_path}")
    fire_df = pd.read_csv(source_data_path, low_memory=False)
    fire_df.columns = [col.lower() for col in fire_df.columns]
    
    # 필수 컬럼 확인
    required_cols = ['startyear', 'startmonth', 'startday', 'fire_area', 'fire_duration_hours']
    missing_cols = [col for col in required_cols if col not in fire_df.columns]
    if missing_cols:
        print(f"❌ 필수 컬럼이 누락되었습니다: {missing_cols}")
        return None
    
    # 시간적 누수 제거
    cleaned_df = remove_temporal_leakage_columns(fire_df)
    
    # 유효한 데이터만 선택
    cleaned_df = cleaned_df.dropna(subset=['fire_area', 'fire_duration_hours'])
    cleaned_df = cleaned_df[cleaned_df['fire_area'] > 0]
    cleaned_df = cleaned_df[cleaned_df['fire_duration_hours'] > 0]
    
    print(f"🔥 처리 가능한 화재 데이터: {len(cleaned_df)}건")
    
    return cleaned_df

def prepare_data_for_models(cleaned_df):
    """모델별 데이터 준비 (누수 제거된 데이터 사용)"""
    print("\n📊 모델별 데이터 준비 중...")
    
    df_clean = cleaned_df.copy()
    
    # 타겟 변수 생성
    df_clean['spread_speed'] = df_clean['fire_area'] / df_clean['fire_duration_hours']
    df_clean['spread_speed_class'] = df_clean['spread_speed'].apply(classify_speed)
    
    # 풍향 데이터 확인 및 처리
    wind_direction_col = None
    for col in ['wd10m_0h', 'wd10m_0h_original']:
        if col in df_clean.columns:
            wind_direction_col = col
            break
    
    if wind_direction_col:
        df_clean['spread_direction_class'] = df_clean[wind_direction_col].apply(convert_degree_to_direction)
    else:
        print("⚠️ 풍향 데이터를 찾을 수 없어 방향 예측을 위해 0으로 설정합니다.")
        df_clean['spread_direction_class'] = 0
    
    # 수치형 컬럼만 선택
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    # 제외할 컬럼들 (타겟 및 파생변수)
    cols_to_exclude = [
        'fire_area', 'fire_duration_hours', 'spread_speed', 
        'spread_speed_class', 'spread_direction_class'
    ]
    
    # 풍향 원본 컬럼도 제외
    if wind_direction_col:
        cols_to_exclude.append(wind_direction_col)
    
    # 피처 컬럼들
    feature_cols = [col for col in numeric_cols if col not in cols_to_exclude]
    
    # 추가 정제: 위치 정보는 유지하되 개인정보성 데이터 확인
    sensitive_cols = [col for col in feature_cols if any(x in col.lower() for x in ['id', 'name', 'address'])]
    if sensitive_cols:
        print(f"⚠️ 민감정보 가능성 컬럼 제외: {sensitive_cols}")
        feature_cols = [col for col in feature_cols if col not in sensitive_cols]
    
    # NaN 값 처리
    X_all = df_clean[feature_cols].fillna(0)
    
    print(f"📋 최종 피처 수: {len(feature_cols)}")
    print(f"📋 데이터 수: {len(df_clean)}")
    print(f"📋 데이터 대 피처 비율: {len(df_clean)/len(feature_cols):.2f}")
    
    if len(df_clean) < len(feature_cols) * 5:
        print("⚠️ 경고: 피처 수가 데이터 수에 비해 너무 많아 과적합 위험이 있습니다.")
    
    return df_clean, X_all, feature_cols

def train_area_model(df_clean, X_all, feature_cols):
    """피해면적 예측 모델 학습 (데이터 누수 방지)"""
    print("\n🎯 피해면적 예측 모델 학습 시작...")
    
    # 데이터 수에 맞는 피처 선택
    data_count = len(df_clean)
    max_features_area = min(100, data_count // 5)  # 데이터 수의 1/5 또는 최대 100개
    
    print(f"📊 Area 모델 - 데이터 수: {data_count}, 최대 피처 수: {max_features_area}")
    
    # 피처 선택 (area 예측에 특화)
    area_features = select_important_features(X_all, feature_cols, max_features_area)
    X_area_selected = X_all[area_features].fillna(0)
    
    # 타겟 준비 (로그 변환)
    y_area = np.log1p(df_clean['fire_area'])
    
    # 이상치 제거 (95% 분위수 이하로 완화)
    area_95th = df_clean['fire_area'].quantile(0.95)
    mask = df_clean['fire_area'] <= area_95th
    X_area = X_area_selected[mask]
    y_area = y_area[mask]
    
    print(f"📊 이상치 제거 후 데이터 수: {len(X_area)}")
    
    # 데이터가 너무 적으면 경고
    if len(X_area) < 20:
        print("⚠️ 경고: 데이터가 너무 적어 모델 성능이 신뢰성이 낮을 수 있습니다.")
    
    # 스케일링
    scaler_area = RobustScaler()
    X_area_scaled = scaler_area.fit_transform(X_area)
    
    # 데이터 수에 따른 하이퍼파라미터 조정
    if data_count < 50:
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],
            'min_samples_leaf': [2, 5],
            'max_features': ['sqrt']
        }
        cv_folds = 3
    elif data_count < 200:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_leaf': [1, 3],
            'max_features': ['sqrt', 0.3]
        }
        cv_folds = 5
    else:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [20, 30],
            'min_samples_leaf': [1, 3],
            'max_features': ['sqrt', 0.3]
        }
        cv_folds = 5
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    rf_area = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    print("🔧 하이퍼파라미터 튜닝 중...")
    grid_search = GridSearchCV(
        estimator=rf_area, param_grid=param_grid, cv=kf,
        scoring='neg_root_mean_squared_error', verbose=1, n_jobs=-1
    )
    grid_search.fit(X_area_scaled, y_area)
    
    # 최적 모델 평가
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_area_scaled)
    r2 = r2_score(y_area, y_pred)
    rmse = np.sqrt(mean_squared_error(y_area, y_pred))
    
    print(f"✅ Area 모델 성능:")
    print(f"   - 사용된 피처 수: {len(area_features)}")
    print(f"   - 최적 파라미터: {grid_search.best_params_}")
    print(f"   - R² Score: {r2:.4f}")
    print(f"   - RMSE: {rmse:.4f}")
    
    return best_model, scaler_area, area_features

def select_important_features(X_all, feature_cols, max_features=50):
    """중요 피처 선택 (데이터 대비 적절한 수)"""
    print(f"\n🔍 중요 피처 선택 중... (최대 {max_features}개)")
    
    # 1. 핵심 도메인 피처 (항상 포함)
    domain_keywords = {
        'temporal': ['startyear', 'startmonth', 'startday'],
        'weather_current': ['t2m_0h', 'rh2m_0h', 'ws10m_0h', 'ps_0h'],
        'fire_weather': ['ffmc_0h', 'dmc_0h', 'dc_0h', 'isi_0h', 'bui_0h', 'fwi_0h'],
        'terrain': ['elevation_mean', 'elevation_std', 'slope_mean', 'slope_std', 'aspect_mode'],
        'vegetation': ['ndvi_before', 'treecover'],
        'historical_weather': ['dry_days_7d_start', 'dry_days_14d_start', 'dry_days_30d_start']
    }
    
    # 각 카테고리별 피처 수집
    priority_features = []
    for category, keywords in domain_keywords.items():
        category_features = []
        for keyword in keywords:
            matching = [f for f in feature_cols if keyword in f.lower()]
            category_features.extend(matching)
        
        # 중복 제거 후 추가
        unique_features = list(set(category_features))
        priority_features.extend(unique_features)
        print(f"   - {category}: {len(unique_features)}개")
    
    # 중복 제거
    priority_features = list(set(priority_features))
    
    # 2. 통계적 특성 기반 추가 피처 (분산이 0이 아닌 것들)
    remaining_features = [f for f in feature_cols if f not in priority_features]
    feature_variances = X_all[remaining_features].var().sort_values(ascending=False)
    
    # 분산이 있는 피처들 중 상위 선택
    additional_count = max_features - len(priority_features)
    if additional_count > 0:
        high_variance_features = feature_variances.head(additional_count).index.tolist()
        priority_features.extend(high_variance_features)
    
    # 최종 피처 선택
    selected_features = priority_features[:max_features]
    
    print(f"📊 최종 선택된 피처: {len(selected_features)}개")
    return selected_features

def train_speed_direction_models(df_clean, X_all, feature_cols):
    """속도/방향 분류 모델 학습 (적절한 피처 수 사용)"""
    print("\n🎯 속도/방향 분류 모델 학습 시작...")
    
    # 데이터 수에 맞는 피처 수 결정
    data_count = len(df_clean)
    max_features = min(50, data_count // 10)  # 데이터 수의 1/10 또는 최대 50개
    
    print(f"📊 데이터 수: {data_count}, 최대 피처 수: {max_features}")
    
    # 중요 피처 선택
    selected_features = select_important_features(X_all, feature_cols, max_features)
    
    X_core = X_all[selected_features].fillna(0)
    
    # 속도 모델 학습
    print("\n🚀 속도 분류 모델 학습...")
    y_speed = df_clean['spread_speed_class'].astype(int)
    scaler_speed = RobustScaler()
    X_speed_scaled = scaler_speed.fit_transform(X_core)
    
    param_grid_clf = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_leaf': [1, 3]
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_speed = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        param_grid_clf, cv=kf, scoring='f1_weighted', n_jobs=-1, verbose=1
    )
    grid_speed.fit(X_speed_scaled, y_speed)
    
    best_speed_model = grid_speed.best_estimator_
    y_pred_speed = best_speed_model.predict(X_speed_scaled)
    
    print(f"✅ Speed 모델 성능:")
    print(classification_report(y_speed, y_pred_speed, zero_division=0))
    
    # 방향 모델 학습 (풍향 관련 피처 제외)
    print("\n🧭 방향 분류 모델 학습...")
    direction_features = [f for f in selected_features if not any(x in f.lower() for x in ['wd10m', 'wd2m'])]
    X_dir = X_all[direction_features].fillna(0)
    y_direction = df_clean['spread_direction_class'].astype(int)
    
    scaler_direction = RobustScaler()
    X_dir_scaled = scaler_direction.fit_transform(X_dir)
    
    # 데이터가 적은 경우 더 간단한 하이퍼파라미터 사용
    if data_count < 100:
        param_grid_simple = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],
            'min_samples_leaf': [2, 5]
        }
        param_grid_clf = param_grid_simple
    
    grid_direction = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        param_grid_clf, cv=min(3, kf.n_splits), scoring='f1_weighted', n_jobs=-1, verbose=1
    )
    grid_direction.fit(X_dir_scaled, y_direction)
    
    best_direction_model = grid_direction.best_estimator_
    y_pred_direction = best_direction_model.predict(X_dir_scaled)
    
    print(f"✅ Direction 모델 성능:")
    print(classification_report(y_direction, y_pred_direction, zero_division=0))
    
    return (best_speed_model, scaler_speed, selected_features,
            best_direction_model, scaler_direction, direction_features)

def main():
    """메인 실행 함수 (데이터 누수 방지)"""
    print("🚀 데이터 누수 방지 모델 학습 시작...")
    
    # 기존 정제된 데이터 로드
    cleaned_df = load_and_clean_existing_data()
    
    if cleaned_df is None or len(cleaned_df) < 10:
        print("❌ 충분한 데이터를 로드하지 못했습니다.")
        return
    
    # 샘플링 (전체 데이터 사용 권장, 테스트시에만 제한)
    if len(cleaned_df) > 200:
        # 전체 데이터 사용
        working_df = cleaned_df.copy()
        print(f"📊 전체 데이터 사용: {len(working_df)}건")
    else:
        # 작은 데이터셋인 경우 모두 사용
        working_df = cleaned_df.copy()
        print(f"📊 사용 가능한 모든 데이터 사용: {len(working_df)}건")
    
    # 모델별 데이터 준비
    df_clean, X_all, feature_cols = prepare_data_for_models(working_df)
    
    # Area 모델 학습 (1191개 피처)
    area_model, area_scaler, area_features = train_area_model(df_clean, X_all, feature_cols)
    
    # Speed/Direction 모델 학습 (25개 핵심 피처)
    (speed_model, speed_scaler, speed_features,
     direction_model, direction_scaler, direction_features) = train_speed_direction_models(df_clean, X_all, feature_cols)
    
    # 모델 저장
    print("\n💾 모델 저장 중...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Area 모델
    joblib.dump(area_model, os.path.join(script_dir, AREA_MODEL_PATH))
    joblib.dump(area_scaler, os.path.join(script_dir, AREA_SCALER_PATH))
    with open(os.path.join(script_dir, AREA_COLUMNS_PATH), 'w') as f:
        json.dump(area_features, f, indent=2)
    
    # Speed 모델
    joblib.dump(speed_model, os.path.join(script_dir, SPEED_MODEL_PATH))
    joblib.dump(speed_scaler, os.path.join(script_dir, SPEED_SCALER_PATH))
    with open(os.path.join(script_dir, SPEED_COLUMNS_PATH), 'w') as f:
        json.dump(speed_features, f, indent=2)
    
    # Direction 모델
    joblib.dump(direction_model, os.path.join(script_dir, DIRECTION_MODEL_PATH))
    joblib.dump(direction_scaler, os.path.join(script_dir, DIRECTION_SCALER_PATH))
    with open(os.path.join(script_dir, DIRECTION_COLUMNS_PATH), 'w') as f:
        json.dump(direction_features, f, indent=2)
    
    print("🎉 DB 호환 통합 모델 학습 완료!")
    print(f"📊 Area 모델: {len(area_features)}개 피처")
    print(f"🚀 Speed 모델: {len(speed_features)}개 피처")
    print(f"🧭 Direction 모델: {len(direction_features)}개 피처")

if __name__ == "__main__":
    main()