import sys
import json
import joblib
import numpy as np
import pandas as pd
import warnings
import os
import traceback

# --- 경로 설정 및 모듈 임포트 ---
try:
    from wildfire.ML.DB_data.oracle_db import OracleDB
    from model_definitions import EnsembleRegressor, EnsembleClassifier
    import model_definitions
    sys.modules['model_definitions'] = model_definitions
except ImportError as e:
    print(f"[초기화 오류] 필수 모듈 임포트에 실패했습니다: {e}")
    sys.exit(1)
    from wildfire.ML.DB_data.oracle_db import OracleDB
    import model_definitions

warnings.filterwarnings("ignore")

# --- 전역 변수 및 상수 ---
MODEL_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS = {}
DIRECTION_MAP = {0: 'N', 1: 'NE', 2: 'E', 3: 'SE', 4: 'S', 5: 'SW', 6: 'W', 7: 'NW'}
SPEED_CATEGORY_MAP = {0: 50.0, 1: 200.0, 2: 500.0}

def initialize_models():
    """모든 모델, 스케일러, 컬럼 목록을 전역 변수 MODELS에 한 번만 로드합니다."""
    global MODELS
    if MODELS:  # 이미 로드되었으면 실행하지 않음
        return
    try:
        print("모든 예측 모델을 로드하는 중...")
        MODELS = {
            "area_model": joblib.load(os.path.join(MODEL_PATH, "area_regressor_model_v3_tuned.joblib")),
            "speed_model": joblib.load(os.path.join(MODEL_PATH, "speed_classifier_model_v2_tuned_cw.joblib")),
            "direction_model": joblib.load(os.path.join(MODEL_PATH, "direction_classifier_model_v2_tuned_cw.joblib")),
            "area_scaler": joblib.load(os.path.join(MODEL_PATH, "area_model_scaler_v3_tuned.joblib")),
            "speed_scaler": joblib.load(os.path.join(MODEL_PATH, "speed_scaler_v2_tuned_cw.joblib")),
            "direction_scaler": joblib.load(os.path.join(MODEL_PATH, "direction_scaler_v2_tuned_cw.joblib")),
        }
        with open(os.path.join(MODEL_PATH, "area_model_columns_v3_tuned.json")) as f:
            MODELS["area_cols"] = json.load(f)
        with open(os.path.join(MODEL_PATH, "speed_model_columns_v2_tuned_cw.json")) as f:
            MODELS["speed_cols"] = json.load(f)
        with open(os.path.join(MODEL_PATH, "direction_model_columns_v2_tuned_cw.json")) as f:
            MODELS["direction_cols"] = json.load(f)
        print("✅ 모델 로드 완료.")
    except Exception as e:
        raise RuntimeError(f"모델 로드 실패: {e}")

def create_column_mapping():
    """DB 컬럼명을 모델이 기대하는 컬럼명으로 매핑하는 딕셔너리를 생성합니다."""
    return {
        # FWI 지수들 - DB에서 대문자로 저장됨
        'ffmc_0h': 'ffmc',
        'dmc_0h': 'dmc', 
        'dc_0h': 'dc',
        'isi_0h': 'isi',
        'bui_0h': 'bui',
        'fwi_0h': 'fwi',
        
        # 기상 데이터 - 현재값들 (0h)
        't2m_0h': 't2m',
        'rh2m_0h': 'rh2m', 
        'ws10m_0h': 'ws10m',
        'wd10m_0h': 'wd10m',
        'ps_0h': 'ps',
        'allsky_sfc_sw_dwn_0h': 'allsky_sfc_sw_dwn',
        'prec_0h': 'prectotcorr',
        
        # 건조일수 관련
        'dry_days_7d_start': 'dry_days_7d_start_past',
        'dry_days_14d_start': 'dry_days_14d_start_past',
        'dry_days_30d_start': 'dry_days_30d_start_past', 
        'dry_days_60d_start': 'dry_days_60d_start_past',
        'dry_days_90d_start': 'dry_days_90d_start_past',
        
        # 시간 관련
        'startmonth': 'start_month',
        'startday': 'start_day',
        'startyear': 'start_year',
        
        # 계절 정보
        'is_spring': 'is_spring',
        'is_summer': 'is_summer', 
        'is_autumn': 'is_autumn',
        'is_winter': 'is_winter'
    }

def map_db_to_model_columns(df):
    """DB 컬럼명을 모델이 기대하는 컬럼명으로 변환합니다."""
    column_mapping = create_column_mapping()
    
    # 현재 시간을 기준으로 시작월과 계절 정보 추가
    import datetime
    now = datetime.datetime.now()
    current_month = now.month
    current_day = now.day
    current_year = now.year
    
    # 시간 정보 추가
    df['start_month'] = current_month
    df['start_day'] = current_day
    df['start_year'] = current_year
    
    # 계절 정보 추가
    df['is_spring'] = 1 if current_month in [3, 4, 5] else 0
    df['is_summer'] = 1 if current_month in [6, 7, 8] else 0  
    df['is_autumn'] = 1 if current_month in [9, 10, 11] else 0
    df['is_winter'] = 1 if current_month in [12, 1, 2] else 0
    
    # DB 컬럼을 모델 컬럼으로 직접 매핑
    for model_col, db_col in column_mapping.items():
        if db_col in df.columns:
            df[model_col] = df[db_col]
            print(f"      매핑됨: {db_col} -> {model_col} = {df[model_col].iloc[0]}")
    
    # FWI 지수들이 DB에 있는지 확인하고 직접 복사
    fwi_columns = ['ffmc', 'dmc', 'dc', 'isi', 'bui', 'fwi']
    for col in fwi_columns:
        if col in df.columns:
            df[f'{col}_0h'] = df[col]
            print(f"      FWI 복사: {col} -> {col}_0h = {df[col].iloc[0]}")
    
    # 기상 데이터도 직접 복사 (_0h 형태로)
    weather_columns = ['t2m', 'rh2m', 'ws10m', 'wd10m', 'ps']
    for col in weather_columns:
        if col in df.columns:
            df[f'{col}_0h'] = df[col]
            print(f"      기상 복사: {col} -> {col}_0h = {df[col].iloc[0]}")
    
    # 건조일수 데이터 확인 및 매핑
    dry_day_patterns = ['dry_days_7d_start_past', 'dry_days_14d_start_past', 
                       'dry_days_30d_start_past', 'dry_days_60d_start_past', 'dry_days_90d_start_past']
    for pattern in dry_day_patterns:
        if pattern in df.columns:
            base_name = pattern.replace('_past', '')
            df[base_name] = df[pattern] 
            print(f"      건조일수 복사: {pattern} -> {base_name} = {df[pattern].iloc[0]}")
    
    # 건조일수 데이터가 DB에 없으므로 추정값 생성
    prec_value = df.get('prectotcorr', [0]).iloc[0] if 'prectotcorr' in df.columns else 0
    rh_value = df.get('rh2m', [50]).iloc[0] if 'rh2m' in df.columns else 50
    
    # 기본 건조일수 추정 (강수량과 습도 기반)
    if prec_value <= 0.1 and rh_value < 40:
        base_dry_days = 7  # 매우 건조
    elif prec_value <= 0.5 and rh_value < 60:
        base_dry_days = 3  # 보통 건조  
    elif prec_value <= 1.0:
        base_dry_days = 1  # 약간 건조
    else:
        base_dry_days = 0  # 습함
    
    # 각 기간별 건조일수 생성 (기본값에서 비례적으로 증가)
    df['dry_days_7d_start'] = base_dry_days
    df['dry_days_14d_start'] = min(base_dry_days * 2, 14)
    df['dry_days_30d_start'] = min(base_dry_days * 4, 30)  
    df['dry_days_60d_start'] = min(base_dry_days * 6, 60)
    df['dry_days_90d_start'] = min(base_dry_days * 8, 90)
    df['consecutive_dry_days_start'] = base_dry_days
    
    print(f"      건조일수 추정값 생성: base={base_dry_days} (강수={prec_value}mm, 습도={rh_value}%)")
    print(f"        7d={df['dry_days_7d_start'].iloc[0]}, 14d={df['dry_days_14d_start'].iloc[0]}, 30d={df['dry_days_30d_start'].iloc[0]}")
    print(f"        60d={df['dry_days_60d_start'].iloc[0]}, 90d={df['dry_days_90d_start'].iloc[0]}, consecutive={df['consecutive_dry_days_start'].iloc[0]}")
    
    return df

def align_features(df, expected_columns):
    """모델이 기대하는 형식에 맞게 데이터프레임의 컬럼을 정렬하고 누락된 컬럼은 0으로 채웁니다."""
    # 먼저 DB 컬럼을 모델 컬럼으로 매핑
    df = map_db_to_model_columns(df)
    
    missing_cols = set(expected_columns) - set(df.columns)
    existing_cols = set(expected_columns) & set(df.columns)
    
    print(f"    총 필요 컬럼: {len(expected_columns)}")
    print(f"    존재하는 컬럼: {len(existing_cols)}")
    print(f"    누락된 컬럼: {len(missing_cols)}")
    
    if len(missing_cols) > 0:
        print(f"    ⚠️ 누락된 컬럼: {len(missing_cols)}개")
        print("    누락된 컬럼 목록 (전체):")
        for i, col in enumerate(sorted(missing_cols)):
            print(f"      {i+1}. {col}")
    
    for c in missing_cols:
        df[c] = 0
    return df[expected_columns]

def run_prediction_pipeline(features, duration_hours=6):
    """피처 딕셔너리를 받아 예측을 수행하고 결과를 반환합니다."""
    features_df = pd.DataFrame([features])
    
    # 디버깅: 입력 피처값 확인
    region_name = features.get('region_name', 'Unknown')
    print(f"\n=== {region_name} 예측 디버깅 ===")
    print(f"총 입력 피처 개수: {len(features)}")
    print("입력 피처 샘플 (처음 10개):")
    for i, (k, v) in enumerate(list(features.items())[:10]):
        print(f"  {k}: {v}")
    
    # 중요한 기상 피처들 확인
    important_weather = ['t2m', 'rh2m', 'ws10m', 'wd10m', 'prectotcorr']
    print("\n주요 기상 데이터:")
    for key in important_weather:
        if key in features:
            print(f"  {key}: {features[key]}")

    # --- Area Model Prediction ---
    area_input_df = align_features(features_df.copy(), MODELS["area_cols"])
    print(f"\nArea 모델:")
    print(f"  입력 컬럼 수: {len(MODELS['area_cols'])}")
    print("  입력 샘플 (처음 5개):")
    for i, (col, val) in enumerate(list(area_input_df.iloc[0].items())[:5]):
        print(f"    {col}: {val}")
    
    area_scaled = MODELS["area_scaler"].transform(area_input_df)
    print(f"  스케일링 후 샘플: {area_scaled[0][:5]}")
    
    area_log = MODELS["area_model"].predict(area_scaled)[0]
    base_area = float(np.expm1(area_log)) if np.isfinite(area_log) else 0.0
    time_factor = (duration_hours / 6.0) ** 1.5
    predicted_area = base_area * time_factor
    print(f"  예측 결과: log={area_log:.4f}, base_area={base_area:.4f}, final_area={predicted_area:.4f}")

    # --- Speed & Direction Model Prediction ---
    speed_input_df = align_features(features_df.copy(), MODELS["speed_cols"])
    speed_scaled = MODELS["speed_scaler"].transform(speed_input_df)
    speed_cat = int(MODELS["speed_model"].predict(speed_scaled)[0])
    print(f"\nSpeed 모델:")
    print(f"  입력 컬럼 수: {len(MODELS['speed_cols'])}")
    print(f"  스케일링 후 샘플: {speed_scaled[0][:5]}")
    print(f"  예측 카테고리: {speed_cat}")

    direction_input_df = align_features(features_df.copy(), MODELS["direction_cols"])
    direction_scaled = MODELS["direction_scaler"].transform(direction_input_df)
    direction_cat = int(MODELS["direction_model"].predict(direction_scaled)[0])
    print(f"\nDirection 모델:")
    print(f"  입력 컬럼 수: {len(MODELS['direction_cols'])}")
    print(f"  스케일링 후 샘플: {direction_scaled[0][:5]}")
    print(f"  예측 카테고리: {direction_cat}")

    # --- Final Result Combination ---
    area_m2 = predicted_area * 10000
    total_distance = np.sqrt(area_m2 / np.pi) if area_m2 > 0 else 0.0
    
    print(f"\n최종 결과:")
    print(f"  피해면적: {predicted_area:.2f} ha")
    print(f"  확산거리: {total_distance:.2f} m") 
    print(f"  확산방향: {DIRECTION_MAP.get(direction_cat, 'N/A')}")
    print(f"  확산속도: {SPEED_CATEGORY_MAP.get(speed_cat, 0.0)} m/h")
    print("=" * 50)
        
    return {
        "final_damage_area": predicted_area,
        "total_distance": total_distance,
        "predicted_speed_category": speed_cat,
        "predicted_direction_category": direction_cat,
        "predicted_spread_direction": DIRECTION_MAP.get(direction_cat, "N/A"),
        "predicted_spread_speed": SPEED_CATEGORY_MAP.get(speed_cat, 0.0),
    }

def predict_by_region(region_name, duration_hours=6):
    """REGION_NAME으로 DB에서 모든 피처를 가져와 예측을 수행하는 메인 함수"""
    initialize_models()  # 모델 초기화 추가
    
    db = None
    try:
        db = OracleDB()
        if not db.conn:
            raise ConnectionError("DB 연결에 실패했습니다.")

        features = db.get_features_by_region(region_name)
        if not features:
            return {"error": f'{region_name}에 해당하는 예측 피처를 DB에서 찾을 수 없습니다.'}
        
        # DB에서 가져온 피처로 예측 실행
        prediction_result = run_prediction_pipeline(features, duration_hours)
        
        # 최종 결과에 region_name 추가
        prediction_result["region_name"] = region_name
        return prediction_result

    except Exception as e:
        return {"error": f"Prediction failed for region {region_name}: {str(e)}", "traceback": traceback.format_exc()}
    finally:
        if db:
            db.close()

if __name__ == "__main__":
    # 스크립트 실행 시 모델 초기화
    try:
        initialize_models()
    except RuntimeError as e:
        print(json.dumps({"error": str(e)}, indent=2, ensure_ascii=False))
        sys.exit(1)

    if len(sys.argv) > 1:
        region = sys.argv[1]
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 6
        
        final_result = predict_by_region(region, duration)
        
        # 결과 출력
        if "error" not in final_result:
            print(f"\n🔥 Prediction Result for {region} ({duration}h):")
            print(f"   Final Area: {final_result['final_damage_area']:.2f} ha")
            print(f"   Distance: {final_result['total_distance']:.2f} m")
            print(f"   Direction: {final_result['predicted_spread_direction']} (cat: {final_result['predicted_direction_category']})")
            print(f"   Speed: {final_result['predicted_spread_speed']:.2f} m/h (cat: {final_result['predicted_speed_category']})")
            print("\n-- JSON Output --")

        print(json.dumps(final_result, indent=2, ensure_ascii=False))
    else:
        print(json.dumps({"error": "Usage: python predict_from_feature.py <REGION_NAME> [duration_hours]"}))
