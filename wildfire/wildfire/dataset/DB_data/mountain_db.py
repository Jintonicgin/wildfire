import pandas as pd
import json
import datetime
import time
import sys
import ee
import warnings

# --- 경로 설정 및 외부 모듈 임포트 ---
# 이 스크립트가 DB_data/ 안에 있으므로, 상위 디렉토리로 경로를 설정해야 합니다.
DATASET_PATH = "/Users/mmymacymac/Developer/Projects/eclipse/WildFire/dataset/"
sys.path.append(DATASET_PATH)

from fetch_all_weather_core import fetch_all_weather_features
from prediction_core import get_gee_features
from oracle_db import OracleDB # DB 저장을 위해 주석 해제하고 사용

warnings.filterwarnings("ignore")

# --- 전역 변수 ---
MODEL_PATH = DATASET_PATH
MODELS_COLUMNS = {}

# --- 초기화 함수 ---
def initialize():
    """GEE 및 모델 컬럼 목록을 초기화합니다."""
    try:
        ee.Initialize(project='wildfire-464907')
        print("✅ GEE가 성공적으로 초기화되었습니다.")
    except Exception:
        print("⚠️ GEE 인증이 필요합니다. 브라우저에서 인증을 완료해주세요.")
        ee.Authenticate()
        ee.Initialize(project='wildfire-464907')
        print("✅ GEE 인증 및 초기화가 완료되었습니다.")

    global MODELS_COLUMNS
    try:
        with open(MODEL_PATH + "area_model_columns_v2.json") as f:
            MODELS_COLUMNS['area_v2'] = json.load(f)
        with open(MODEL_PATH + "speed_model_columns.json") as f:
            MODELS_COLUMNS['speed'] = json.load(f)
        with open(MODEL_PATH + "direction_model_columns.json") as f:
            MODELS_COLUMNS['direction'] = json.load(f)
        print("✅ 모델별 필요 피처 목록을 성공적으로 로드했습니다.")
    except FileNotFoundError as e:
        print(f"❌ 에러: 모델 컬럼 파일을 찾을 수 없습니다. ({e})")
        return False
    return True

# --- 실제 데이터 수집 함수 (최종 버전) ---
def get_real_features_for_coord(lat, lon, timestamp):
    """주어진 좌표에 대해 실제 기상 및 지형 데이터를 수집합니다."""
    print(f"\n--- 좌표 ({lat:.4f}, {lon:.4f})에 대한 데이터 수집 시작 ---")

    # 1. 기상 데이터 수집 (total_duration_hours 포함)
    print("[1/2] NASA에서 기상 데이터를 가져오는 중...")
    weather_features = fetch_all_weather_features(lat, lon, timestamp)
    if not weather_features or not weather_features.get("success"):
        print(f"❌ ({lat:.4f}, {lon:.4f})의 기상 피처 수집 실패.")
        return None
    print("✅ 기상 데이터 수집 완료.")

    # 2. 지형 데이터 수집 (min/max 포함)
    print("[2/2] GEE에서 지형 데이터를 가져오는 중...")
    gee_features = get_gee_features(lat, lon)
    if not gee_features or gee_features.get("slope_mean", -999) == -999:
        print(f"❌ ({lat:.4f}, {lon:.4f})의 GEE 피처 수집 실패. 기본값으로 대체합니다.")
        gee_features = {
            "ndvi_before": -999, "treecover_pre_fire_5x5": -999, 
            "elevation_mean": -999, "elevation_std": -999, "elevation_min": -999, "elevation_max": -999,
            "slope_mean": -999, "slope_std": -999, "slope_min": -999, "slope_max": -999,
            "aspect_mode": -999, "aspect_std": -999
        }
    print("✅ 지형 데이터 수집 완료.")

    # 3. 모든 피처 통합
    all_features = {**weather_features, **gee_features}
    all_features.update({
        "lat": lat, "lng": lon, "startyear": timestamp.year,
        "startmonth": timestamp.month, "startday": timestamp.day,
    })
    
    # 상호작용 피처 추가 (예시)
    duration = all_features.get("duration_hours", 1.0)
    all_features["duration_x_ws10m"] = duration * all_features.get("WS10M", 0)
    all_features["duration_x_t2m"] = duration * all_features.get("T2M", 0)
    all_features["duration_x_rh2m"] = duration * all_features.get("RH2M", 0)
    all_features["duration_x_fwi"] = duration * all_features.get("FWI", 0)
    all_features["duration_x_isi"] = duration * all_features.get("ISI", 0)

    return all_features

# --- 피처 검증 함수 ---
def verify_features(collected_features):
    """수집된 피처가 모델 요구사항을 충족하는지 검증합니다."""
    if not collected_features:
        return False
    print("\n--- 피처 검증 시작 ---")
    all_good = True
    available_keys = set(collected_features.keys())
    for model_name, required_cols in MODELS_COLUMNS.items():
        missing_cols = set(required_cols) - available_keys
        if not missing_cols:
            print(f"✅ [{model_name} 모델]의 모든 피처가 존재합니다.")
        else:
            all_good = False
            print(f"❌ [{model_name} 모델]에 필요한 피처 {len(missing_cols)}개가 누락되었습니다: {sorted(list(missing_cols))}")
    return all_good

# --- 메인 실행 로직 ---
def main():
    """CSV 파일을 읽고, 각 좌표의 피처를 수집 및 검증 후 DB에 저장합니다."""
    if not initialize():
        return

    csv_path = f"{DATASET_PATH}gangwon_mountain_points.csv"
    try:
        df = pd.read_csv(csv_path)
        print(f"\n✅ '{csv_path}' 파일을 성공적으로 읽었습니다. 총 {len(df)}개의 좌표를 처리합니다.")
    except FileNotFoundError:
        print(f"❌ 에러: '{csv_path}' 파일을 찾을 수 없습니다.")
        return

    db = OracleDB() # DB 연결
    all_results_for_json = []

    for index, row in df.iterrows():
        region_name = row['region'] # region 컬럼 읽기
        lat = row['lat']
        lon = row['lng']
        timestamp = datetime.datetime.now()

        features = get_real_features_for_coord(lat, lon, timestamp)

        if features:
            features['region_name'] = region_name # features 딕셔너리에 region_name 추가

        if features and verify_features(features):
            print("\n🎉 모든 피처 검증 통과! DB 저장 준비 완료.")
            all_results_for_json.append(features)
            # 아래 라인의 주석을 해제하여 DB에 저장
            db.insert_mountain_features(features)
            print("--- (시뮬레이션) DB에 데이터 저장 완료 ---")
        else:
            print("\n⚠️ 피처 검증 실패. DB에 저장하지 않습니다.")

        print("\n" + "="*50 + "\n")
        time.sleep(2)

    db.close() # DB 연결 해제

    if all_results_for_json:
        output_filename = f'{DATASET_PATH}DB_data/gangwon_mountain_features_for_db.json'
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(all_results_for_json, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n\n🎉 모든 작업 완료! 검증된 피처를 '{output_filename}' 파일에 저장했습니다.")
    else:
        print("\n\n- 작업 완료. 검증을 통과한 피처가 없습니다.")

if __name__ == "__main__":
    main()