import pandas as pd
import json
import datetime
import time
import sys
import ee
import warnings
from tqdm import tqdm

# --- 경로 설정 및 외부 모듈 임포트 ---
import os

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(DATASET_PATH)

try:
    from wildfire.ML.fetch_all_weather import fetch_all_weather_features
    from wildfire.ML.predict import get_gee_features
    from DB_data.oracle_db import OracleDB
    from wildfire.ML.feature_utils import generate_statistical_features, generate_custom_features  # NEW IMPORT
except ImportError as e:
    print(f"⚠️ 모듈 임포트 오류: {e}")
    sys.path.append(os.path.dirname(__file__))
    from oracle_db import OracleDB
    from wildfire.ML.fetch_all_weather import fetch_all_weather_features
    from wildfire.ML.predict import get_gee_features
    # Add fallback for feature_utils if needed, but for now assume it's always there

warnings.filterwarnings("ignore")


def get_features_for_db(lat, lon):
    """예측 시점(현재)을 기준으로 DB에 저장할 모든 피처를 생성합니다."""
    timestamp = datetime.datetime.now()

    print(f"\n--- 좌표 ({lat:.4f}, {lon:.4f})에 대한 데이터 수집 시작 ---")
    # 1. GEE 피처 가져오기
    gee_features = get_gee_features(lat, lon)
    if not gee_features:
        print(f"Warning: GEE 피처 수집 실패 ({lat}, {lon})")
        gee_features = {}

    # 2. 과거 날씨 피처 모두 가져오기 (5일 전을 기준으로 데이터 수집)
    weather_features = fetch_all_weather_features(lat, lon, timestamp, offset_days=10)
    if not weather_features or not weather_features.get("success"):
        print(f"Warning: 날씨 피처 수집 실패 ({lat}, {lon})")
        return None

    # 3. 통계 피처 생성 (72개)
    print("📊 통계 피처 생성 중...")
    stats_features = generate_statistical_features(weather_features)
    print(f"✅ 통계 피처 {len(stats_features)}개 생성 완료")

    # 4. 커스텀 피처 생성 (7개)
    print("🎯 커스텀 위험 피처 생성 중...")
    custom_features = generate_custom_features(weather_features, gee_features)
    print(f"✅ 커스텀 피처 {len(custom_features)}개 생성 완료")

    # 5. 모든 피처 통합
    all_features = {**gee_features, **weather_features, **stats_features, **custom_features}
    all_features.update({
        "lat": lat,
        "lng": lon
    })

    all_features.pop('success', None)
    print(f"🎉 총 피처 생성 완료: {len(all_features)}개 피처 (기존 549개 + 추가 79개)")
    return all_features


def main():
    """
    gangwon_mountain_points.csv의 지역 목록을 읽어, 각 지역의
    예측용 피처를 생성하고 DB에 저장/업데이트합니다.
    """
    try:
        ee.Initialize(project='wildfire-464907')
        print("✅ GEE가 성공적으로 초기화되었습니다.")
    except Exception:
        print("⚠️ GEE 인증이 필요합니다. 브라우저에서 인증을 완료해주세요.")
        ee.Authenticate()
        ee.Initialize(project='wildfire-464907')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_csv_path = os.path.join(script_dir, "..", "gangwon_mountain_points.csv")

    try:
        df = pd.read_csv(source_csv_path)
        print(f"\n✅ '{source_csv_path}' 파일을 성공적으로 읽었습니다. 총 {len(df)}개의 좌표를 처리합니다.")
    except FileNotFoundError:
        print(f"❌ 에러: '{source_csv_path}' 파일을 찾을 수 없습니다.")
        return

    try:
        db = OracleDB()
        if not db.conn: raise Exception("DB 연결 실패")
        print("✅ DB에 성공적으로 연결되었습니다.")
    except Exception as e:
        print(f"⚠️ DB 연결 실패: {e}. 작업을 중단합니다.")
        return

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="DB 저장 진행률"):
        region_name, lat, lon = row['region'], row['lat'], row['lng']

        try:
            features_to_store = get_features_for_db(lat, lon)
            if not features_to_store:
                print(f"Warning: {region_name} 지역의 피처 생성에 실패하여 DB에 저장하지 않습니다.")
                continue

            # 1. 이름 정규화를 위한 매핑 적용
            special_map = {
                'T2M_0H_PAST': 'T2M_0H', 'RH2M_0H_PAST': 'RH2M_0H', 'WS10M_0H_PAST': 'WS10M_0H',
                'WD10M_0H_PAST': 'WD10M_0H', 'PREC_0H_PAST': 'PRECTOTCORR_0H', 'PS_0H_PAST': 'PS_0H',
                'SOLAR_0H_PAST': 'ALLSKY_SFC_SW_DWN_0H', 'FFMC_0H_PAST': 'FFMC_0H',
                'DMC_0H_PAST': 'DMC_0H', 'DC_0H_PAST': 'DC_0H', 'ISI_0H_PAST': 'ISI_0H',
                'BUI_0H_PAST': 'BUI_0H', 'FWI_0H_PAST': 'FWI_0H'
            }
            mapped_features = {}
            for key, value in features_to_store.items():
                key_upper = str(key).upper()
                mapped_key = special_map.get(key_upper, key_upper)
                mapped_features[mapped_key] = value

            # 기본 정보 추가
            mapped_features['REGION_NAME'] = region_name
            mapped_features['LAT'] = lat
            mapped_features['LNG'] = lon

            # 기존 insert_mountain_features 함수 사용
            db.insert_mountain_features(mapped_features)

        except Exception as e:
            print(f"❌ {region_name} 지역 처리 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()

    if db:
        db.close()
        print("\n✅ DB 연결이 해제되었습니다.")

    print(f"\n🎉 모든 작업 완료! {len(df)}개의 지역 데이터가 DB에 업데이트되었습니다.")


if __name__ == "__main__":
    main()

