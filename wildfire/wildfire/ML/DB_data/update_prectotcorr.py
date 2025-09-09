import os
import sys
import pandas as pd
from tqdm import tqdm
import datetime
import warnings
import joblib # For np.float32/64, pd.isna
import numpy as np # For np.float32/64
import ee

# --- 경로 설정 및 모듈 임포트 ---
# mountain_db.py와 동일한 임포트 경로 설정
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(DATASET_PATH)

try:
    from wildfire.ML.fetch_all_weather import fetch_all_weather_features
    from wildfire.ML.predict import get_gee_features
    from wildfire.ML.DB_data.oracle_db import OracleDB
except ImportError as e:
    print(f"⚠️ 모듈 임포트 오류: {e}")
    sys.path.append(os.path.dirname(__file__))
    from wildfire.ML.DB_data.oracle_db import OracleDB
    from wildfire.ML.fetch_all_weather import fetch_all_weather_features
    from wildfire.ML.predict import get_gee_features

warnings.filterwarnings("ignore")

# --- mountain_db.py에서 필요한 함수들 복사 (임시 스크립트용) ---
# 실제 프로젝트에서는 이 함수들을 별도 유틸리티 파일로 분리하여 임포트하는 것이 좋습니다.

def generate_statistical_features(weather_features):
    """시간별 날씨 데이터로부터 통계 피처들을 생성합니다."""
    stats_features = {}
    weather_params = ['t2m', 'rh2m', 'ws10m', 'wd10m', 'prec', 'ps', 'solar', 'ws2m', 'wd2m']
    
    for param in weather_params:
        all_values = []
        day_values = []
        for hour in range(0, 169, 3):
            key = f"{param}_{hour}h_past"
            if key in weather_features:
                val = weather_features[key]
                if val is not None and pd.notna(val):
                    all_values.append(float(val))
                    if hour <= 24:
                        day_values.append(float(val))
        
        for values, suffix in [(all_values, "_past"), (day_values, "_24h_past")]:
            if values:
                mean_val = sum(values) / len(values)
                stats_features[f"{param}_max{suffix}"] = max(values)
                stats_features[f"{param}_min{suffix}"] = min(values)
                stats_features[f"{param}_mean{suffix}"] = mean_val
                stats_features[f"{param}_std{suffix}"] = (sum([(x - mean_val)**2 for x in values]) / len(values))**0.5
            else:
                for stat in ['max', 'min', 'mean', 'std']:
                    stats_features[f"{param}_{stat}{suffix}"] = -999
    return stats_features

def generate_custom_features(weather_features, gee_features):
    """커스텀 화재 위험 피처들을 생성합니다."""
    custom_features = {}
    t2m_0h = weather_features.get('t2m_0h_past', 20)
    rh2m_0h = weather_features.get('rh2m_0h_past', 50)
    ws10m_0h = weather_features.get('ws10m_0h_past', 5)
    wd10m_0h = weather_features.get('wd10m_0h_past', 180)
    prec_0h = weather_features.get('prec_0h_past', 0)
    slope_mean = gee_features.get('slope_mean', 5)
    slope_std = gee_features.get('slope_std', 2) 
    aspect_mode = gee_features.get('aspect_mode', 180)
    elevation_mean = gee_features.get('elevation_mean', 300)
    elevation_std = gee_features.get('elevation_std', 50)
    ndvi = gee_features.get('ndvi_before', 0.5)
    treecover = gee_features.get('treecover_pre_fire_5x5', 50)
    wind_slope_angle = abs(wd10m_0h - aspect_mode)
    if wind_slope_angle > 180: wind_slope_angle = 360 - wind_slope_angle
    custom_features['wind_slope_interaction'] = float(ws10m_0h * slope_mean * (1 - wind_slope_angle/180))
    expected_temp_at_elevation = t2m_0h - (elevation_mean * 0.0065)
    custom_features['elevation_temp_deviation'] = float(t2m_0h - expected_temp_at_elevation)
    custom_features['terrain_complexity_index'] = float((slope_std / max(slope_mean, 1)) + (elevation_std / max(elevation_mean, 1)))
    if elevation_mean < 200 and rh2m_0h > 70: custom_features['valley_moisture_trap'] = 1
    else: custom_features['valley_moisture_trap'] = 0
    temp_stress = max(0, (t2m_0h - 25) / 15) if t2m_0h > 25 else 0
    ndvi_stress = max(0, (0.6 - ndvi) / 0.6) if ndvi < 0.6 else 0
    custom_features['vegetation_heat_stress'] = float(temp_stress * ndvi_stress)
    total_precip_7d = weather_features.get('total_prec_7d_start_past', 0)
    drought_stress = max(0, (10 - total_precip_7d) / 10) if total_precip_7d < 10 else 0
    custom_features['vegetation_drought_stress'] = float(drought_stress * ndvi_stress)
    if treecover > 70: expected_humidity = rh2m_0h + 10
    else: expected_humidity = rh2m_0h
    custom_features['forest_humidity_effect'] = float(expected_humidity - rh2m_0h)
    dry_factor = max(0, (100 - rh2m_0h) / 100)
    wind_factor = min(ws10m_0h / 20.0, 1.0)
    terrain_factor = min(slope_mean / 30.0, 1.0)
    custom_features['dry_windy_combo'] = float(dry_factor * wind_factor * (1 + terrain_factor))
    if ndvi < 0.3 and rh2m_0h < 40: fuel_score = 1.0
    elif treecover > 50 and rh2m_0h < 50 and total_precip_7d < 5: fuel_score = 0.8
    elif ndvi > 0.6 and rh2m_0h > 60: fuel_score = 0.2
    else: fuel_score = 0.5
    custom_features['fuel_combo'] = float(fuel_score)
    spread_index = (wind_factor * 0.3 + dry_factor * 0.3 + temp_stress * 0.2 + terrain_factor * 0.2)
    custom_features['potential_spread_index'] = float(spread_index)
    custom_features['terrain_var_effect'] = float(terrain_factor * (1 + slope_std/10))
    wind_values = []
    for hour in range(0, 25, 3):
        key = f"ws10m_{hour}h_past"
        if key in weather_features and weather_features[key] is not None:
            wind_values.append(weather_features[key])
    if len(wind_values) > 3:
        wind_mean = sum(wind_values) / len(wind_values)
        wind_std = (sum([(x - wind_mean)**2 for x in wind_values]) / len(wind_values))**0.5
        custom_features['wind_steady_flag'] = 1 if wind_std < 2.0 else 0
        custom_features['wind_strength_consistency'] = float(wind_mean * (1 - wind_std/wind_mean) if wind_mean > 0 else 0)
    else:
        custom_features['wind_steady_flag'] = 0
        custom_features['wind_strength_consistency'] = 0
    total_precip_30d = weather_features.get('total_prec_30d_start_past', 0)
    dry_days_30d = weather_features.get('dry_days_30d_start_past', 30)
    if total_precip_30d > 0: custom_features['dry_to_rain_ratio_30d'] = float(dry_days_30d / max(total_precip_30d, 0.1))
    else: custom_features['dry_to_rain_ratio_30d'] = 30.0
    base_risk = (dry_factor + wind_factor + temp_stress) / 3
    terrain_amplifier = 1 + terrain_factor * 0.5
    vegetation_modifier = 1 - (ndvi * 0.3) if ndvi > 0.5 else 1 + (0.5 - ndvi) * 0.5
    custom_features['integrated_fire_risk_score'] = float(base_risk * terrain_amplifier * vegetation_modifier)
    return custom_features

def get_features_for_db(lat, lon):
    """예측 시점(현재)을 기준으로 DB에 저장할 모든 피처를 생성합니다."""
    timestamp = datetime.datetime.now()
    
    print(f"\n--- 좌표 ({lat:.4f}, {lon:.4f})에 대한 데이터 수집 시작 ---")
    gee_features = get_gee_features(lat, lon)
    if not gee_features: gee_features = {}

    weather_features = fetch_all_weather_features(lat, lon, timestamp, offset_days=10)
    if not weather_features or not weather_features.get("success"):
        print(f"Warning: 날씨 피처 수집 실패 ({lat}, {lon})")
        return None

    print("📊 통계 피처 생성 중...")
    stats_features = generate_statistical_features(weather_features)
    print(f"✅ 통계 피처 {len(stats_features)}개 생성 완료")
    
    print("🎯 커스텀 위험 피처 생성 중...")
    custom_features = generate_custom_features(weather_features, gee_features)
    print(f"✅ 커스텀 피처 {len(custom_features)}개 생성 완료")

    all_features = {**gee_features, **weather_features, **stats_features, **custom_features}
    all_features.update({
        "lat": lat,
        "lng": lon
    })
    
    all_features.pop('success', None)
    print(f"🎉 총 피처 생성 완료: {len(all_features)}개 피처 (기존 549개 + 추가 79개)")
    return all_features

# --- 메인 실행 로직 ---
def main():
    try:
        ee.Initialize(project='wildfire-464907')
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        source_csv_path = os.path.join(script_dir, "..", "gangwon_mountain_points.csv")

        df = pd.read_csv(source_csv_path)
        print(f"\n✅ '{source_csv_path}' 파일을 성공적으로 읽었습니다. 총 {len(df)}개의 좌표를 처리합니다.")

        db = OracleDB()
        if not db.conn:
            raise Exception("DB 연결 실패")
        print("✅ DB에 성공적으로 연결되었습니다.")

        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="PRECTOTCORR_0H 업데이트 진행률"):
            region_name = row['region']
            lat = row['lat']
            lon = row['lng']

            try:
                features_for_update = get_features_for_db(lat, lon)
                if not features_for_update:
                    print(f"Warning: {region_name} 지역의 피처 생성에 실패하여 PRECTOTCORR_0H를 업데이트하지 않습니다.")
                    continue

                # Apply special_map to get PRECTOTCORR_0H
                # This part is crucial to get the value in the correct mapped name
                special_map = {
                    'PREC_0H_PAST': 'PRECTOTCORR_0H',
                }
                
                prectotcorr_value = None
                # Iterate through features_for_update to find the original PREC_0H_PAST value
                for key, value in features_for_update.items():
                    if str(key).upper() == 'PREC_0H_PAST':
                        prectotcorr_value = value
                        break
                
                if prectotcorr_value is not None:
                    # Ensure value is float or None for DB
                    if isinstance(prectotcorr_value, (np.float32, np.float64)):
                        prectotcorr_value = float(prectotcorr_value)
                    elif pd.isna(prectotcorr_value):
                        prectotcorr_value = None
                    
                    # Call the new update function
                    db.update_prectotcorr_0h(region_name, prectotcorr_value)
                else:
                    print(f"Warning: {region_name}에 대한 PREC_0H_PAST 값을 찾을 수 없습니다. PRECTOTCORR_0H를 업데이트하지 않습니다.")

            except Exception as e:
                print(f"❌ {region_name} 지역 PRECTOTCORR_0H 업데이트 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()

    except Exception as e:
        print(f"⚠️ 스크립트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if db:
            db.close()
        print("\n✅ 모든 PRECTOTCORR_0H 업데이트 작업 완료.")

if __name__ == "__main__":
    main()
