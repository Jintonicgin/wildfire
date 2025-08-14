import sys
import json
import joblib
import datetime
import numpy as np
import pandas as pd
import ee
import warnings
import math

# 외부 모듈
sys.path.append("/Users/mmymacymac/Developer/Projects/eclipse/WildFire/dataset/")
from fetch_all_weather import fetch_all_weather_features

warnings.filterwarnings("ignore")

MODEL_PATH = "/Users/mmymacymac/Developer/Projects/eclipse/WildFire/dataset/"

# 전역 변수 및 초기화 플래그
MODELS = None
_initialized = False

# --- 초기화 함수 ---
def _initialize_prediction_environment():
    global MODELS, _initialized
    if _initialized: # 이미 초기화되었다면 건너뛰기
        return

    # GEE 초기화
    try:
        ee.Initialize(project='wildfire-464907')
    except Exception:
        ee.Authenticate()
        ee.Initialize(project='wildfire-464907')

    # 모델 및 스케일러 로드
    MODELS = {
        "area_model": joblib.load(MODEL_PATH + "area_regressor_model_v2.joblib"),
        "speed_model": joblib.load(MODEL_PATH + "speed_classifier_model.joblib"),
        "direction_model": joblib.load(MODEL_PATH + "direction_classifier_model.joblib"),
        "area_scaler": joblib.load(MODEL_PATH + "area_model_scaler_v2.joblib"),
        "speed_scaler": joblib.load(MODEL_PATH + "speed_model_scaler.joblib"),
    }
    with open(MODEL_PATH + "area_model_columns_v2.json") as f: # v2 모델 컬럼 로드
        MODELS["area_cols"] = json.load(f)
    with open(MODEL_PATH + "speed_model_columns.json") as f:
        MODELS["speed_cols"] = json.load(f)
    with open(MODEL_PATH + "direction_model_columns.json") as f:
        MODELS["direction_cols"] = json.load(f)
    
    _initialized = True

# --- 지리 정보 및 계절 피처 ---
def get_gee_features(lat, lon):
    point = ee.Geometry.Point([lon, lat])

    try:
        # 기간 설정 (NDVI용)
        today = ee.Date(datetime.datetime.utcnow().isoformat(timespec='seconds') + 'Z')
        end = today.advance(-5, 'day')
        start = end.advance(-30, 'day')

        # NDVI (MODIS)
        ndvi_img = ee.ImageCollection("MODIS/061/MOD13Q1") \
            .filterBounds(point).filterDate(start, end) \
            .sort('system:time_start', False).first()

        ndvi = ndvi_img.select('NDVI').reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=250,
            maxPixels=1e8
        ).getInfo()

        # Treecover
        treecover = ee.Image("UMD/hansen/global_forest_change_2022_v1_10") \
            .select("treecover2000") \
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=30,
                maxPixels=1e8
            ).getInfo()

        # Elevation, Slope, Aspect with larger buffer
        buffer = point.buffer(500) # 500m 버퍼로 주변 지역 통계 계산

        elev_img = ee.Image("USGS/SRTMGL1_003")
        slope_img = ee.Terrain.slope(elev_img)
        aspect_img = ee.Terrain.aspect(elev_img)

        # Reducer를 mean, stdDev, min, max 모두 계산하도록 수정
        reducer = ee.Reducer.mean().combine(
            reducer2=ee.Reducer.stdDev(),
            sharedInputs=True
        ).combine(
            reducer2=ee.Reducer.minMax(),
            sharedInputs=True
        )

        elev_stats = elev_img.reduceRegion(reducer=reducer, geometry=buffer, scale=90, maxPixels=1e8).getInfo()
        slope_stats = slope_img.reduceRegion(reducer=reducer, geometry=buffer, scale=90, maxPixels=1e8).getInfo()
        aspect_stats = aspect_img.reduceRegion(reducer=ee.Reducer.mode().combine(ee.Reducer.stdDev(), sharedInputs=True), geometry=buffer, scale=90, maxPixels=1e8).getInfo()

        # 🔍 로그 추가 (올바른 키 사용: 'mean', 'stdDev', 'min', 'max')

        return {
            "ndvi_before": float(ndvi.get("NDVI", -999)) / 10000 if ndvi.get("NDVI") else -999,
            "treecover_pre_fire_5x5": float(treecover.get("treecover2000", -999)),
            "elevation_mean": float(elev_stats.get("elevation_mean", -999)),
            "elevation_std": float(elev_stats.get("elevation_stdDev", -999)),
            "elevation_min": float(elev_stats.get("elevation_min", -999)),
            "elevation_max": float(elev_stats.get("elevation_max", -999)),
            "slope_mean": float(slope_stats.get("slope_mean", -999)),
            "slope_std": float(slope_stats.get("slope_stdDev", -999)),
            "slope_min": float(slope_stats.get("slope_min", -999)),
            "slope_max": float(slope_stats.get("slope_max", -999)),
            "aspect_mode": float(aspect_stats.get("aspect_mode", -999)),
            "aspect_std": float(aspect_stats.get("aspect_stdDev", -999)),
        }

    except Exception as e:
        return {k: -999 for k in [
            "ndvi_before", "treecover_pre_fire_5x5",
            "elevation_mean", "elevation_min", "elevation_max", "elevation_std",
            "slope_mean", "slope_min", "slope_max", "slope_std",
            "aspect_mode", "aspect_std"
        ]}

def add_season_flags(timestamp):
    month = timestamp.month
    return {
        "is_spring": int(month in [3, 4, 5]),
        "is_summer": int(month in [6, 7, 8]),
        "is_autumn": int(month in [9, 10, 11]),
        "is_winter": int(month in [12, 1, 2]),
    }

# --- 좌표 이동 함수 ---
def move_coordinate(lat, lon, distance_m, bearing_deg):
    R = 6378137  # 지구 반지름 (미터)
    d = distance_m
    brng = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    lat2 = math.asin(math.sin(lat1) * math.cos(d / R) +
                     math.cos(lat1) * math.sin(d / R) * math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(d / R) * math.cos(lat1),
                           math.cos(d / R) - math.sin(lat1) * math.sin(lat2))

    return math.degrees(lat2), math.degrees(lon2)

# --- 단일 시간 단계 예측 함수 ---
def predict_single_timestep(lat, lon, timestamp, gee_features, simulation_hours_total): # simulation_hours_total 인자 추가
    _initialize_prediction_environment() # 각 프로세스에서 한 번만 초기화

    base = {"startyear": timestamp.year, "startmonth": timestamp.month, "startday": timestamp.day}
    weather_features = fetch_all_weather_features(lat, lon, timestamp, offset_days=4)
    season = add_season_flags(timestamp)

    # all_features에 필요한 모든 피처를 명시적으로 포함
    all_features = {
        **weather_features, # weather_features의 모든 키-값을 언팩
        **base,
        **gee_features,
        **season,
        "lat": lat,
        "lng": lon,
        "duration_hours": float(simulation_hours_total), # 사용자가 요청한 총 시뮬레이션 시간 사용
        "total_duration_hours": float(simulation_hours_total), # area_v2 모델을 위한 복사본
        # duration_x_... 피처 계산 및 추가
        "duration_x_ws10m": weather_features.get("duration_hours", 0) * weather_features.get("WS10M", 0),
        "duration_x_t2m": weather_features.get("duration_hours", 0) * weather_features.get("T2M", 0),
        "duration_x_rh2m": weather_features.get("duration_hours", 0) * weather_features.get("RH2M", 0),
        "duration_x_fwi": weather_features.get("duration_hours", 0) * weather_features.get("FWI", 0),
        "duration_x_isi": weather_features.get("duration_hours", 0) * weather_features.get("ISI", 0),
    }

    df = pd.DataFrame([all_features])

    # 피처 준비
    area_input = df[MODELS["area_cols"]].copy().fillna(0)
    speed_input = df[MODELS["speed_cols"]].copy().fillna(0)
    direction_input = df[MODELS["direction_cols"]].copy().fillna(0)

    area_scaled = MODELS["area_scaler"].transform(area_input)
    speed_scaled = MODELS["speed_scaler"].transform(speed_input)

    # 예측
    area_log = MODELS["area_model"].predict(area_scaled)[0]
    area = float(np.expm1(area_log))
    if not np.isfinite(area) or area < 0: area = 0.0

    spread_dist_m = float(np.sqrt(area * 10000 / np.pi)) # 1시간 동안의 확산 거리
    speed_cat = int(MODELS["speed_model"].predict(speed_scaled)[0])
    direction_class = str(MODELS["direction_model"].predict(direction_input)[0])

    return {
        "hourly_damage_area": area,
        "spread_speed_category": speed_cat,
        "spread_direction": direction_class,
        "predicted_distance_m": spread_dist_m,
        "wind_direction_deg": float(weather_features.get("WD10M_0h", -999))
    }

# --- 시뮬레이션 예측 함수 ---
def predict_simulation(input_json):
    _initialize_prediction_environment() # 메인 프로세스에서 호출될 경우를 대비

    current_lat = input_json["latitude"]
    current_lon = input_json["longitude"]
    start_timestamp = datetime.datetime.fromisoformat(input_json["timestamp"])
    simulation_hours = input_json.get("durationHours", 1)

    total_damage_area = 0
    path_trace = []

    # GEE 피처를 시뮬레이션 시작 시 한 번만 가져옴
    initial_gee_features = get_gee_features(current_lat, current_lon)

    for hour in range(simulation_hours):
        current_timestamp = start_timestamp + datetime.timedelta(hours=hour)
        
        # 현재 시간 단계 예측 (GEE 피처 재사용) - simulation_hours를 전달
        timestep_result = predict_single_timestep(current_lat, current_lon, current_timestamp, initial_gee_features, simulation_hours)
        
        total_damage_area += timestep_result["hourly_damage_area"]
        
        # 다음 시간 단계의 위치 계산
        distance_m = timestep_result["predicted_distance_m"]
        direction_deg = float(timestep_result["wind_direction_deg"]) # 풍향을 확산 방향으로 가정
        
        path_trace.append({
            "hour": hour + 1,
            "lat": current_lat,
            "lon": current_lon,
            "hourly_damage_area": timestep_result["hourly_damage_area"],
            "cumulative_damage_area": total_damage_area,
            "wind_direction_deg": direction_deg
        })

        # 다음 위치로 이동
        current_lat, current_lon = move_coordinate(current_lat, current_lon, distance_m, direction_deg)

    return {
        "simulation_hours": simulation_hours,
        "final_damage_area": total_damage_area,
        "final_lat": current_lat,
        "final_lon": current_lon,
        "path_trace": path_trace
    }

if __name__ == "__main__":
    _initialize_prediction_environment() # 스크립트 직접 실행 시 초기화
    try:
        input_data = json.loads(sys.stdin.read())
        result = predict_simulation(input_data)
        print(json.dumps(result, indent=2))
    except Exception as e:
        import traceback
        print(json.dumps({"error": f"예측 실패: {str(e)}", "traceback": traceback.format_exc()}))
