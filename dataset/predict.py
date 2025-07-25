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

# GEE 초기화
try:
    ee.Initialize(project='wildfire-464907')
except Exception:
    ee.Authenticate()
    ee.Initialize(project='wildfire-464907')

MODEL_PATH = "/Users/mmymacymac/Developer/Projects/eclipse/WildFire/dataset/"

# --- 모델 및 데이터 로드 ---
def load_models():
    models = {
        "area_model": joblib.load(MODEL_PATH + "area_regressor_model_v3.joblib"),
        "speed_model": joblib.load(MODEL_PATH + "speed_classifier_model.joblib"),
        "direction_model": joblib.load(MODEL_PATH + "direction_classifier_model.joblib"),
        "area_scaler": joblib.load(MODEL_PATH + "area_model_scaler_v3.joblib"),
        "speed_scaler": joblib.load(MODEL_PATH + "speed_model_scaler.joblib"),
    }
    with open(MODEL_PATH + "area_model_columns_v3.json") as f:
        models["area_cols"] = json.load(f)
    with open(MODEL_PATH + "speed_model_columns.json") as f:
        models["speed_cols"] = json.load(f)
    with open(MODEL_PATH + "direction_model_columns.json") as f:
        models["direction_cols"] = json.load(f)
    return models

MODELS = load_models()

# --- 지리 정보 및 계절 피처 --- (이전과 동일)
def get_gee_features(lat, lon):
    point = ee.Geometry.Point([lon, lat])
    today = datetime.date.today()
    end = today - datetime.timedelta(days=5)
    start = end - datetime.timedelta(days=30)

    ndvi = ee.ImageCollection("MODIS/061/MOD13Q1") \
        .filterBounds(point) \
        .filterDate(str(start), str(end)) \
        .sort('system:time_start', False) \
        .first() \
        .select('NDVI') \
        .reduceRegion(ee.Reducer.mean(), point, 250) \
        .getInfo()

    treecover = ee.Image("UMD/hansen/global_forest_change_2022_v1_10") \
        .select("treecover2000") \
        .reduceRegion(ee.Reducer.mean(), point, 30) \
        .getInfo()

    elev_img = ee.Image("USGS/SRTMGL1_003").clip(point.buffer(500))
    slope_img = ee.Terrain.slope(elev_img)
    aspect_img = ee.Terrain.aspect(elev_img)

    elev_stats = elev_img.reduceRegion(ee.Reducer.minMax().combine(
        reducer2=ee.Reducer.mean().combine(
            reducer2=ee.Reducer.stdDev(), sharedInputs=True), sharedInputs=True),
        point, 90).getInfo()

    slope_stats = slope_img.reduceRegion(ee.Reducer.minMax().combine(
        reducer2=ee.Reducer.mean().combine(
            reducer2=ee.Reducer.stdDev(), sharedInputs=True), sharedInputs=True),
        point, 90).getInfo()

    aspect_stats = aspect_img.reduceRegion(ee.Reducer.mode().combine(
        reducer2=ee.Reducer.stdDev(), sharedInputs=True),
        point, 90).getInfo()

    return {
        "ndvi_before": float(ndvi.get("NDVI", -999)) / 10000 if ndvi.get("NDVI") else -999,
        "treecover_pre_fire_5x5": float(treecover.get("treecover2000", -999)),
        "elevation_mean": float(elev_stats.get("elevation_mean", -999)),
        "elevation_min": float(elev_stats.get("elevation_min", -999)),
        "elevation_max": float(elev_stats.get("elevation_max", -999)),
        "elevation_std": float(elev_stats.get("elevation_stdDev", -999)),
        "slope_mean": float(slope_stats.get("slope_mean", -999)),
        "slope_min": float(slope_stats.get("slope_min", -999)),
        "slope_max": float(slope_stats.get("slope_max", -999)),
        "slope_std": float(slope_stats.get("slope_stdDev", -999)),
        "aspect_mode": float(aspect_stats.get("aspect_mode", -999)),
        "aspect_std": float(aspect_stats.get("aspect_stdDev", -999)),
    }

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
def predict_single_timestep(lat, lon, timestamp):
    base = {"startyear": timestamp.year, "startmonth": timestamp.month, "startday": timestamp.day}
    weather = fetch_all_weather_features(lat, lon, timestamp, offset_days=4)
    gee = get_gee_features(lat, lon)
    season = add_season_flags(timestamp)

    all_features = {**base, **gee, **season, **weather}
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
        "wind_direction_deg": float(weather.get("WD10M_0h", -999))
    }

# --- 시뮬레이션 예측 함수 ---
def predict_simulation(input_json):
    current_lat = input_json["latitude"]
    current_lon = input_json["longitude"]
    start_timestamp = datetime.datetime.fromisoformat(input_json["timestamp"])
    simulation_hours = input_json.get("durationHours", 1)

    total_damage_area = 0
    path_trace = []

    for hour in range(simulation_hours):
        current_timestamp = start_timestamp + datetime.timedelta(hours=hour)
        
        # 현재 시간 단계 예측
        timestep_result = predict_single_timestep(current_lat, current_lon, current_timestamp)
        
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
    try:
        input_data = json.loads(sys.stdin.read())
        result = predict_simulation(input_data)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(json.dumps({"error": f"예측 실패: {str(e)}", "traceback": traceback.format_exc()}))
