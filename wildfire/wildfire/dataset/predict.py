import sys
import json
import warnings
import datetime
import numpy as np
import pandas as pd
import joblib
import ee
import math
import traceback

from wildfire.dataset.fetch_all_weather import fetch_and_engineer_features
from wildfire.dataset import model_definitions

warnings.filterwarnings("ignore")
sys.modules['model_definitions'] = model_definitions

MODEL_PATH = "/Users/mmymacymac/Developer/Projects/WildFire_projects/wildfire/wildfire/dataset/"
MODELS = None
_initialized = False


def _initialize_prediction_environment():
    global MODELS, _initialized
    if _initialized:
        return

    try:
        ee.Initialize(project='wildfire-464907')
    except Exception:
        ee.Authenticate()
        ee.Initialize(project='wildfire-464907')

    MODELS = {
        "area_model": joblib.load(MODEL_PATH + "area_regressor_model_v2.joblib"),
        "speed_model": joblib.load(MODEL_PATH + "models/speed_classifier_model.joblib"),
        "direction_model": joblib.load(MODEL_PATH + "models/direction_classifier_model.joblib"),
        "area_scaler": joblib.load(MODEL_PATH + "area_model_scaler_v2.joblib"),
        "speed_scaler": joblib.load(MODEL_PATH + "models/speed_scaler.joblib"),
        "direction_scaler": joblib.load(MODEL_PATH + "models/direction_scaler.joblib"),
    }
    with open(MODEL_PATH + "area_model_columns_v2.json") as f:
        MODELS["area_cols"] = json.load(f)
    with open(MODEL_PATH + "speed_model_columns.json") as f:
        MODELS["speed_cols"] = json.load(f)
    with open(MODEL_PATH + "direction_model_columns.json") as f:
        MODELS["direction_cols"] = json.load(f)

    _initialized = True


def get_gee_features(lat, lon):
    point = ee.Geometry.Point([lon, lat])
    try:
        today = ee.Date(datetime.datetime.utcnow().isoformat(timespec='seconds') + 'Z')
        end = today.advance(-5, 'day')
        start = end.advance(-30, 'day')

        ndvi_img = ee.ImageCollection("MODIS/061/MOD13Q1").filterBounds(point).filterDate(start, end).sort(
            'system:time_start', False).first()
        ndvi = ndvi_img.select('NDVI').reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=250,
                                                    maxPixels=1e8).getInfo()

        treecover = ee.Image("UMD/hansen/global_forest_change_2022_v1_10").select("treecover2000").reduceRegion(
            reducer=ee.Reducer.mean(), geometry=point, scale=30, maxPixels=1e8).getInfo()

        buffer = point.buffer(500)
        elev_img = ee.Image("USGS/SRTMGL1_003")
        slope_img = ee.Terrain.slope(elev_img)
        aspect_img = ee.Terrain.aspect(elev_img)

        reducer = ee.Reducer.mean().combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True).combine(
            reducer2=ee.Reducer.minMax(), sharedInputs=True)

        elev_stats = elev_img.reduceRegion(reducer=reducer, geometry=buffer, scale=90, maxPixels=1e8).getInfo()
        slope_stats = slope_img.reduceRegion(reducer=reducer, geometry=buffer, scale=90, maxPixels=1e8).getInfo()
        aspect_stats = aspect_img.reduceRegion(
            reducer=ee.Reducer.mode().combine(ee.Reducer.stdDev(), sharedInputs=True), geometry=buffer, scale=90,
            maxPixels=1e8).getInfo()

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
    except Exception:
        return {k: -999 for k in [
            "ndvi_before", "treecover_pre_fire_5x5", "elevation_mean", "elevation_min", "elevation_max",
            "elevation_std",
            "slope_mean", "slope_min", "slope_max", "slope_std", "aspect_mode", "aspect_std"
        ]}


def move_coordinate(lat, lon, distance_m, bearing_deg):
    R = 6378137
    d = distance_m
    brng = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(math.sin(lat1) * math.cos(d / R) + math.cos(lat1) * math.sin(d / R) * math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(d / R) * math.cos(lat1),
                             math.cos(d / R) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)


def align_features_to_model(features, expected_columns):
    """학습 시 사용된 컬럼명과 일치하도록 피처를 정렬하고 누락된 피처는 0으로 채움"""
    aligned_features = {}

    # 예상 컬럼에 따라 피처 매핑
    for col in expected_columns:
        if col in features:
            aligned_features[col] = features[col]
        else:
            # 누락된 피처는 0으로 채움
            aligned_features[col] = 0.0
            print(f"⚠️  누락된 피처 '{col}'를 0으로 대체")

    return pd.DataFrame([aligned_features])


def predict_single_timestep(lat, lon, timestamp, gee_features, simulation_hours_total):
    _initialize_prediction_environment()

    # --- Speed/Direction 모델을 위한 상세 피처 생성 ---
    detailed_features = fetch_and_engineer_features(lat, lon, timestamp)
    detailed_features.update(gee_features)
    detailed_features['lat'] = lat
    detailed_features['lng'] = lon
    detailed_features['duration_hours'] = float(simulation_hours_total)
    detailed_features['total_duration_hours'] = float(simulation_hours_total)

    speed_input_df = align_features_to_model(detailed_features, MODELS["speed_cols"])
    direction_input_df = align_features_to_model(detailed_features, MODELS["direction_cols"])

    # --- Area 모델을 위한 최소 피처 생성 ---
    # area_model_columns_v2.json에 정의된 피처만 사용
    area_features = {
        'lat': lat,
        'lng': lon,
        'duration_hours': float(simulation_hours_total),
        'total_duration_hours': float(simulation_hours_total),
        'T2M': detailed_features.get('T2M_0h', 0),  # 현재 기온
        'RH2M': detailed_features.get('RH2M_0h', 0), # 현재 습도
        'WS10M': detailed_features.get('WS10M_0h', 0),# 현재 풍속
        'WD10M': detailed_features.get('WD10M_0h', 0),# 현재 풍향
        'PRECTOTCORR': detailed_features.get('PRECTOTCORR_0h', 0), # 현재 강수량
        'FFMC': detailed_features.get('FFMC', 0),
        'DMC': detailed_features.get('DMC', 0),
        'DC': detailed_features.get('DC', 0),
        'ISI': detailed_features.get('ISI', 0),
        'BUI': detailed_features.get('BUI', 0),
        'FWI': detailed_features.get('FWI', 0),
    }
    area_input_df = align_features_to_model(area_features, MODELS["area_cols"])

    # 스케일링
    area_scaled = MODELS["area_scaler"].transform(area_input_df)
    speed_scaled = MODELS["speed_scaler"].transform(speed_input_df)
    direction_scaled = MODELS["direction_scaler"].transform(direction_input_df)

    area_log = MODELS["area_model"].predict(area_scaled)[0]
    area = float(np.expm1(area_log))
    if not np.isfinite(area) or area < 0:
        area = 0.0

    speed_cat = int(MODELS["speed_model"].predict(speed_scaled)[0])
    direction_class = str(MODELS["direction_model"].predict(direction_scaled)[0])

    return {
        "hourly_damage_area": area,
        "spread_speed_category": speed_cat,
        "spread_direction": direction_class,
        "predicted_distance_m": float(np.sqrt(area * 10000 / np.pi)),
        "wind_direction_deg": float(detailed_features.get("WD10M_0h", -999))
    }


def predict_simulation(input_json):
    _initialize_prediction_environment()
    current_lat = input_json["latitude"]
    current_lon = input_json["longitude"]
    start_timestamp = datetime.datetime.fromisoformat(input_json["timestamp"])
    simulation_hours = input_json.get("durationHours", 1)
    total_damage_area = 0
    path_trace = []
    gee_features = get_gee_features(current_lat, current_lon)

    for hour in range(simulation_hours):
        current_timestamp = start_timestamp + datetime.timedelta(hours=hour)
        timestep_result = predict_single_timestep(current_lat, current_lon, current_timestamp, gee_features,
                                                  simulation_hours)
        
        # 🔧 수정: 면적을 누적하는 대신, 매 시간의 예측 결과로 업데이트
        # area_model 자체가 총 시간을 입력받아 총 면적을 예측하므로 누적은 불필요
        total_damage_area = timestep_result["hourly_damage_area"]

        distance_m = timestep_result["predicted_distance_m"]
        direction_deg = float(timestep_result["wind_direction_deg"])
        path_trace.append({
            "hour": hour + 1,
            "lat": current_lat,
            "lon": current_lon,
            "hourly_damage_area": timestep_result["hourly_damage_area"],
            "cumulative_damage_area": total_damage_area, # 이 값은 이제 해당 시간까지의 총 면적을 의미
            "wind_direction_deg": direction_deg
        })
        current_lat, current_lon = move_coordinate(current_lat, current_lon, distance_m, direction_deg)

    return {
        "simulation_hours": simulation_hours,
        "final_damage_area": total_damage_area,
        "final_lat": current_lat,
        "final_lon": current_lon,
        "path_trace": path_trace
    }


if __name__ == "__main__":
    _initialize_prediction_environment()
    try:
        input_data = json.loads(sys.stdin.read())
        result = predict_simulation(input_data)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e), "traceback": traceback.format_exc()}))