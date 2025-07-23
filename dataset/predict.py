import sys
import json
import joblib
import datetime
import numpy as np
import pandas as pd
import ee
import warnings

# 외부 모듈
sys.path.append("/Users/mmymacymac/Developer/Projects/eclipse/WildFire/dataset/")
from fetch_all_weather import fetch_all_weather_features
from fwi_calc import fwi_calc

warnings.filterwarnings("ignore")

# ✅ GEE 초기화
try:
    ee.Initialize(project='wildfire-464907')
except Exception:
    ee.Authenticate()
    ee.Initialize(project='wildfire-464907')

MODEL_PATH = "/Users/mmymacymac/Developer/Projects/eclipse/WildFire/dataset/"

def load_models():
    models = {
        "area_model": joblib.load(MODEL_PATH + "area_regressor_model.joblib"),
        "speed_model": joblib.load(MODEL_PATH + "speed_classifier_model.joblib"),
        "direction_model": joblib.load(MODEL_PATH + "direction_classifier_model.joblib"),
        "scaler": joblib.load(MODEL_PATH + "speed_model_scaler.joblib"),
    }
    with open(MODEL_PATH + "area_model_columns.json") as f:
        models["area_cols"] = json.load(f)
    with open(MODEL_PATH + "speed_model_columns.json") as f:
        models["speed_cols"] = json.load(f)
    with open(MODEL_PATH + "direction_model_columns.json") as f:
        models["direction_cols"] = json.load(f)
    return models

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

def prepare_features(df, models):
    df = df.copy()
    for category, key in [("area", "area_cols"), ("speed", "speed_cols"), ("direction", "direction_cols")]:
        expected = set(models[key])
        actual = set(df.columns)
        if not expected.issubset(actual):
            missing = sorted(list(expected - actual))
            raise ValueError(f"{category} 모델에 필요한 컬럼이 누락됨: {missing}")

    area_input = df[models["area_cols"]].copy().fillna(0)
    speed_input = df[models["speed_cols"]].copy().fillna(0)
    direction_input = df[models["direction_cols"]].copy().fillna(0)
    speed_scaled = models["scaler"].transform(speed_input)
    return area_input, speed_scaled, direction_input

def predict(input_json):
    lat = input_json["latitude"]
    lon = input_json["longitude"]
    timestamp = datetime.datetime.fromisoformat(input_json["timestamp"])
    duration_hours = input_json.get("durationHours", 1)

    base = {
        "startyear": timestamp.year,
        "startmonth": timestamp.month,
        "startday": timestamp.day
    }

    weather = fetch_all_weather_features(lat, lon, timestamp)
    gee = get_gee_features(lat, lon)
    fwi = fwi_calc(
        T=weather.get("T2M_0h", 20),
        RH=weather.get("RH2M_0h", 40),
        W=weather.get("WS10M_0h", 3),
        P=weather.get("PRECTOTCORR_0h", 0),
        month=timestamp.month
    )

    season_flags = add_season_flags(timestamp)

    all_features = {**base, **weather, **gee, **fwi, **season_flags}
    df = pd.DataFrame([all_features])

    models = load_models()
    area_input, speed_input, direction_input = prepare_features(df, models)

    area_log = models["area_model"].predict(area_input)[0]
    area = float(np.expm1(area_log))
    if not np.isfinite(area) or area < 0:
        area = 0.0

    spread_distance_m = float(np.sqrt(area * 10000 / np.pi))
    speed_category = int(models["speed_model"].predict(speed_input)[0])
    direction_class = str(models["direction_model"].predict(direction_input)[0])

    return {
        "damage_area": area,
        "spread_speed_category": speed_category,
        "spread_direction": direction_class,
        "predicted_distance_m": spread_distance_m,
        "wind_direction_deg": float(weather.get("WD10M_0h", -999))
    }

if __name__ == "__main__":
    try:
        input_data = json.loads(sys.stdin.read())
        result = predict(input_data)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": f"예측 실패: {str(e)}"}))