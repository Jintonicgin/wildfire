import sys
import json
import joblib
import datetime
import numpy as np
import pandas as pd
import ee
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import math

sys.path.append("/Users/mmymacymac/Developer/Projects/eclipse/WildFire/dataset/")
from fetch_all_weather import fetch_all_weather_features

warnings.filterwarnings("ignore")

MODEL_PATH = "/Users/mmymacymac/Developer/Projects/eclipse/WildFire/dataset/"

class EnsembleRegressor:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        preds = [model.predict(X) for model in self.models]
        return np.mean(np.column_stack(preds), axis=1)

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
        "speed_model": joblib.load(MODEL_PATH + "speed_classifier_model.joblib"),
        "direction_model": joblib.load(MODEL_PATH + "direction_classifier_model.joblib"),
        "area_scaler": joblib.load(MODEL_PATH + "area_model_scaler_v2.joblib"),
        "speed_scaler": joblib.load(MODEL_PATH + "speed_model_scaler.joblib"),
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

        ndvi_img = ee.ImageCollection("MODIS/061/MOD13Q1") \
            .filterBounds(point).filterDate(start, end) \
            .sort('system:time_start', False).first()

        ndvi = ndvi_img.select('NDVI').reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=250,
            maxPixels=1e8
        ).getInfo()

        treecover = ee.Image("UMD/hansen/global_forest_change_2022_v1_10") \
            .select("treecover2000") \
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=30,
                maxPixels=1e8
            ).getInfo()

        buffer = point.buffer(500) 

        elev_img = ee.Image("USGS/SRTMGL1_003")
        slope_img = ee.Terrain.slope(elev_img)
        aspect_img = ee.Terrain.aspect(elev_img)

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

def move_coordinate(lat, lon, distance_m, bearing_deg):
    R = 6378137  
    d = distance_m
    brng = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    lat2 = math.asin(math.sin(lat1) * math.cos(d / R) +
                     math.cos(lat1) * math.sin(d / R) * math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(d / R) * math.cos(lat1),
                           math.cos(d / R) - math.sin(lat1) * math.sin(lat2))

    return math.degrees(lat2), math.degrees(lon2)

def predict_single_timestep(lat, lon, timestamp, gee_features, simulation_hours_total):
    _initialize_prediction_environment() 

    base = {"startyear": timestamp.year, "startmonth": timestamp.month, "startday": timestamp.day}
    weather_features = fetch_all_weather_features(lat, lon, timestamp, offset_days=4)
    season = add_season_flags(timestamp)

    all_features = {
        **weather_features, 
        **base,
        **gee_features,
        **season,
        "lat": lat,
        "lng": lon,
        "duration_hours": float(simulation_hours_total), 
        "total_duration_hours": float(simulation_hours_total), 
        "duration_x_ws10m": weather_features.get("duration_hours", 0) * weather_features.get("WS10M", 0),
        "duration_x_t2m": weather_features.get("duration_hours", 0) * weather_features.get("T2M", 0),
        "duration_x_rh2m": weather_features.get("duration_hours", 0) * weather_features.get("RH2M", 0),
        "duration_x_fwi": weather_features.get("duration_hours", 0) * weather_features.get("FWI", 0),
        "duration_x_isi": weather_features.get("duration_hours", 0) * weather_features.get("ISI", 0),
    }

    df = pd.DataFrame([all_features])

    area_input = df[MODELS["area_cols"]].copy().fillna(0)
    speed_input = df[MODELS["speed_cols"]].copy().fillna(0)
    direction_input = df[MODELS["direction_cols"]].copy().fillna(0)

    area_scaled = MODELS["area_scaler"].transform(area_input)
    speed_scaled = MODELS["speed_scaler"].transform(speed_input)

    area_log = MODELS["area_model"].predict(area_scaled)[0]
    area = float(np.expm1(area_log))
    if not np.isfinite(area) or area < 0: area = 0.0

    spread_dist_m = float(np.sqrt(area * 10000 / np.pi)) 
    speed_cat = int(MODELS["speed_model"].predict(speed_scaled)[0])
    direction_class = str(MODELS["direction_model"].predict(direction_input)[0])

    return {
        "hourly_damage_area": area,
        "spread_speed_category": speed_cat,
        "spread_direction": direction_class,
        "predicted_distance_m": spread_dist_m,
        "wind_direction_deg": float(weather_features.get("WD10M_0h", -999))
    }

def predict_simulation(input_json):
    _initialize_prediction_environment() 

    current_lat = input_json["latitude"]
    current_lon = input_json["longitude"]
    start_timestamp = datetime.datetime.fromisoformat(input_json["timestamp"])
    simulation_hours = input_json.get("durationHours", 1)

    total_damage_area = 0
    path_trace = []

    initial_gee_features = get_gee_features(current_lat, current_lon)

    for hour in range(simulation_hours):
        current_timestamp = start_timestamp + datetime.timedelta(hours=hour)
        
        timestep_result = predict_single_timestep(current_lat, current_lon, current_timestamp, initial_gee_features, simulation_hours)
        
        total_damage_area += timestep_result["hourly_damage_area"]
        
       
        distance_m = timestep_result["predicted_distance_m"]
        direction_deg = float(timestep_result["wind_direction_deg"])
        
        path_trace.append({
            "hour": hour + 1,
            "lat": current_lat,
            "lon": current_lon,
            "hourly_damage_area": timestep_result["hourly_damage_area"],
            "cumulative_damage_area": total_damage_area,
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
        import traceback
        print(json.dumps({"error": f"예측 실패: {str(e)}", "traceback": traceback.format_exc()}))
