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
import os

from wildfire.ML.fetch_all_weather import fetch_all_weather_features
from wildfire.ML import model_definitions

warnings.filterwarnings("ignore")
sys.modules['model_definitions'] = model_definitions

MODEL_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS = None
_initialized = False

DIRECTION_MAP = {0: 'N', 1: 'NE', 2: 'E', 3: 'SE', 4: 'S', 5: 'SW', 6: 'W', 7: 'NW'}
SPEED_CATEGORY_MAP = {0: 50.0, 1: 200.0, 2: 500.0}

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
        
        if all_values:
            stats_features[f"{param}_max_past"] = max(all_values)
            stats_features[f"{param}_min_past"] = min(all_values)
            mean_val = sum(all_values) / len(all_values)
            stats_features[f"{param}_mean_past"] = mean_val
            stats_features[f"{param}_std_past"] = (sum([(x - mean_val)**2 for x in all_values]) / len(all_values))**0.5
        else:
            stats_features[f"{param}_max_past"] = -999
            stats_features[f"{param}_min_past"] = -999
            stats_features[f"{param}_mean_past"] = -999
            stats_features[f"{param}_std_past"] = -999
            
        if day_values:
            stats_features[f"{param}_max_24h_past"] = max(day_values)
            stats_features[f"{param}_min_24h_past"] = min(day_values)
            mean_val = sum(day_values) / len(day_values)
            stats_features[f"{param}_mean_24h_past"] = mean_val
            stats_features[f"{param}_std_24h_past"] = (sum([(x - mean_val)**2 for x in day_values]) / len(day_values))**0.5
        else:
            stats_features[f"{param}_max_24h_past"] = -999
            stats_features[f"{param}_min_24h_past"] = -999
            stats_features[f"{param}_mean_24h_past"] = -999
            stats_features[f"{param}_std_24h_past"] = -999
    
    return stats_features

def generate_custom_features(weather_features, gee_features):
    """커스텀 화재 위험 피처들을 생성합니다."""
    custom_features = {}
    
    t2m_0h = weather_features.get('t2m_0h_past', 0)
    rh2m_0h = weather_features.get('rh2m_0h_past', 50)
    ws10m_0h = weather_features.get('ws10m_0h_past', 0)
    
    slope_mean = gee_features.get('slope_mean', 0)
    ndvi = gee_features.get('ndvi_before', 0)
    treecover = gee_features.get('treecover_pre_fire_5x5', 0)
    
    custom_features['dry_windy_combo'] = 1 if rh2m_0h < 30 and ws10m_0h > 10 else 0
    
    if ndvi < 0.3 and rh2m_0h < 40:
        fuel_score = 1
    elif treecover > 50 and rh2m_0h < 50:
        fuel_score = 0.7
    else:
        fuel_score = 0.3
    custom_features['fuel_combo'] = fuel_score
    
    wind_factor = min(ws10m_0h / 20.0, 1.0)
    dry_factor = max(0, (100 - rh2m_0h) / 100.0)
    temp_factor = max(0, (t2m_0h - 20) / 20.0) if t2m_0h > 20 else 0
    custom_features['potential_spread_index'] = (wind_factor + dry_factor + temp_factor) / 3.0
    
    custom_features['terrain_var_effect'] = min(slope_mean / 30.0, 1.0)
    
    wind_values = [weather_features[key] for hour in range(0, 25, 3) if (key := f"ws10m_{hour}h_past") in weather_features and weather_features[key] is not None]
    if len(wind_values) > 3:
        wind_std = (sum([(x - sum(wind_values)/len(wind_values))**2 for x in wind_values]) / len(wind_values))**0.5
        custom_features['wind_steady_flag'] = 1 if wind_std < 2.0 else 0
    else:
        custom_features['wind_steady_flag'] = 0
        
    total_precip_30d = weather_features.get('total_prec_30d_start_past', 0)
    dry_days_30d = weather_features.get('dry_days_30d_start_past', 30)
    
    custom_features['dry_to_rain_ratio_30d'] = dry_days_30d / max(total_precip_30d, 0.1) if total_precip_30d > 0 else 30.0
    
    if ndvi < 0.2:
        custom_features['ndvi_stress'] = 1.0
    elif ndvi < 0.4:
        custom_features['ndvi_stress'] = 0.6
    else:
        custom_features['ndvi_stress'] = 0.2
    
    return custom_features

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
    _initialized = True

def get_gee_features(lat, lon):
    point = ee.Geometry.Point([lon, lat])
    try:
        today = ee.Date(datetime.datetime.utcnow().isoformat(timespec='seconds') + 'Z')
        end = today.advance(-5, 'day')
        start = end.advance(-30, 'day')
        ndvi_img = ee.ImageCollection("MODIS/061/MOD13Q1").filterBounds(point).filterDate(start, end).sort('system:time_start', False).first()
        ndvi = ndvi_img.select('NDVI').reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=250, maxPixels=1e8).getInfo()
        treecover = ee.Image("UMD/hansen/global_forest_change_2022_v1_10").select("treecover2000").reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=30, maxPixels=1e8).getInfo()
        buffer = point.buffer(500)
        elev_img = ee.Image("USGS/SRTMGL1_003")
        slope_img = ee.Terrain.slope(elev_img)
        aspect_img = ee.Terrain.aspect(elev_img)
        reducer = ee.Reducer.mean().combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True).combine(reducer2=ee.Reducer.minMax(), sharedInputs=True)
        elev_stats = elev_img.reduceRegion(reducer=reducer, geometry=buffer, scale=90, maxPixels=1e8).getInfo()
        slope_stats = slope_img.reduceRegion(reducer=reducer, geometry=buffer, scale=90, maxPixels=1e8).getInfo()
        aspect_stats = aspect_img.reduceRegion(reducer=ee.Reducer.mode().combine(ee.Reducer.stdDev(), sharedInputs=True), geometry=buffer, scale=90, maxPixels=1e8).getInfo()
        ndvi_val = ndvi.get("NDVI", None)
        return {
            "ndvi_before": (float(ndvi_val) / 10000) if (ndvi_val is not None) else -999,
            "treecover_pre_fire_5x5": float(treecover.get("treecover2000", -999)),
            "elevation_mean": float(elev_stats.get("elevation_mean", -999)),
            "elevation_std": float(elev_stats.get("elevation_stddev", -999)),
            "elevation_min": float(elev_stats.get("elevation_min", -999)),
            "elevation_max": float(elev_stats.get("elevation_max", -999)),
            "slope_mean": float(slope_stats.get("slope_mean", -999)),
            "slope_std": float(slope_stats.get("slope_stddev", -999)),
            "slope_min": float(slope_stats.get("slope_min", -999)),
            "slope_max": float(slope_stats.get("slope_max", -999)),
            "aspect_mode": float(aspect_stats.get("aspect_mode", -999)),
            "aspect_std": float(aspect_stats.get("aspect_stddev", -999)),
        }
    except Exception:
        traceback.print_exc()
        return {k: -999 for k in [
            "ndvi_before", "treecover_pre_fire_5x5", "elevation_mean", "elevation_min", "elevation_max",
            "elevation_std", "slope_mean", "slope_min", "slope_max", "slope_std", "aspect_mode", "aspect_std"
        ]}

def move_coordinate(lat, lon, distance_m, bearing_deg):
    R = 6378137
    d = distance_m
    brng = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(math.sin(lat1) * math.cos(d / R) + math.cos(lat1) * math.sin(d / R) * math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(d / R) * math.cos(lat1), math.cos(d / R) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)

def align_features_to_model(features_df, expected_columns):
    missing_cols = set(expected_columns) - set(features_df.columns)
    for c in missing_cols:
        features_df[c] = 0
    return features_df[expected_columns]

def predict_single_timestep(lat, lon, timestamp, gee_features, simulation_hours_total):
    _initialize_prediction_environment()
    
    # 1. 4일 전 기준으로 날씨 피처 수집
    weather_features = fetch_all_weather_features(lat, lon, timestamp, offset_days=4)
    if not weather_features or not weather_features.get("success"):
        print(f"❌ Weather feature fetching failed for {lat},{lon} at {timestamp}")
        return { "error": "Weather feature fetching failed" }

    # 2. 통계 및 커스텀 피처 생성
    stats_features = generate_statistical_features(weather_features)
    custom_features = generate_custom_features(weather_features, gee_features)

    # 3. 모든 피처 통합
    combined_features = {**weather_features, **gee_features, **stats_features, **custom_features}
    features_df = pd.DataFrame([combined_features])

    # --- Area Model Prediction ---
    area_input_df = align_features_to_model(features_df.copy(), MODELS["area_cols"])
    area_scaled = MODELS["area_scaler"].transform(area_input_df)
    area_log = MODELS["area_model"].predict(area_scaled)[0]
    base_area = float(np.expm1(area_log)) if np.isfinite(area_log) else 0.0
    time_factor = (simulation_hours_total / 6.0) ** 1.5
    area = base_area * time_factor

    # --- Speed and Direction Model Prediction ---
    speed_input_df = align_features_to_model(features_df.copy(), MODELS["speed_cols"])
    speed_scaled = MODELS["speed_scaler"].transform(speed_input_df)
    speed_cat = int(MODELS["speed_model"].predict(speed_scaled)[0])
    
    direction_input_df = align_features_to_model(features_df.copy(), MODELS["direction_cols"])
    direction_scaled = MODELS["direction_scaler"].transform(direction_input_df)
    direction_cat = int(MODELS["direction_model"].predict(direction_scaled)[0])

    wind_dir = float(weather_features.get("wd10m_0h_past", -999) or -999)

    return {
        "hourly_damage_area": area,
        "spread_speed_category": speed_cat,
        "spread_direction_category": direction_cat,
        "predicted_distance_m": float(np.sqrt(max(0.0, area) * 10000 / np.pi)),
        "wind_direction_deg": wind_dir
    }

def predict_simulation(input_json):
    _initialize_prediction_environment()
    current_lat = input_json["latitude"]
    current_lon = input_json["longitude"]
    start_timestamp = datetime.datetime.fromisoformat(input_json["timestamp"])
    simulation_hours = input_json.get("durationHours", 1)
    path_trace = []

    print(f"🌍 Starting simulation for ({current_lat:.4f}, {current_lon:.4f}) over {simulation_hours} hours")
    
    initial_gee_features = get_gee_features(current_lat, current_lon)
    print("Initial GEE features fetched.")

    cumulative_area = 0.0
    total_distance_traveled = 0.0

    for hour in range(simulation_hours):
        print(f"⏰ Simulating hour {hour + 1}/{simulation_hours}...")
        current_timestamp = start_timestamp + datetime.timedelta(hours=hour)
        
        current_gee_features = get_gee_features(current_lat, current_lon)

        timestep_result = predict_single_timestep(current_lat, current_lon, current_timestamp, current_gee_features, hour + 1)
        
        if "error" in timestep_result:
            print(f"Error in timestep {hour+1}: {timestep_result['error']}")
            break

        timestep_result.update({
            "simulation_hour": hour + 1,
            "current_lat": current_lat,
            "current_lon": current_lon,
            "timestamp": current_timestamp.isoformat()
        })
        path_trace.append(timestep_result)
        
        hourly_area = timestep_result.get("hourly_damage_area", 0)
        cumulative_area = max(cumulative_area, hourly_area)
        
        predicted_direction_cat = timestep_result.get('spread_direction_category', -1)
        direction_map_deg = {0: 0, 1: 45, 2: 90, 3: 135, 4: 180, 5: 225, 6: 270, 7: 315}
        direction_deg = direction_map_deg.get(predicted_direction_cat, timestep_result["wind_direction_deg"])
        
        current_radius = np.sqrt(cumulative_area * 10000 / np.pi)
        move_distance = current_radius - total_distance_traveled

        if move_distance > 0:
            current_lat, current_lon = move_coordinate(current_lat, current_lon, move_distance, direction_deg)
            total_distance_traveled += move_distance

        dir_cat = timestep_result.get('spread_direction_category', 0)
        print(f"   📍 Position: ({current_lat:.4f}, {current_lon:.4f})")
        print(f"   🔥 Area: {cumulative_area:.2f} ha, Distance: {total_distance_traveled:.1f}m")
        print(f"   🧭 Direction: {DIRECTION_MAP.get(dir_cat, 'N/A')} ({direction_deg:.1f}°), Speed: {SPEED_CATEGORY_MAP.get(timestep_result.get('spread_speed_category', 0), 0):.1f} m/h")

    print(f"\n📈 Simulation Complete:")
    print(f"   Total Area: {cumulative_area:.2f} ha")
    print(f"   Total Distance: {total_distance_traveled:.1f} m")
    print(f"   Final Position: ({current_lat:.4f}, {current_lon:.4f})")

    final_step = path_trace[-1] if path_trace else {}
    final_damage_area = final_step.get("hourly_damage_area", 0)
    speed_cat = final_step.get("spread_speed_category", 0)
    direction_cat = final_step.get("spread_direction_category", 0)

    return {
        "simulation_hours": simulation_hours,
        "final_damage_area": cumulative_area,
        "final_lat": current_lat,
        "final_lon": current_lon,
        "path_trace": path_trace,
        "predicted_spread_direction": DIRECTION_MAP.get(direction_cat, "N/A"),
        "total_spread_distance": total_distance_traveled,
        "predicted_spread_speed": SPEED_CATEGORY_MAP.get(speed_cat, 0.0)
    }

if __name__ == "__main__":
    _initialize_prediction_environment()
    try:
        # Example for direct execution
        # input_data = {
        #     "latitude": 37.75,
        #     "longitude": 128.86,
        #     "timestamp": "2024-01-01T12:00:00",
        #     "durationHours": 6
        # }
        input_data = json.loads(sys.stdin.read())
        result = predict_simulation(input_data)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e), "traceback": traceback.format_exc()}))
