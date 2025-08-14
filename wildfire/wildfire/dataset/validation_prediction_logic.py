import sys
import json
import joblib
import datetime
import numpy as np
import pandas as pd
import ee # GEE API는 사용하지 않지만, 기존 코드 구조 유지를 위해 임포트
import warnings
import math
import os # os 모듈 임포트

# fwi_calc.py 모듈 임포트
sys.path.append("/Users/mmymacymac/Developer/Projects/eclipse/WildFire/dataset/")
from fwi_calc import fwi_calc

warnings.filterwarnings("ignore")

# MODEL_PATH를 절대 경로로 지정
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + os.sep + "dataset" + os.sep

# 전역 변수 및 초기화 플래그
MODELS = None
_initialized = False

# --- 초기화 함수 (predict.py와 동일) ---
def _initialize_prediction_environment():
    global MODELS, _initialized
    if _initialized: # 이미 초기화되었다면 건너뛰기
        return

    # GEE 초기화 (검증 시 API 호출은 없지만, 모델 로드 시 필요할 수 있으므로 유지)
    try:
        ee.Initialize(project='wildfire-464907')
    except Exception:
        ee.Authenticate()
        ee.Initialize(project='wildfire-464907')

    # 모델 및 스케일러 로드 (predict.py와 동일)
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

# --- 지리 정보 피처 (CSV에서 직접 가져옴) ---
def get_gee_features_from_csv(row):
    return {
        "ndvi_before": float(row.get("ndvi_pre_fire_latest")) if pd.notna(row.get("ndvi_pre_fire_latest")) else -999,
        "treecover_pre_fire_5x5": float(row.get("forest_cover_5km_percent")) if pd.notna(row.get("forest_cover_5km_percent")) else -999, # CSV 컬럼명에 맞춤
        "elevation_mean": float(row.get("elevation")) if pd.notna(row.get("elevation")) else -999,
        "elevation_min": float(row.get("elevation")) if pd.notna(row.get("elevation")) else -999, # CSV에 min/max/std 없음, mean으로 대체
        "elevation_max": float(row.get("elevation")) if pd.notna(row.get("elevation")) else -999,
        "elevation_std": 0.0, # CSV에 없음, 0으로 대체
        "slope_mean": float(row.get("slope")) if pd.notna(row.get("slope")) else -999,
        "slope_min": float(row.get("slope")) if pd.notna(row.get("slope")) else -999, # CSV에 min/max/std 없음, mean으로 대체
        "slope_max": float(row.get("slope")) if pd.notna(row.get("slope")) else -999,
        "slope_std": 0.0, # CSV에 없음, 0으로 대체
        "aspect_mode": float(row.get("aspect")) if pd.notna(row.get("aspect")) else -999,
        "aspect_std": 0.0, # CSV에 없음, 0으로 대체
    }

# --- 기상 피처 (CSV에서 직접 가져와서 파생 피처 계산) ---
def fetch_all_weather_features_from_csv(row, timestamp):
    # CSV에서 직접 원본 기상 데이터 추출
    # CSV 컬럼명: t2m_1, rh2m_1, ws2m_1, wd2m_1, prectotcorr_1, ps_1, allsky_sfc_sw_dwn_1, ws10m_1, wd10m_1
    # 그리고 t2m_2, ..., t2m_59 등
    
    # 현재 시간 (timestamp)을 기준으로 과거 59시간까지의 데이터 추출
    weather_list = []
    
    # CSV의 start_datetime을 기준으로 각 시간별 데이터의 실제 시간 계산
    # CSV의 start_datetime은 fire_start_date + start_time
    # CSV의 t2m_1은 fire_start_date + start_time - 58시간
    # CSV의 t2m_59는 fire_start_date + start_time
    
    # CSV의 start_datetime을 datetime 객체로 변환
    csv_start_dt = datetime.datetime.strptime(row['start_datetime'], '%Y-%m-%d %H:%M:%S')

    DEFAULTS = {
        "T2M": 15.0, "RH2M": 50.0, "WS2M": 1.5, "WD2M": 180.0,
        "WS10M": 2.0, "WD10M": 180.0, "PRECTOTCORR": 0.0, "PS": 101.3,
        "ALLSKY_SFC_SW_DWN": 0.0
    }

    # CSV에서 59시간치 데이터 추출
    for i in range(1, 60):
        current_dt_for_csv = csv_start_dt - datetime.timedelta(hours=(59 - i))
        
        w = {}
        for key_base in ["t2m", "rh2m", "ws2m", "wd2m", "prectotcorr", "ps", "allsky_sfc_sw_dwn", "ws10m", "wd10m"]:
            col_name = f"{key_base}_{i}"
            val = row.get(col_name, np.nan)
            w[key_base.upper()] = float(val) if pd.notna(val) else np.nan
        
        # 누락된 값 처리 (fetch_all_weather.py의 로직과 유사하게)
        last_w = weather_list[-1] if weather_list else {}
        for key in DEFAULTS.keys():
            if np.isnan(w.get(key, np.nan)):
                fallback_key = None
                if key == 'WD10M': fallback_key = 'WD2M'
                elif key == 'WD2M': fallback_key = 'WD10M'
                elif key == 'WS10M': fallback_key = 'WS2M'
                elif key == 'WS2M': fallback_key = 'WS10M'

                if fallback_key and not np.isnan(w.get(fallback_key, np.nan)):
                    w[key] = w[fallback_key]
                elif not np.isnan(last_w.get(key, np.nan)):
                    w[key] = last_w[key]
                else:
                    w[key] = DEFAULTS[key]
        
        w["dt"] = current_dt_for_csv.strftime("%Y-%m-%d %H:%M")
        weather_list.append(w)

    # make_precip_features 로직 (CSV 데이터 기반)
    precip_feats = {}
    periods=[7, 14, 30, 60, 90]
    
    # CSV에서 일별 강수량 데이터 추출
    daily_precip_data = {}
    for i in range(1, 60): # 59시간치 데이터에서 일별 강수량 합산
        current_dt_for_csv = csv_start_dt - datetime.timedelta(hours=(59 - i))
        current_date_str = current_dt_for_csv.strftime("%Y%m%d")
        precip_val = row.get(f"prectotcorr_{i}", np.nan)
        if pd.notna(precip_val):
            daily_precip_data[current_date_str] = daily_precip_data.get(current_date_str, 0.0) + float(precip_val)

    for ndays in periods:
        vals = []
        for i in range(ndays):
            current_date_str = (timestamp - datetime.timedelta(days=i)).strftime("%Y%m%d") # 예측 기준 시간의 과거 날짜
            vals.append(float(daily_precip_data.get(current_date_str, np.nan)))
        
        arr = np.array(vals[::-1], dtype=float)
        precip_feats[f"total_precip_{ndays}d_start"] = float(np.nansum(arr)) if not np.all(np.isnan(arr)) else 0.0
        precip_feats[f"dry_days_{ndays}d_start"] = int(np.nansum(arr < 1)) if not np.all(np.isnan(arr)) else ndays
    
    cons = 0
    if arr.size > 0 and not np.all(np.isnan(arr)):
        for v in arr[::-1]:
            if np.isnan(v) or v < 1:
                cons += 1
            else:
                break
    precip_feats[f"consecutive_dry_days_start"] = int(cons)

    # FWI 계산을 위한 target_weather (timestamp 기준)
    # weather_list에서 timestamp에 가장 가까운 00시 데이터를 찾거나, 첫 번째 데이터 사용
    target_weather = next((w for w in weather_list if "00:00" in w["dt"]), weather_list[0])

    fwi = fwi_calc(
        T=target_weather.get("T2M", 20),
        RH=target_weather.get("RH2M", 40),
        W=target_weather.get("WS10M", 3),
        P=target_weather.get("PRECTOTCORR", 0),
        month=timestamp.month # 예측 기준 시간의 월 사용
    )

    fwi_no_suffix = {k: round(float(v), 2) for k, v in fwi.items()}
    fwi_with_suffix = {f"{k}_0h": round(float(v), 2) for k, v in fwi.items()}

    derived = {
        "dry_windy_combo": int(target_weather.get("RH2M", 0) < 35 and target_weather.get("WS10M", 0) > 5),
        "fuel_combo": float(fwi_no_suffix.get("FFMC", 0) * fwi_no_suffix.get("ISI", 0)),
        "potential_spread_index": float(fwi_no_suffix.get("ISI", 0) * fwi_no_suffix.get("FWI", 0)),
        "terrain_var_effect": 0.0,
        "wind_steady_flag": int(np.nanmax([w.get("WS10M", 0) for w in weather_list]) - np.nanmin([w.get("WS10M", 0) for w in weather_list]) < 1.5) if weather_list else 0,
        "dry_to_rain_ratio_30d": (
            precip_feats.get("dry_days_30d_start", 0) / 
            max(precip_feats.get("total_precip_30d_start", 0), 1e-2)
        ),
        "ndvi_stress": 0.6 # CSV에 ndvi_stress 컬럼이 없으므로 고정값 사용
    }

    month = timestamp.month
    season_flags = {
        "is_spring": int(month in [3, 4, 5]),
        "is_summer": int(month in [6, 7, 8]),
        "is_autumn": int(month in [9, 10, 11]),
        "is_winter": int(month in [12, 1, 2])
    }

    result = {
        "lat": row['latitude'], # CSV row에서 직접 lat/lng 가져옴
        "lng": row['longitude'],
        "start_dt": weather_list[0]["dt"], # weather_list의 첫 시간
        "end_dt": weather_list[-1]["dt"], # weather_list의 마지막 시간
        "duration_hours": round((datetime.datetime.strptime(weather_list[-1]["dt"], "%Y-%m-%d %H:%M") - datetime.datetime.strptime(weather_list[0]["dt"], "%Y-%m-%d %H:%M")).total_seconds() / 3600, 2), # weather_list 기반으로 계산
        "weather_timeseries": weather_list, # 이 부분은 모델 입력에 직접 사용되지 않음
        
        # 현재 시점의 날씨 피처 (접미사 없음)
        "T2M": target_weather.get("T2M", -999),
        "RH2M": target_weather.get("RH2M", -999),
        "WS10M": target_weather.get("WS10M", -999),
        "WD10M": target_weather.get("WD10M", -999),
        "PRECTOTCORR": target_weather.get("PRECTOTCORR", -999),
        "PS": target_weather.get("PS", -999),
        "ALLSKY_SFC_SW_DWN": target_weather.get("ALLSKY_SFC_SW_DWN", -999),

        # 현재 시점의 날씨 피처 (_0h 접미사 있음)
        "T2M_0h": target_weather.get("T2M", -999),
        "RH2M_0h": target_weather.get("RH2M", -999),
        "WS2M_0h": target_weather.get("WS2M", -999),
        "WD2M_0h": target_weather.get("WD2M", -999),
        "WS10M_0h": target_weather.get("WS10M", -999),
        "WD10M_0h": target_weather.get("WD10M", -999),
        "PRECTOTCORR_0h": target_weather.get("PRECTOTCORR", -999),
        "PS_0h": target_weather.get("PS", -999),
        "ALLSKY_SFC_SW_DWN_0h": target_weather.get("ALLSKY_SFC_SW_DWN", -999),

        **precip_feats,
        **fwi_no_suffix,
        **fwi_with_suffix,
        **derived,
        **season_flags,
        "success": True
    }
    return result

# --- 단일 시간 단계 예측 함수 (predict.py와 동일하지만, 데이터 소스 변경) --- 
def predict_single_timestep_for_validation(lat, lon, timestamp, raw_data_row, simulation_hours_total):
    _initialize_prediction_environment()

    base = {"startyear": timestamp.year, "startmonth": timestamp.month, "startday": timestamp.day}
    
    # CSV에서 가져온 raw_data_row를 사용하여 GEE 피처 생성
    gee_features = get_gee_features_from_csv(raw_data_row)
    
    # CSV에서 가져온 raw_data_row를 사용하여 기상 피처 생성
    weather_features = fetch_all_weather_features_from_csv(raw_data_row, timestamp)
    
    season = add_season_flags(timestamp)

    all_features = {
        **weather_features, # weather_features의 모든 키-값을 언팩
        **base,
        **gee_features,
        **season,
        "lat": lat,
        "lng": lon,
        "duration_hours": float(simulation_hours_total), # 사용자가 요청한 총 시뮬레이션 시간 사용
        # duration_x_... 피처 계산 및 추가
        "duration_x_ws10m": float(simulation_hours_total) * weather_features.get("WS10M", 0),
        "duration_x_t2m": float(simulation_hours_total) * weather_features.get("T2M", 0),
        "duration_x_rh2m": float(simulation_hours_total) * weather_features.get("RH2M", 0),
        "duration_x_fwi": float(simulation_hours_total) * weather_features.get("FWI", 0),
        "duration_x_isi": float(simulation_hours_total) * weather_features.get("ISI", 0),
    }

    df = pd.DataFrame([all_features])

    # 명시적으로 정수형으로 변환해야 하는 컬럼들
    integer_cols = ["startyear", "startmonth", "startday", "is_spring", "is_summer", "is_autumn", "is_winter"]
    for col in integer_cols:
        if col in df.columns:
            # NaN 값은 0으로 채운 후 정수형으로 변환
            df[col] = df[col].fillna(0).apply(lambda x: int(x) if pd.notna(x) else 0)

    # 피처 준비
    area_input = df[MODELS["area_cols"]].copy().fillna(0)
    speed_input = df[MODELS["speed_cols"]].copy().fillna(0)
    direction_input = df[MODELS["direction_cols"]].copy().fillna(0)

    area_scaled = MODELS["area_scaler"].transform(area_input)
    speed_scaled = MODELS["speed_scaler"].transform(speed_input)

    area_log = MODELS["area_model"].predict(area_scaled)[0]
    area = float(np.expm1(area_log))
    if not np.isfinite(area) or area < 0: area = 0.0

    spread_dist_m = float(np.sqrt(area * 10000 / np.pi)) # 1시간 동안의 확산 거리
    
    # 예측 결과가 np.nan일 경우를 대비하여 기본값 설정
    speed_cat_raw = MODELS["speed_model"].predict(speed_scaled)[0]
    speed_cat = int(speed_cat_raw) if pd.notna(speed_cat_raw) else 0 # 기본값 0 (느림)

    direction_class_raw = MODELS["direction_model"].predict(direction_input)[0]
    direction_class = str(direction_class_raw) if pd.notna(direction_class_raw) else "N" # 기본값 "N"

    return {
        "hourly_damage_area": area,
        "spread_speed_category": speed_cat,
        "spread_direction": direction_class,
        "predicted_distance_m": spread_dist_m,
        "wind_direction_deg": float(weather_features.get("WD10M_0h", -999))
    }

# --- 시뮬레이션 예측 함수 (predict.py와 동일하지만, 데이터 소스 변경) --- 
def predict_simulation_for_validation(input_json, raw_data_row):
    _initialize_prediction_environment()

    current_lat = input_json["latitude"]
    current_lon = input_json["longitude"]
    start_timestamp = datetime.datetime.fromisoformat(input_json["timestamp"])
    simulation_hours = input_json.get("durationHours", 1)

    total_damage_area = 0
    path_trace = []

    # GEE 피처는 시뮬레이션 시작 시 한 번만 가져옴 (CSV에서)
    # raw_data_row를 get_gee_features_from_csv에 전달
    # initial_gee_features = get_gee_features_from_csv(raw_data_row) # 이 줄은 필요 없음

    for hour in range(simulation_hours):
        current_timestamp = start_timestamp + datetime.timedelta(hours=hour)
        
        # 현재 시간 단계 예측 (GEE 피처 재사용, 기상 데이터는 CSV에서)
        # raw_data_row를 fetch_all_weather_features_from_csv에 전달
        timestep_result = predict_single_timestep_for_validation(
            current_lat, current_lon, current_timestamp, raw_data_row, simulation_hours
        )
        
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

# predict.py의 add_season_flags, move_coordinate 함수도 여기에 포함
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

# 이 파일은 직접 실행되지 않고 다른 모듈에서 임포트하여 사용
if __name__ == "__main__":
    print("이 모듈은 직접 실행되지 않습니다. validate_model.py에서 임포트하여 사용하세요.")
