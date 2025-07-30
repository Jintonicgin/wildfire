import datetime
import sys
import json
import numpy as np
import traceback
from fwi_calc import fwi_calc
import requests

# --- API 호출 함수 (수정됨) ---
def _fetch_nasa_raw_hourly_data_for_day(lat, lng, yyyymmdd, max_retry=3):
    url = (
        f"https://power.larc.nasa.gov/api/temporal/hourly/point?"
        "parameters=T2M,RH2M,WS2M,WD2M,PRECTOTCORR,PS,ALLSKY_SFC_SW_DWN,WS10M,WD10M"
        f"&community=RE&longitude={lng}&latitude={lat}&start={yyyymmdd}&end={yyyymmdd}&format=JSON"
    )

    for attempt in range(max_retry):
        try:
            res = requests.get(url, timeout=30)
            res.raise_for_status()
            raw_data = res.json()
            # 이 함수는 하루 전체의 시간별 데이터를 포함하는 'parameter' 딕셔너리를 반환
            return raw_data.get("properties", {}).get("parameter", {})
        except Exception as e:
            print(f"❌ NASA Raw Hourly Data API 호출 실패: {e}", file=sys.stderr) # 오류 로깅 추가
            if attempt == max_retry - 1:
                # 마지막 시도에서도 실패하면 None 반환
                return None
            # 다음 시도를 위해 계속

def fetch_nasa_daily_precip(lat, lng, start_date, end_date, max_retry=3):
    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters=PRECTOTCORR&community=RE&longitude={lng}&latitude={lat}"
        f"&start={start_date}&end={end_date}&format=JSON"
    )
    for attempt in range(max_retry):
        try:
            res = requests.get(url, timeout=30)
            res.raise_for_status()
            data = res.json().get("properties", {}).get("parameter", {})
            return data
        except Exception as e:
            print(f"❌ NASA Daily Precip API 호출 실패: {e}", file=sys.stderr) # 오류 로깅 추가
            if attempt == max_retry - 1:
                return None

def make_precip_features(lat, lng, end_dt, periods=[7, 14, 30, 60, 90]):
    res = {}
    
    # 가장 긴 기간의 데이터를 한 번에 가져옴
    max_ndays = max(periods)
    start_date_for_api = (end_dt - datetime.timedelta(days=max_ndays - 1)).strftime("%Y%m%d")
    end_date_for_api = end_dt.strftime("%Y%m%d")
    
    # API 호출은 한 번만 수행
    full_precip_dict = fetch_nasa_daily_precip(lat, lng, start_date_for_api, end_date_for_api)
    if full_precip_dict is None: # API 호출 실패 시 빈 딕셔너리로 초기화
        full_precip_dict = {}

    for ndays in periods:
        vals = []
        for i in range(ndays): # 과거 ndays 일 동안의 데이터
            current_date_str = (end_dt - datetime.timedelta(days=i)).strftime("%Y%m%d")
            vals.append(float(full_precip_dict.get("PRECTOTCORR", {}).get(current_date_str, np.nan))) # PRECTOTCORR 키 추가
        
        arr = np.array(vals[::-1], dtype=float) # 최신 데이터가 뒤로 오도록 역순 정렬
        res[f"total_precip_{ndays}d_start"] = float(np.nansum(arr)) if not np.all(np.isnan(arr)) else 0.0
        res[f"dry_days_{ndays}d_start"] = int(np.nansum(arr < 1)) if not np.all(np.isnan(arr)) else ndays
    
    cons = 0
    # arr이 비어있거나 모두 NaN인 경우를 처리
    if arr.size > 0 and not np.all(np.isnan(arr)):
        for v in arr[::-1]: # 최신 데이터부터 역순으로 확인
            if np.isnan(v) or v < 1:
                cons += 1
            else:
                break
    res[f"consecutive_dry_days_start"] = int(cons)
    return res

def make_time_points(start_dt, end_dt, interval_hours=3):
    points = [start_dt]
    curr_dt = start_dt
    while True:
        curr_dt += datetime.timedelta(hours=interval_hours)
        if curr_dt >= end_dt:
            break
        points.append(curr_dt)
    if points[-1] != end_dt:
        points.append(end_dt)
    return points

def fetch_all_weather_features(lat, lon, timestamp, offset_days=4):
    # timestamp에서 offset_days만큼 과거로 이동
    adjusted_timestamp = timestamp - datetime.timedelta(days=offset_days)

    end_dt = adjusted_timestamp - datetime.timedelta(days=3)
    start_dt = end_dt - datetime.timedelta(hours=6)

    precip_feats = make_precip_features(lat, lon, end_dt)
    weather_list = []
    time_points = make_time_points(start_dt, end_dt, interval_hours=3)
    
    DEFAULTS = {
        "T2M": 15.0, "RH2M": 50.0, "WS2M": 1.5, "WD2M": 180.0,
        "WS10M": 2.0, "WD10M": 180.0, "PRECTOTCORR": 0.0, "PS": 101.3,
        "ALLSKY_SFC_SW_DWN": 0.0
    }

    # 각 날짜별로 시간별 데이터를 한 번만 가져오도록 최적화
    hourly_data_cache = {}
    for dt in time_points:
        yyyymmdd = dt.strftime("%Y%m%d")
        if yyyymmdd not in hourly_data_cache:
            # 하루 전체의 시간별 데이터를 가져와 캐시
            raw_hourly_data = _fetch_nasa_raw_hourly_data_for_day(lat, lon, yyyymmdd)
            if raw_hourly_data is None: # API 호출 실패 시 빈 딕셔너리로 초기화
                raw_hourly_data = {}
            hourly_data_cache[yyyymmdd] = raw_hourly_data
        
        # 캐시된 하루치 데이터에서 해당 시간의 데이터 추출
        hour_key = f"{yyyymmdd}{dt.strftime('%H').zfill(2)}" # 수정된 부분
        w = {}
        for key in DEFAULTS.keys():
            # hourly_data_cache[yyyymmdd]는 이제 {param: {YYYYMMDDHH: value, ...}} 형태
            w[key] = hourly_data_cache[yyyymmdd].get(key, {}).get(hour_key, np.nan)

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
        
        for key in DEFAULTS.keys():
            if isinstance(w.get(key), np.number):
                w[key] = float(w[key])

        w["dt"] = dt.strftime("%Y-%m-%d %H:%M")
        weather_list.append(w)

    target_weather = next((w for w in weather_list if "00:00" in w["dt"]), weather_list[0])

    fwi = fwi_calc(
        T=target_weather.get("T2M", 20),
        RH=target_weather.get("RH2M", 40),
        W=target_weather.get("WS10M", 3),
        P=target_weather.get("PRECTOTCORR", 0),
        month=end_dt.month
    )

    # FWI 피처 (접미사 없음)
    fwi_no_suffix = {k: round(float(v), 2) for k, v in fwi.items()}

    # FWI 피처 (_0h 접미사 있음)
    fwi_with_suffix = {f"{k}_0h": round(float(v), 2) for k, v in fwi.items()}

    derived = {
        "dry_windy_combo": int(target_weather.get("RH2M", 0) < 35 and target_weather.get("WS10M", 0) > 5),
        "fuel_combo": float(fwi_no_suffix.get("FFMC", 0) * fwi_no_suffix.get("ISI", 0)),
        "potential_spread_index": float(fwi_no_suffix.get("ISI", 0) * fwi_no_suffix.get("FWI", 0)),
        "terrain_var_effect": 0.0,
        "wind_steady_flag": int(np.nanmax([w.get("WS10M", 0) for w in weather_list]) - np.nanmin([w.get("WS10M", 0) for w in weather_list]) < 1.5),
        "dry_to_rain_ratio_30d": (
            precip_feats.get("dry_days_30d_start", 0) / 
            max(precip_feats.get("total_precip_30d_start", 0), 1e-2)
        ),
        "ndvi_stress": 0.6
    }

    month = end_dt.month
    season_flags = {
        "is_spring": int(month in [3, 4, 5]),
        "is_summer": int(month in [6, 7, 8]),
        "is_autumn": int(month in [9, 10, 11]),
        "is_winter": int(month in [12, 1, 2])
    }

    result = {
        "lat": lat,
        "lng": lon,
        "start_dt": start_dt.strftime("%Y-%m-%d %H:%M"),
        "end_dt": end_dt.strftime("%Y-%m-%d %H:%M"),
        "duration_hours": round((end_dt - start_dt).total_seconds() / 3600, 2),
        "weather_timeseries": weather_list,
        
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

if __name__ == "__main__":
    try:
        lat = float(sys.argv[1])
        lng = float(sys.argv[2])
        start_yyyymmdd = sys.argv[3]
        start_hhmm = sys.argv[4]
        end_yyyymmdd = sys.argv[5]
        end_hhmm = sys.argv[6]
        start_dt = datetime.datetime.strptime(start_yyyymmdd + start_hhmm, "%Y%m%d%H%M")
        end_dt = datetime.datetime.strptime(end_yyyymmdd + end_hhmm, "%Y%m%d%H%M")
    except Exception:
        print(json.dumps({
            "success": False,
            "error": "인자 부족! 사용법: python fetch_weather_data.py lat lng start_yyyymmdd start_hhmm end_yyyymmdd end_hhmm"
        }, ensure_ascii=False))
        sys.exit(2)

    try:
        result = fetch_all_weather_features(lat, lng, end_dt)
        print(json.dumps(result, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }, ensure_ascii=False))
        sys.exit(2)
