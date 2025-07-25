import requests
import datetime
import sys
import json
import numpy as np
import traceback
from fwi_calc import fwi_calc

def fetch_nasa_hourly_weather(lat, lng, yyyymmdd, hour_str, max_retry=3):
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
            data = raw_data.get("properties", {}).get("parameter", {})

            hour_key = f"{yyyymmdd}{hour_str.zfill(2)}"

            result = {
                "T2M": data.get("T2M", {}).get(hour_key, np.nan),
                "RH2M": data.get("RH2M", {}).get(hour_key, np.nan),
                "WS2M": data.get("WS2M", {}).get(hour_key, np.nan),
                "WD2M": data.get("WD2M", {}).get(hour_key, np.nan),
                "WS10M": data.get("WS10M", {}).get(hour_key, np.nan),
                "WD10M": data.get("WD10M", {}).get(hour_key, np.nan),
                "PRECTOTCORR": data.get("PRECTOTCORR", {}).get(hour_key, np.nan),
                "PS": data.get("PS", {}).get(hour_key, np.nan),
                "ALLSKY_SFC_SW_DWN": data.get("ALLSKY_SFC_SW_DWN", {}).get(hour_key, np.nan)
            }
            return result
        except Exception as e:
            if attempt == max_retry - 1:
                raise e

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
            data = res.json().get("properties", {}).get("parameter", {}).get("PRECTOTCORR", {})
            return data
        except Exception as e:
            if attempt == max_retry - 1:
                raise e

def make_precip_features(lat, lng, end_dt, periods=[7, 14, 30, 60, 90]):
    res = {}
    dt_str = end_dt.strftime("%Y%m%d")
    for ndays in periods:
        sdate = (end_dt - datetime.timedelta(days=ndays - 1)).strftime("%Y%m%d")
        precip_dict = fetch_nasa_daily_precip(lat, lng, sdate, dt_str)
        vals = [float(precip_dict.get((end_dt - datetime.timedelta(days=i)).strftime("%Y%m%d"), np.nan)) for i in range(ndays)][::-1]
        arr = np.array(vals, dtype=float)
        res[f"total_precip_{ndays}d_start"] = float(np.nansum(arr))
        res[f"dry_days_{ndays}d_start"] = int(np.sum(arr < 1))
    cons = 0
    for v in arr[::-1]:
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

    for dt in time_points:
        yyyymmdd = dt.strftime("%Y%m%d")
        hour_str = dt.strftime("%H")
        w = fetch_nasa_hourly_weather(lat, lon, yyyymmdd, hour_str)
        
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
