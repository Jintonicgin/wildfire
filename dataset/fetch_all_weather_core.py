import datetime
import sys
import json
import numpy as np
import traceback
from fwi_calc import fwi_calc
import requests

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
            return raw_data.get("properties", {}).get("parameter", {})
        except Exception as e:
            print(f"❌ NASA Raw Hourly Data API 호출 실패: {e}")
            if attempt == max_retry - 1:
                return None

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
            if attempt == max_retry - 1:
                return None

def make_precip_features(lat, lng, end_dt, periods=[7, 14, 30, 60, 90]):
    res = {}
    max_ndays = max(periods)
    start_date = (end_dt - datetime.timedelta(days=max_ndays - 1)).strftime("%Y%m%d")
    end_date = end_dt.strftime("%Y%m%d")
    full_precip = fetch_nasa_daily_precip(lat, lng, start_date, end_date)
    if full_precip is None:
        full_precip = {}

    for ndays in periods:
        vals = []
        for i in range(ndays):
            d = (end_dt - datetime.timedelta(days=i)).strftime("%Y%m%d")
            vals.append(float(full_precip.get("PRECTOTCORR", {}).get(d, np.nan)))
        arr = np.array(vals[::-1], dtype=float)
        res[f"total_precip_{ndays}d_start"] = float(np.nansum(arr)) if not np.all(np.isnan(arr)) else 0.0
        res[f"dry_days_{ndays}d_start"] = int(np.nansum(arr < 1)) if not np.all(np.isnan(arr)) else ndays

    cons = 0
    if arr.size > 0 and not np.all(np.isnan(arr)):
        for v in arr[::-1]:
            if np.isnan(v) or v < 1:
                cons += 1
            else:
                break
    res["consecutive_dry_days_start"] = int(cons)
    return res

def make_time_points(start_dt, end_dt, interval_hours=3):
    points = [start_dt]
    curr = start_dt
    while True:
        curr += datetime.timedelta(hours=interval_hours)
        if curr >= end_dt:
            break
        points.append(curr)
    if points[-1] != end_dt:
        points.append(end_dt)
    return points

def fetch_all_weather_features(lat, lon, timestamp, offset_days=7):
    target_dt = timestamp - datetime.timedelta(days=offset_days)
    start_dt = target_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    end_dt = start_dt + datetime.timedelta(hours=23)

    precip_feats = make_precip_features(lat, lon, end_dt)
    time_points = make_time_points(start_dt, end_dt)
    weather_list = []

    DEFAULTS = {
        "T2M": 15.0, "RH2M": 50.0, "WS2M": 1.5, "WD2M": 180.0,
        "WS10M": 2.0, "WD10M": 180.0, "PRECTOTCORR": 0.0, "PS": 1013.0,
        "ALLSKY_SFC_SW_DWN": 0.0
    }

    hourly_cache = {}
    for dt in time_points:
        ymd = dt.strftime("%Y%m%d")
        if ymd not in hourly_cache:
            raw = _fetch_nasa_raw_hourly_data_for_day(lat, lon, ymd)
            hourly_cache[ymd] = raw or {}

        hour_key = f"{ymd}{dt.strftime('%H').zfill(2)}"
        w = {}
        for key in DEFAULTS:
            w[key] = hourly_cache[ymd].get(key, {}).get(hour_key, np.nan)

        last_w = weather_list[-1] if weather_list else {}
        for key in DEFAULTS:
            if np.isnan(w.get(key, np.nan)):
                fb = None
                if key == 'WD10M': fb = 'WD2M'
                elif key == 'WD2M': fb = 'WD10M'
                elif key == 'WS10M': fb = 'WS2M'
                elif key == 'WS2M': fb = 'WS10M'
                if fb and not np.isnan(w.get(fb, np.nan)):
                    w[key] = w[fb]
                elif not np.isnan(last_w.get(key, np.nan)):
                    w[key] = last_w[key]
                else:
                    w[key] = DEFAULTS[key]
        for key in DEFAULTS:
            if isinstance(w.get(key), np.number):
                w[key] = float(w[key])
        w["dt"] = dt.strftime("%Y-%m-%d %H:%M")
        weather_list.append(w)

    # 평균 기반 target_weather 생성
    all_keys = DEFAULTS.keys()
    target_weather = {}
    for key in all_keys:
        if key != "ALLSKY_SFC_SW_DWN":
            values = [w.get(key) for w in weather_list if w.get(key) not in [None, -999] and not np.isnan(w.get(key))]
            if values:
                avg_val = float(np.mean(values))
                target_weather[key] = avg_val
            else:
                target_weather[key] = DEFAULTS[key]
                print(f"⚠️ {key} 평균값 없음, DEFAULT 사용 → {DEFAULTS[key]}")

    # --- ✅ 일조량 처리 로직 수정: 최대값 사용, -999 필터링 ---
    ymd_for_max = end_dt.strftime("%Y%m%d")
    daily_insolation_data = hourly_cache.get(ymd_for_max, {}).get("ALLSKY_SFC_SW_DWN", {})
    if daily_insolation_data:
        valid_values = [
            float(v) for v in daily_insolation_data.values()
            if v is not None and v != -999 and not np.isnan(v)
        ]
        if valid_values:
            max_val = float(np.max(valid_values))
            target_weather["ALLSKY_SFC_SW_DWN"] = max_val
            print(f"✅ 일조량은 하루 중 최대값({max_val})을 사용합니다.")
        else:
            target_weather["ALLSKY_SFC_SW_DWN"] = DEFAULTS["ALLSKY_SFC_SW_DWN"]
            print(f"⚠️ 일조량 데이터가 모두 결측치라 DEFAULT({DEFAULTS['ALLSKY_SFC_SW_DWN']})를 사용합니다.")
    else:
        target_weather["ALLSKY_SFC_SW_DWN"] = DEFAULTS["ALLSKY_SFC_SW_DWN"]
        print(f"⚠️ {ymd_for_max} 날짜의 일조량 데이터가 캐시에 없어 DEFAULT({DEFAULTS['ALLSKY_SFC_SW_DWN']})를 사용합니다.")

    print(" 최종 ALLSKY_SFC_SW_DWN:", target_weather["ALLSKY_SFC_SW_DWN"])
    print(" FWI 입력값:", target_weather.get("T2M"), target_weather.get("RH2M"), target_weather.get("WS10M"), target_weather.get("PRECTOTCORR"))

    fwi = fwi_calc(
        T=target_weather.get("T2M", 20),
        RH=target_weather.get("RH2M", 40),
        W=target_weather.get("WS10M", 3),
        P=target_weather.get("PRECTOTCORR", 0),
        month=end_dt.month,
        consecutive_dry_days=precip_feats.get("consecutive_dry_days_start", 0),
        total_precip_30d=precip_feats.get("total_precip_30d_start", 0)
    )

    if fwi.get("DMC", -999) == -999 or fwi.get("DC", -999) == -999:
        print("⚠️ FWI 계산에서 DMC 또는 DC 값이 -999입니다. 입력값이 비정상일 수 있습니다.")

    fwi_no_suffix = {k: round(float(v), 2) for k, v in fwi.items()}
    fwi_with_suffix = {f"{k}_0h": round(float(v), 2) for k, v in fwi.items()}

    derived = {
        "dry_windy_combo": int(target_weather.get("RH2M", 0) < 35 and target_weather.get("WS10M", 0) > 5),
        "fuel_combo": float(fwi_no_suffix.get("FFMC", 0) * fwi_no_suffix.get("ISI", 0)),
        "potential_spread_index": float(fwi_no_suffix.get("ISI", 0) * fwi_no_suffix.get("FWI", 0)),
        "terrain_var_effect": 0.0,
        "wind_steady_flag": int(np.nanmax([w.get("WS10M", 0) for w in weather_list]) -
                                np.nanmin([w.get("WS10M", 0) for w in weather_list]) < 1.5),
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
        "lat": lat, "lng": lon,
        "start_dt": start_dt.strftime("%Y-%m-%d %H:%M"),
        "end_dt": end_dt.strftime("%Y-%m-%d %H:%M"),
        "duration_hours": round((end_dt - start_dt).total_seconds() / 3600, 2),
        "total_duration_hours": round((end_dt - start_dt).total_seconds() / 3600, 2),
        "weather_timeseries": weather_list,
        **{k: target_weather.get(k, -999) for k in DEFAULTS},
        **{f"{k}_0h": target_weather.get(k, -999) for k in DEFAULTS},
        **precip_feats,
        **fwi_no_suffix,
        **fwi_with_suffix,
        **derived,
        **season_flags,
        "success": True
    }
    return result

if __name__ == "__main__":
    lat = float(sys.argv[1])
    lon = float(sys.argv[2])
    timestamp = datetime.datetime.now()
    result = fetch_all_weather_features(lat, lon, timestamp)
    print(json.dumps(result, indent=2, ensure_ascii=False))