import datetime
import sys
import json
import numpy as np
import pandas as pd
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
    points = []
    curr = start_dt
    while curr <= end_dt:
        points.append(curr)
        curr += datetime.timedelta(hours=interval_hours)
    return points

def fetch_all_weather_features(lat, lon, timestamp, offset_days=7):
    target_dt = timestamp - datetime.timedelta(days=offset_days)
    end_dt = target_dt
    start_dt = end_dt - datetime.timedelta(hours=42)

    precip_feats = make_precip_features(lat, lon, end_dt)
    time_points = make_time_points(start_dt, end_dt)
    weather_list = []

    DEFAULTS = {
        "T2M": 15.0, "RH2M": 50.0, "WS2M": 1.5, "WD2M": 180.0,
        "WS10M": 2.0, "WD10M": 180.0, "PRECTOTCORR": 0.0, "PS": 1013.0,
        "ALLSKY_SFC_SW_DWN": 0.0
    }

    hourly_cache = {}
    all_dates = sorted(list(set([dt.strftime("%Y%m%d") for dt in time_points])))
    
    for ymd in all_dates:
        raw = _fetch_nasa_raw_hourly_data_for_day(lat, lon, ymd)
        hourly_cache[ymd] = raw or {}

    for dt in time_points:
        ymd = dt.strftime("%Y%m%d")
        hour_key = f"{ymd}{dt.strftime('%H').zfill(2)}"
        w = {}
        for key in DEFAULTS:
            w[key] = hourly_cache[ymd].get(key, {}).get(hour_key, np.nan)

        last_w = weather_list[-1] if weather_list else {}
        for key in DEFAULTS:
            if np.isnan(w.get(key, np.nan)):
                fb_map = {'WD10M': 'WD2M', 'WD2M': 'WD10M', 'WS10M': 'WS2M', 'WS2M': 'WS10M'}
                fb = fb_map.get(key)
                if fb and not np.isnan(w.get(fb, np.nan)):
                    w[key] = w[fb]
                elif not np.isnan(last_w.get(key, np.nan)):
                    w[key] = last_w[key]
                else:
                    w[key] = DEFAULTS[key]
        
        for key in DEFAULTS:
            if isinstance(w.get(key), np.number):
                w[key] = float(w[key])
        w["dt"] = dt
        weather_list.append(w)

    ts_df = pd.DataFrame(weather_list).set_index("dt").sort_index()
    ts_df.fillna(method='ffill', inplace=True)
    ts_df.fillna(method='bfill', inplace=True)

    result = {}
    
    stats_df = ts_df.tail(24)
    for col in ['T2M', 'RH2M', 'WS10M', 'WD10M', 'PRECTOTCORR', 'PS', 'ALLSKY_SFC_SW_DWN']:
        result[f'{col}_mean'] = stats_df[col].mean()
        result[f'{col}_std'] = stats_df[col].std()

    for hours in range(3, 43, 3):
        end_time = ts_df.index.max()
        start_time = end_time - pd.Timedelta(hours=hours)
        prev_start_time = start_time - pd.Timedelta(hours=3)
        
        current_period = ts_df[(ts_df.index > start_time) & (ts_df.index <= end_time)]
        prev_period = ts_df[(ts_df.index > prev_start_time) & (ts_df.index <= start_time)]

        for param, name in [('T2M', 'temp'), ('RH2M', 'humidity'), ('WS10M', 'wind_speed')]:
            current_mean = current_period[param].mean()
            prev_mean = prev_period[param].mean()
            change = current_mean - prev_mean if pd.notna(current_mean) and pd.notna(prev_mean) else 0
            result[f'{name}_change_{hours-3}_{hours}h'] = change
        
        wd_current = current_period['WD10M'].mean()
        wd_prev = prev_period['WD10M'].mean()
        wd_change = 180 - abs(abs(wd_current - wd_prev) - 180) if pd.notna(wd_current) and pd.notna(wd_prev) else 0
        result[f'wind_direction_change_{hours-3}_{hours}h'] = wd_change

    target_weather = ts_df.iloc[-1].to_dict()
    fwi = fwi_calc(
        T=target_weather.get("T2M", 20), RH=target_weather.get("RH2M", 40),
        W=target_weather.get("WS10M", 3), P=target_weather.get("PRECTOTCORR", 0),
        month=end_dt.month
    )
    
    result.update({f"{k}_0h": v for k, v in target_weather.items()})
    result.update({f"{k}_0h": v for k, v in fwi.items()})
    
    result.update({
        "dry_windy_combo": int(target_weather.get("RH2M", 0) < 35 and target_weather.get("WS10M", 0) > 5),
        "fuel_combo": float(fwi.get("FFMC", 0) * fwi.get("ISI", 0)),
        "potential_spread_index": float(fwi.get("ISI", 0) * fwi.get("FWI", 0)),
        "wind_steady_flag": int(ts_df['WS10M'].max() - ts_df['WS10M'].min() < 1.5),
        "dry_to_rain_ratio_30d": (
            precip_feats.get("dry_days_30d_start", 0) /
            max(precip_feats.get("total_precip_30d_start", 0), 1e-2)
        ),
        "ndvi_stress": 0.6, 
        "terrain_var_effect": 0.0 
    })

    result.update(precip_feats)
    result.update({
        "startyear": end_dt.year, "startmonth": end_dt.month, "startday": end_dt.day,
        "is_spring": int(end_dt.month in [3, 4, 5]),
        "is_summer": int(end_dt.month in [6, 7, 8]),
        "is_autumn": int(end_dt.month in [9, 10, 11]),
        "is_winter": int(end_dt.month in [12, 1, 2])
    })
    
    result["success"] = True
    return result

if __name__ == "__main__":
    lat = float(sys.argv[1])
    lon = float(sys.argv[2])
    timestamp = datetime.datetime.now()
    result = fetch_all_weather_features(lat, lon, timestamp)
    print(json.dumps(result, indent=2, ensure_ascii=False))