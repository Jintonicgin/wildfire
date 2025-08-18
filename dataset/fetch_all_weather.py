import datetime
import sys
import json
import numpy as np
import pandas as pd
import traceback
from wildfire.dataset.fwi_calc import fwi_calc
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


def engineer_additional_features(result, time_intervals_for_engineer, end_dt):
    """🔧 학습 시와 완전히 동일한 피처 생성 (단순한 접미사 사용)"""

    # 1. 시간적 변화량 피처
    for i in range(len(time_intervals_for_engineer) - 1):
        t0, t1 = time_intervals_for_engineer[i], time_intervals_for_engineer[i + 1]
        for prefix in ["WS10M", "T2M", "RH2M"]:
            c1, c2 = f"{prefix}_{t0}h", f"{prefix}_{t1}h"
            new_col = f"{prefix.lower()}_change_{t0}_{t1}h"
            if c1 in result and c2 in result:
                result[new_col] = abs(result[c1] - result[c2])

    # 2. 🔧 핵심: 학습 시와 동일한 단순 접미사 사용 (_mean, _std, _max, _min)
    weather_params = ["T2M", "RH2M", "WS10M", "PRECTOTCORR", "PS", "ALLSKY_SFC_SW_DWN"]
    for param in weather_params:
        cols_to_avg = [f"{param}_{t}h" for t in time_intervals_for_engineer if f"{param}_{t}h" in result]
        if cols_to_avg:
            values = [result[c] for c in cols_to_avg]
            result[f"{param}_mean"] = np.mean(values)
            result[f"{param}_std"] = np.std(values)
            result[f"{param}_max"] = np.max(values)
            result[f"{param}_min"] = np.min(values)

    # 3. 조합 피처 (단순 접미사 사용)
    if "T2M_mean" in result and "RH2M_mean" in result and "WS10M_mean" in result:
        result["dryness_index"] = result["T2M_mean"] * (100 - result["RH2M_mean"]) / 100
        result["wind_humidity_ratio"] = result["WS10M_mean"] / (result["RH2M_mean"] + 1e-5)
        result["wind_temp_product"] = result["WS10M_mean"] * result["T2M_mean"]

    return result


def fetch_all_weather_features(lat, lon, timestamp, offset_days=7):
    target_dt = timestamp - datetime.timedelta(days=offset_days)
    end_dt = target_dt
    start_dt = end_dt - datetime.timedelta(hours=171) # 🔧 train.py와 동일하게 171시간 전까지 데이터 수집

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

    # 시간별 데이터 추출 (train.py와 동일한 간격)
    time_intervals_for_engineer = sorted(list(range(0, 172, 3)))
    for hour_val in time_intervals_for_engineer:
        for param in ["T2M", "RH2M", "WS10M", "WD10M", "PRECTOTCORR", "PS", "ALLSKY_SFC_SW_DWN"]:
            target_time_for_loc = end_dt - datetime.timedelta(hours=hour_val)
            time_diff = (ts_df.index - target_time_for_loc).to_series().abs()
            closest_time_pos = time_diff.argmin()
            result[f"{param}_{hour_val}h"] = ts_df.iloc[closest_time_pos][param]

    # 파생 피처 생성 (train.py와 동일한 로직)
    result = engineer_additional_features(result, time_intervals_for_engineer, end_dt)

    # FWI 계산
    target_weather = ts_df.iloc[-1].to_dict()
    fwi = fwi_calc(
        T=target_weather.get("T2M", 20), RH=target_weather.get("RH2M", 40),
        W=target_weather.get("WS10M", 3), P=target_weather.get("PRECTOTCORR", 0),
        month=end_dt.month
    )

    # 🔧 핵심 수정: 모델이 기대하는 피처 이름과 정확히 일치하도록 생성
    # 1. 현재 기상 조건에 _0h 접미사 추가
    for k, v in target_weather.items():
        result[f"{k}_0h"] = v
    
    # 2. FWI 결과에 _0h 접미사 추가
    for k, v in fwi.items():
        result[f"{k}_0h"] = v

    # 3. 'month' 피처 생성
    result["month"] = end_dt.month

    # 추가 지표들 (train.py와 동일)
    result.update({
        "dry_windy_combo": int(target_weather.get("RH2M", 0) < 35 and target_weather.get("WS10M", 0) > 5),
        "fuel_combo": float(fwi.get("FFMC", 0) * fwi.get("ISI", 0)),
        "potential_spread_index": float(fwi.get("ISI", 0) * fwi.get("FWI", 0)),
        "wind_steady_flag": int(ts_df['WS10M'].max() - ts_df['WS10M'].min() < 1.5),
        "dry_to_rain_ratio_30d": (
                precip_feats.get("dry_days_30d_start", 0) /
                max(precip_feats.get("total_precip_30d_start", 0), 1e-2)
        ),
        "ndvi_stress": 0.6, # Placeholder
        "terrain_var_effect": 0.0 # Placeholder
    })

    # 강수량 관련 피처
    result.update(precip_feats)

    # 날짜/시간 관련 피처 (train.py와 동일)
    result.update({
        "startyear": end_dt.year, "startday": end_dt.day,
        "is_spring": int(end_dt.month in [3, 4, 5]),
        "is_summer": int(end_dt.month in [6, 7, 8]),
        "is_autumn": int(end_dt.month in [9, 10, 11]),
        "is_winter": int(end_dt.month in [12, 1, 2])
    })

    result["success"] = True
    return result


def fetch_and_engineer_features(lat, lon, timestamp):
    """메인 함수: 호환 가능한 피처 생성"""
    features = fetch_all_weather_features(lat, lon, timestamp)
    if not features.get("success", False):
        raise ValueError("Failed to fetch and engineer weather features.")
    features.pop("success", None)
    return features