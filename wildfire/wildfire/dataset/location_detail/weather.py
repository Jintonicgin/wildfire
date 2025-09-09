import pandas as pd
import requests
from datetime import datetime, timedelta
import time

INPUT_XLSX_PATH = "./location_detail/gangwon_fire_data.xlsx"
OUTPUT_CSV_PATH = "./gangwon_fire_data_with_climate_hourly_segmented.csv"
NASA_POWER_API_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
NASA_POWER_DAILY_API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point" 
API_SLEEP = 1.0 

PARAMS = [
    "T2M", "RH2M", "WS2M", "WS10M",
    "WD2M", "WD10M", "PRECTOTCORR", "PS", "ALLSKY_SFC_SW_DWN"
]

RAIN_PERIODS = [7, 14, 30, 60, 90]

def parse_datetime(date_str, time_str):
    if pd.isna(date_str) or pd.isna(time_str):
        return None
    try:
        dt_str = f"{str(date_str).split()[0]} {time_str}"
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        return dt
    except Exception as e:
        print(f"날짜 변환 실패: {e} ({date_str} {time_str})")
        return None

def get_nasa_power_hourly_data_for_date(lat, lon, yyyymmdd, retry=3):
    params = {
        "start": yyyymmdd,
        "end": yyyymmdd,
        "latitude": lat,
        "longitude": lon,
        "community": "RE",
        "parameters": ",".join(PARAMS),
        "format": "JSON"
    }
    for attempt in range(1, retry + 1):
        try:
            response = requests.get(NASA_POWER_API_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("properties", {}).get("parameter", {})
        except Exception as e:
            print(f"NASA API 실패 (시도 {attempt}): {e} - 날짜 {yyyymmdd}")
            if attempt == retry:
                return None
            time.sleep(2)
    return None

def get_nasa_power_daily_rainfall(lat, lon, start_date, end_date, retry=3):
    """일 단위 누적 강수량, 무강수일수 계산용 데이터 호출"""
    params = {
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "latitude": lat,
        "longitude": lon,
        "community": "RE",
        "parameters": "PRECTOTCORR",
        "format": "JSON"
    }
    for attempt in range(1, retry + 1):
        try:
            response = requests.get(NASA_POWER_DAILY_API_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("properties", {}).get("parameter", {}).get("PRECTOTCORR", {})
        except Exception as e:
            print(f"NASA Daily API 실패 (시도 {attempt}): {e} - {start_date} ~ {end_date}")
            if attempt == retry:
                return None
            time.sleep(2)
    return None

def get_hour_keys_in_range(start_dt, end_dt):
    keys = []
    current = start_dt.replace(minute=0, second=0, microsecond=0)
    while current < end_dt:
        keys.append(current.strftime("%Y%m%d%H"))
        current += timedelta(hours=1)
    return keys

def aggregate_data_for_interval(data, start_dt, end_dt):
    agg = {}
    hour_keys = get_hour_keys_in_range(start_dt, end_dt)
    for param in PARAMS:
        vals = []
        if param not in data:
            agg[param] = None
            continue
        for key in hour_keys:
            val = data[param].get(key)
            if val is not None and val != "":
                vals.append(val)
        if vals:
            agg[param] = sum(vals) if param == "PRECTOTCORR" else sum(vals) / len(vals)
        else:
            agg[param] = None
    return agg

def floor_to_hour_round_down(dt):
    return dt.replace(minute=0, second=0, microsecond=0)

def ceil_to_hour_round_up(dt):
    if dt.minute > 0 or dt.second > 0 or dt.microsecond > 0:
        dt = dt + timedelta(hours=1)
    return dt.replace(minute=0, second=0, microsecond=0)

def calculate_rainfall_and_dry_days(lat, lon, fire_date):
    """산불 발생일 기준으로 과거 여러 기간 누적강수량과 무강수일수 계산"""
    results = {}
    for days in RAIN_PERIODS:
        start_date = fire_date - timedelta(days=days)
        end_date = fire_date - timedelta(days=1)  # 산불 발생일 바로 전날까지
        daily_rain_data = get_nasa_power_daily_rainfall(lat, lon, start_date, end_date)
        if daily_rain_data is None:
            results[f"rainfall_cumulative_{days}d"] = None
            results[f"dry_days_{days}d"] = None
            print(f"[{lat}, {lon}] {days}일 누적 강수량/무강수일수 데이터 없음")
            continue

        rainfall_vals = []
        dry_days_count = 0
        for date_str, val in daily_rain_data.items():
            if val is None:
                continue
            rainfall_vals.append(val)
            if val == 0:
                dry_days_count += 1

        results[f"rainfall_cumulative_{days}d"] = sum(rainfall_vals) if rainfall_vals else None
        results[f"dry_days_{days}d"] = dry_days_count
        print(f"[{lat}, {lon}] {days}일 누적 강수량: {results[f'rainfall_cumulative_{days}d']}, 무강수일수: {results[f'dry_days_{days}d']}")

    return results

def main():
    df = pd.read_excel(INPUT_XLSX_PATH, engine='openpyxl')

    df['start_datetime'] = df.apply(lambda r: parse_datetime(r['fire_start_date'], r['start_time']), axis=1)
    df['end_datetime'] = df.apply(lambda r: parse_datetime(r['fire_end_date'], r['end_time']), axis=1)

    max_segments = 10  
    for seg_i in range(1, max_segments + 1):
        for param in PARAMS:
            col_name = f"{param.lower()}_{seg_i}"
            if col_name not in df.columns:
                df[col_name] = None

    for idx, row in df.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        start_dt = row['start_datetime']
        end_dt = row['end_datetime']
        fire_date = pd.to_datetime(row['fire_start_date'])

        if None in (lat, lon, start_dt, end_dt):
            print(f"[{idx}] 좌표/시간 정보 부족, 스킵")
            continue

        start_dt_floor = floor_to_hour_round_down(start_dt)
        end_dt_ceil = ceil_to_hour_round_up(end_dt)
        duration_hours = (end_dt_ceil - start_dt_floor).total_seconds() / 3600

        print(f"[{idx}] 처리 중 (위도: {lat}, 경도: {lon}) 구간: {start_dt_floor} ~ {end_dt_ceil} ({duration_hours:.2f}시간)")

        segments = []

        start_segment_end = min(start_dt_floor + timedelta(hours=1), end_dt_ceil)
        segments.append((start_dt_floor, start_segment_end))

        end_segment_start = max(end_dt_ceil - timedelta(hours=1), start_segment_end)

        cur_start = start_segment_end
        while cur_start < end_segment_start:
            remaining = (end_segment_start - cur_start).total_seconds() / 3600
            if remaining < 3:
                break
            cur_end = cur_start + timedelta(hours=3)
            segments.append((cur_start, cur_end))
            cur_start = cur_end

        if end_segment_start < end_dt_ceil:
            segments.append((end_segment_start, end_dt_ceil))

        daily_data_cache = {}

        for seg_i, (seg_start, seg_end) in enumerate(segments, start=1):
            date_key = seg_start.date()
            yyyymmdd = date_key.strftime("%Y%m%d")

            if yyyymmdd not in daily_data_cache:
                print(f"  NASA POWER API 호출 날짜: {yyyymmdd}")
                data = get_nasa_power_hourly_data_for_date(lat, lon, yyyymmdd)
                if data is None:
                    print(f"  [{idx}] NASA POWER API 실패 날짜: {yyyymmdd}")
                    data = {}
                daily_data_cache[yyyymmdd] = data

            data = daily_data_cache.get(yyyymmdd, {})
            agg = aggregate_data_for_interval(data, seg_start, seg_end)
            print(f"  구간 {seg_i}: {seg_start} ~ {seg_end} 데이터: {agg}")

            for param in PARAMS:
                col_name = f"{param.lower()}_{seg_i}"
                df.at[idx, col_name] = agg.get(param)

            time.sleep(API_SLEEP)

        rain_dry_results = calculate_rainfall_and_dry_days(lat, lon, fire_date)
        for key, val in rain_dry_results.items():
            df.at[idx, key] = val

    df = df.copy()

    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\n✅ 기후 데이터 시간대별 분할 및 누적강수량/무강수일수 포함 CSV 저장 완료: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()