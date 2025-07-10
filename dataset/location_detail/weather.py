import pandas as pd
import requests
from datetime import datetime, timedelta
import time

INPUT_XLSX_PATH = "./location_detail/gangwon_fire_data.xlsx"
OUTPUT_CSV_PATH = "./gangwon_fire_data_with_climate_hourly_segmented.csv"
NASA_POWER_API_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
API_SLEEP = 1.0  # API 호출 간 딜레이 (초)

PARAMS = [
    "T2M", "RH2M", "WS2M", "WS10M",
    "WD2M", "WD10M", "PRECTOTCORR", "PS", "ALLSKY_SFC_SW_DWN"
]

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
            # PRECTOTCORR (강수량)은 합산, 나머지는 평균
            agg[param] = sum(vals) if param == "PRECTOTCORR" else sum(vals) / len(vals)
        else:
            agg[param] = None
    return agg

def floor_to_hour(dt):
    return dt.replace(minute=0, second=0, microsecond=0)

def ceil_to_hour(dt):
    if dt.minute > 0 or dt.second > 0 or dt.microsecond > 0:
        dt = dt + timedelta(hours=1)
    return dt.replace(minute=0, second=0, microsecond=0)

def main():
    df = pd.read_excel(INPUT_XLSX_PATH, engine='openpyxl')

    df['start_datetime'] = df.apply(lambda r: parse_datetime(r['fire_start_date'], r['start_time']), axis=1)
    df['end_datetime'] = df.apply(lambda r: parse_datetime(r['fire_end_date'], r['end_time']), axis=1)

    max_segments = 10  
    for seg_i in range(1, max_segments + 1):
        for param in PARAMS:
            col_name = f"{param.lower()}_{seg_i}"
            df[col_name] = None

    for idx, row in df.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        start_dt = row['start_datetime']
        end_dt = row['end_datetime']

        if None in (lat, lon, start_dt, end_dt):
            print(f"[{idx}] 좌표/시간 정보 부족, 스킵")
            continue

        start_dt_floor = floor_to_hour(start_dt)
        end_dt_ceil = ceil_to_hour(end_dt)
        duration_hours = (end_dt_ceil - start_dt_floor).total_seconds() / 3600

        print(f"[{idx}] 처리 중 (위도: {lat}, 경도: {lon}) 구간: {start_dt_floor} ~ {end_dt_ceil} ({duration_hours:.2f}시간)")

        segments = []

        # 시작 구간 (start_dt_floor ~ start_dt_floor + 1시간)
        start_segment_end = min(start_dt_floor + timedelta(hours=1), end_dt_ceil)
        segments.append((start_dt_floor, start_segment_end))

        # 종료 구간 (end_dt_ceil - 1시간 ~ end_dt_ceil)
        end_segment_start = max(end_dt_ceil - timedelta(hours=1), start_segment_end)

        # 중간 구간 (3시간 단위)
        cur_start = start_segment_end
        while cur_start < end_segment_start:
            cur_end = min(cur_start + timedelta(hours=3), end_segment_start)
            segments.append((cur_start, cur_end))
            cur_start = cur_end

        # 종료 구간 추가 (종료 1시간 구간)
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
                if col_name not in df.columns:
                    df[col_name] = None  # 컬럼 없으면 생성

                df.at[idx, col_name] = agg.get(param)

            time.sleep(API_SLEEP)

    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\n✅ 기후 데이터 시간대별 분할 포함 CSV 저장 완료: {OUTPUT_CSV_PATH}")

def reprocess_failed_rows(excel_path, failed_idx_list, output_csv_path):
    df = pd.read_excel(excel_path, engine='openpyxl')

    for idx in failed_idx_list:
        row = df.loc[idx]
        lat = row['latitude']
        lon = row['longitude']

        # 시작, 종료 날짜와 시간 합쳐서 datetime으로 생성
        start_dt = parse_datetime(row['fire_start_date'], row['start_time'])
        end_dt = parse_datetime(row['fire_end_date'], row['end_time'])

        if None in (lat, lon, start_dt, end_dt):
            print(f"[{idx}] 좌표/시간 정보 부족, 스킵")
            continue

        duration_hours = (end_dt - start_dt).total_seconds() / 3600
        print(f"[재처리] 행 {idx} 위도 {lat} 경도 {lon} 전체 구간: {start_dt} ~ {end_dt} ({duration_hours:.2f}시간)")

        if duration_hours <= 4:
            start_floor = floor_to_hour(start_dt)
            end_ceil = ceil_to_hour(end_dt)

            mid_point = start_dt + (end_dt - start_dt)/2
            mid_floor = floor_to_hour(mid_point)
            mid_ceil = ceil_to_hour(mid_point)

            segments = [
                (start_floor, start_floor + timedelta(hours=1)),
                (mid_floor, mid_ceil),
                (end_ceil - timedelta(hours=1), end_ceil)
            ]
        else:
            segments = []
            start_floor = floor_to_hour(start_dt)
            end_ceil = ceil_to_hour(end_dt)

            segments.append((start_floor, start_floor + timedelta(hours=1)))
            segments.append((end_ceil - timedelta(hours=1), end_ceil))

            cur_start = start_floor + timedelta(hours=1)
            while cur_start < end_ceil - timedelta(hours=1):
                cur_end = min(cur_start + timedelta(hours=3), end_ceil - timedelta(hours=1))
                segments.append((cur_start, cur_end))
                cur_start = cur_end

        daily_data_cache = {}

        for seg_i, (seg_start, seg_end) in enumerate(segments, start=1):
            date_key = seg_start.date()
            yyyymmdd = date_key.strftime("%Y%m%d")

            if yyyymmdd not in daily_data_cache:
                data = get_nasa_power_hourly_data_for_date(lat, lon, yyyymmdd)
                if data is None:
                    print(f"  [{idx}] NASA POWER API 실패 날짜: {yyyymmdd}")
                    data = {}
                daily_data_cache[yyyymmdd] = data

            data = daily_data_cache.get(yyyymmdd, {})
            agg = aggregate_data_for_interval(data, seg_start, seg_end)

            for param in PARAMS:
                col_name = f"{param.lower()}_{seg_i}"
                if col_name not in df.columns:
                    df[col_name] = None
                df.at[idx, col_name] = agg.get(param)

            sample_params = ['t2m', 'rh2m', 'ws2m']
            check_vals = {p: df.at[idx, f"{p}_{seg_i}"] for p in sample_params}
            print(f"    입력 확인 - 행 {idx} 구간 {seg_i}: {check_vals}")

            time.sleep(1.0)

    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 재처리 완료 및 CSV 저장: {output_csv_path}")


if __name__ == "__main__":
    failed_idx = [166, 435, 488, 649, 659, 758, 842, 843, 844, 845, 846]

    excel_path = './gangwon_fire_data_with_climate_hourly_segmented.xlsx'
    output_csv_path = './gangwon_fire_data_with_climate_hourly_segmented_updated.csv'

    reprocess_failed_rows(excel_path, failed_idx, output_csv_path)