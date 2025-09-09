from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests
from datetime import datetime, timedelta
import time

# 📁 파일 경로
INPUT_CSV_PATH = "./gangwon_fire_data_with_wind_vectors.csv"
OUTPUT_CSV_PATH = "./gangwon_fire_data_with_climate_hourly_parallel.csv"
NASA_POWER_API_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

# 📌 파라미터
API_SLEEP = 0.1  # 호출 간 간격
PARAMS = [
    "T2M", "RH2M", "WS2M", "WS10M",
    "WD2M", "WD10M", "PRECTOTCORR", "PS", "ALLSKY_SFC_SW_DWN"
]

# 🔧 도우미 함수
def floor_to_hour_round_down(dt):
    return dt.replace(minute=0, second=0, microsecond=0)

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
            print(f"❌ NASA API 호출 실패 (시도 {attempt}): {e} - 날짜: {yyyymmdd}")
            if attempt == retry:
                return None
            time.sleep(2)
    return None

def get_3hour_prior_data(row):
    lat = row["latitude"]
    lon = row["longitude"]
    fire_date = str(row["fire_start_date"]).split()[0]
    start_time = row["start_time"]

    try:
        fire_dt = datetime.strptime(f"{fire_date} {start_time}", "%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print(f"❌ datetime 변환 실패: {e} ({fire_date} {start_time})")
        return None

    fire_dt_floor = floor_to_hour_round_down(fire_dt)
    prior_dt = fire_dt_floor - timedelta(hours=3)

    yyyymmdd = prior_dt.strftime("%Y%m%d")
    keys = [(prior_dt + timedelta(hours=i)).strftime("%Y%m%d%H") for i in range(3)]

    print(f"[{row.name+1}] 📍 ({lat:.4f}, {lon:.4f}) | 시작 시각: {fire_dt_floor} → 키: {', '.join(keys)}")

    data = get_nasa_power_hourly_data_for_date(lat, lon, yyyymmdd)
    if data is None:
        print(f"   ❌ 데이터 없음, 스킵")
        return None

    result = {"index": row.name}
    for param in PARAMS:
        values = []
        for k in keys:
            v = data.get(param, {}).get(k)
            if v is not None:
                values.append(v)
        if values:
            result[param] = sum(values) / len(values)
        else:
            result[param] = None
    return result

# 🔁 병렬 처리 및 재시도 포함 메인 함수
def main():
    df = pd.read_csv(INPUT_CSV_PATH)
    results = []
    failed_indices = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_index = {executor.submit(get_3hour_prior_data, row): row.name for idx, row in df.iterrows()}
        for future in as_completed(future_to_index):
            result = future.result()
            if result is not None:
                results.append(result)
            else:
                failed_indices.append(future_to_index[future])

    if failed_indices:
        print(f"\n🔁 재시도 시작 (실패 {len(failed_indices)}건)")
        for idx in failed_indices:
            row = df.loc[idx]
            result = get_3hour_prior_data(row)
            if result:
                results.append(result)

    if not results:
        print("❌ 결과 없음")
        return

    df_result = pd.DataFrame(results).set_index("index")
    df_merged = df.join(df_result, how="left")
    df_merged.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\n✅ 완료: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()