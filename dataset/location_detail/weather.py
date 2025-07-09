import pandas as pd
import requests
import time
from datetime import datetime

CORRECTED_COORDS_CSV_PATH = "./location_detail/gangwon_fire_data_with_coords.csv"
OUTPUT_CSV_PATH = "./gangwon_fire_data_with_climate.csv"

def get_nasa_power_data(lat, lon, date_str, max_retries=3):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "start": date_str,
        "end": date_str,
        "latitude": lat,
        "longitude": lon,
        "community": "AG",
        "parameters": "T2M,RH2M,WS2M,WD2M,PRECTOTCORR",
        "format": "JSON"
    }

    delay = 1
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            daily_data = data.get('properties', {}).get('parameter', {})
            result = {
                "temp_avg": daily_data.get('T2M', {}).get(date_str),
                "humidity": daily_data.get('RH2M', {}).get(date_str),
                "wind_speed": daily_data.get('WS2M', {}).get(date_str),
                "wind_dir": daily_data.get('WD2M', {}).get(date_str),
                "precip": daily_data.get('PRECTOTCORR', {}).get(date_str)
            }
            print(f"✅ NASA POWER API 성공: {lat}, {lon}, {date_str} -> {result}")
            return result
        except Exception as e:
            print(f"❌ NASA POWER API 요청 실패 ({lat}, {lon}, {date_str}) 시도 {attempt+1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                print(f"⏳ {delay}초 후 재시도...")
                time.sleep(delay)
                delay *= 2
            else:
                print("⚠️ 최대 재시도 횟수 초과, 다음으로 넘어갑니다.")
                return {
                    "temp_avg": None,
                    "humidity": None,
                    "wind_speed": None,
                    "wind_dir": None,
                    "precip": None
                }

def main():
    df = pd.read_csv(CORRECTED_COORDS_CSV_PATH, encoding="utf-8-sig")

    temp_avgs, humidities = [], []
    wind_speeds, wind_dirs = [], []
    precips = []

    for idx, row in df.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        fire_date = row['fire_date']
        print(f"\nProcessing {idx}: ({lat}, {lon}) / {fire_date}")

        if pd.notna(lat) and pd.notna(lon) and fire_date:
            date_str = datetime.strptime(fire_date, "%Y-%m-%d").strftime("%Y%m%d")
            climate = get_nasa_power_data(lat, lon, date_str)
            temp_avgs.append(climate["temp_avg"])
            humidities.append(climate["humidity"])
            wind_speeds.append(climate["wind_speed"])
            wind_dirs.append(climate["wind_dir"])
            precips.append(climate["precip"])
        else:
            print("⚠️ 좌표 또는 날짜 정보 부족으로 기후 데이터 수집 불가")
            temp_avgs.append(None)
            humidities.append(None)
            wind_speeds.append(None)
            wind_dirs.append(None)
            precips.append(None)

        time.sleep(0.2)

    df['temp_avg'] = temp_avgs
    df['humidity'] = humidities
    df['wind_speed'] = wind_speeds
    df['wind_dir'] = wind_dirs
    df['precipitation'] = precips

    print("\n=== 수집된 기후 데이터 예시 ===")
    print(df[['corrected_address', 'fire_date', 'temp_avg', 'humidity', 'wind_speed', 'wind_dir', 'precipitation']].head(20))

    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\n✅ 기후 데이터 포함 CSV 저장 완료: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()