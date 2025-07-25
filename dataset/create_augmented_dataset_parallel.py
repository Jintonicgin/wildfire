import pandas as pd
import numpy as np
import datetime
import requests
import time
import sys
import concurrent.futures
from tqdm import tqdm

# fwi_calc.py 스크립트에서 fwi_calc 함수를 가져옵니다.
try:
    from fwi_calc import fwi_calc
except ImportError:
    print("오류: fwi_calc.py를 찾을 수 없습니다. 이 스크립트를 dataset 디렉토리 내에서 실행해 주세요.")
    sys.exit(1)

# --- 설정 ---
SOURCE_DATA_PATH = "gangwon_fire_data_full_merged.csv"
OUTPUT_DATA_PATH = "gangwon_fire_data_augmented_parallel.csv"
API_DELAY = 0.1 # 병렬 처리 시에는 지연을 줄일 수 있습니다.
MAX_WORKERS = 8   # 동시에 실행할 프로세스 수 (CPU 코어 수에 맞게 조절)

# --- NASA POWER API 헬퍼 함수 (프로세스 간에 공유될 수 있도록 전역 스코프에 정의) ---
def fetch_nasa_hourly_weather(lat, lng, yyyymmdd, hour_str, max_retry=3):
    """특정 위경도와 시각의 시간당 날씨 데이터를 NASA POWER API로부터 가져옵니다."""
    url = (
        f"https://power.larc.nasa.gov/api/temporal/hourly/point?"
        "parameters=T2M,RH2M,WS10M,WD10M,PRECTOTCORR"
        f"&community=RE&longitude={lng}&latitude={lat}&start={yyyymmdd}&end={yyyymmdd}&format=JSON"
    )
    for attempt in range(max_retry):
        try:
            # 짧은 지연을 각 요청 전에 추가하여 API 서버 부하를 줄입니다.
            time.sleep(API_DELAY)
            res = requests.get(url, timeout=20)
            res.raise_for_status()
            data = res.json().get("properties", {}).get("parameter", {})
            if not data:
                return None
            
            hour_key = f"{yyyymmdd}{hour_str.zfill(2)}"
            
            weather_data = {
                "T2M": data.get("T2M", {}).get(hour_key, np.nan),
                "RH2M": data.get("RH2M", {}).get(hour_key, np.nan),
                "WS10M": data.get("WS10M", {}).get(hour_key, np.nan),
                "WD10M": data.get("WD10M", {}).get(hour_key, np.nan),
                "PRECTOTCORR": data.get("PRECTOTCORR", {}).get(hour_key, np.nan),
            }
            return weather_data
        except requests.exceptions.RequestException:
            if attempt == max_retry - 1:
                return None # 최종 실패

def process_task(task_info):
    """하나의 시간대(task)에 대한 모든 피처를 계산하는 작업 함수"""
    index, row, duration_hours = task_info
    
    current_timestamp = row['start_datetime'] + datetime.timedelta(hours=duration_hours)
    
    # 날씨/FWI 피처 가져오기
    features = get_features_for_timestamp(row['latitude'], row['longitude'], current_timestamp)

    if features is None:
        return None

    # 선형 보간법으로 중간 피해 면적 추정
    estimated_damage_area = (row['final_damage_area_ha'] / row['total_duration_hours']) * duration_hours

    return {
        'original_fire_id': index,
        'start_datetime': row['start_datetime'],
        'total_duration_hours': row['total_duration_hours'],
        'final_damage_area_ha': row['final_damage_area_ha'],
        'duration_hours': duration_hours,
        'estimated_damage_area': estimated_damage_area,
        'timestamp': current_timestamp,
        'lat': row['latitude'],
        'lng': row['longitude'],
        **features
    }

def get_features_for_timestamp(lat, lon, timestamp):
    """특정 시점의 날씨 및 FWI 피처를 계산합니다."""
    yyyymmdd = timestamp.strftime("%Y%m%d")
    hour_str = timestamp.strftime("%H")
    
    weather = fetch_nasa_hourly_weather(lat, lon, yyyymmdd, hour_str)
    
    if weather is None or any(pd.isna(v) for v in weather.values()):
        return None

    fwi_result = fwi_calc(
        T=weather.get("T2M"), RH=weather.get("RH2M"),
        W=weather.get("WS10M"), P=weather.get("PRECTOTCORR"),
        month=timestamp.month
    )
    
    return {**weather, **fwi_result}

# --- 메인 실행 로직 ---
def main():
    print(f"1. 원본 데이터 로딩: {SOURCE_DATA_PATH}")
    try:
        df = pd.read_csv(SOURCE_DATA_PATH)
    except FileNotFoundError:
        print(f"오류: 원본 데이터 파일 '{SOURCE_DATA_PATH}'을 찾을 수 없습니다.")
        return

    df['start_datetime'] = pd.to_datetime(df['start_datetime'])
    df['end_datetime'] = pd.to_datetime(df['end_datetime'])
    df['total_duration_hours'] = (df['end_datetime'] - df['start_datetime']).dt.total_seconds() / 3600
    df = df[df['total_duration_hours'] > 0].copy()
    # 컬럼명 통일: 원본 데이터의 컬럼명을 스크립트에서 사용하는 이름으로 변경합니다.
    df.rename(columns={
        'damage_area': 'final_damage_area_ha'
    }, inplace=True)

    # 병렬 처리할 모든 작업 목록 생성
    tasks = []
    for index, row in df.iterrows():
        total_duration = int(np.ceil(row['total_duration_hours']))
        for duration_hours in range(1, total_duration + 1):
            tasks.append((index, row, duration_hours))
    
    print(f"2. 데이터 증강 시작 (총 {len(tasks)}개 시점 데이터, {MAX_WORKERS}개 프로세서 사용)")
    
    augmented_data = []
    # ProcessPoolExecutor를 사용하여 병렬 처리
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # tqdm을 사용하여 진행 상황 표시
        results = list(tqdm(executor.map(process_task, tasks), total=len(tasks), desc="시간대별 데이터 처리 중"))

    # 성공적으로 처리된 결과만 필터링
    augmented_data = [res for res in results if res is not None]

    if not augmented_data:
        print("오류: 증강된 데이터가 생성되지 않았습니다. API 요청에 문제가 있었을 수 있습니다.")
        return

    print(f"3. 증강된 데이터프레임 생성 (총 {len(augmented_data)}개 시점 데이터)")
    augmented_df = pd.DataFrame(augmented_data)
    
    print(f"4. 새로운 데이터셋 저장: {OUTPUT_DATA_PATH}")
    augmented_df.to_csv(OUTPUT_DATA_PATH, index=False, encoding='utf-8-sig')
    
    print("작업 완료!")

if __name__ == "__main__":
    # 멀티프로세싱을 안전하게 사용하기 위해 main 함수 호출을 이 블록 안에 둡니다.
    main()
