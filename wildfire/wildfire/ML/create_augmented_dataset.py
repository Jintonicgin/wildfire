
import pandas as pd
import numpy as np
import datetime
import requests
import time
import sys
from tqdm import tqdm

# fwi_calc.py 스크립트에서 fwi_calc 함수를 가져옵니다.
# 경로 문제가 발생할 경우, 이 스크립트를 dataset 디렉토리에서 실행해야 합니다.
try:
    from fwi_calc import fwi_calc
except ImportError:
    print("오류: fwi_calc.py를 찾을 수 없습니다. 이 스크립트를 dataset 디렉토리 내에서 실행해 주세요.")
    sys.exit(1)

# --- 설정 ---
SOURCE_DATA_PATH = "gangwon_fire_data_full_merged.csv"
OUTPUT_DATA_PATH = "gangwon_fire_data_augmented.csv"
# NASA POWER API 요청 사이의 지연 시간 (초). 너무 빠르면 API가 차단할 수 있음.
API_DELAY = 1.1

# --- NASA POWER API 헬퍼 함수 ---
def fetch_nasa_hourly_weather(lat, lng, yyyymmdd, hour_str, max_retry=5):
    """특정 위경도와 시각의 시간당 날씨 데이터를 NASA POWER API로부터 가져옵니다."""
    url = (
        f"https://power.larc.nasa.gov/api/temporal/hourly/point?"
        "parameters=T2M,RH2M,WS10M,WD10M,PRECTOTCORR"
        f"&community=RE&longitude={lng}&latitude={lat}&start={yyyymmdd}&end={yyyymmdd}&format=JSON"
    )
    for attempt in range(max_retry):
        try:
            res = requests.get(url, timeout=30)
            res.raise_for_status()
            data = res.json().get("properties", {}).get("parameter", {})
            # API는 가끔 빈 데이터를 반환하므로 확인이 필요합니다.
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
        except requests.exceptions.RequestException as e:
            print(f"  - API 요청 오류 (시도 {attempt + 1}/{max_retry}): {e}")
            if attempt < max_retry - 1:
                time.sleep(API_DELAY * (attempt + 1)) # 재시도 전 대기
            else:
                return None # 최종 실패

def get_features_for_timestamp(lat, lon, timestamp):
    """특정 시점의 날씨 및 FWI 피처를 계산합니다."""
    yyyymmdd = timestamp.strftime("%Y%m%d")
    hour_str = timestamp.strftime("%H")
    
    weather = fetch_nasa_hourly_weather(lat, lon, yyyymmdd, hour_str)
    
    if weather is None or any(pd.isna(v) for v in weather.values()):
        return None # 날씨 데이터 조회 실패 시 건너뜀

    fwi_result = fwi_calc(
        T=weather.get("T2M"),
        RH=weather.get("RH2M"),
        W=weather.get("WS10M"),
        P=weather.get("PRECTOTCORR"),
        month=timestamp.month
    )
    
    # 날씨와 FWI 결과를 합쳐서 반환
    all_features = {**weather, **fwi_result}
    return all_features

# --- 메인 실행 로직 ---
def main():
    print(f"1. 원본 데이터 로딩: {SOURCE_DATA_PATH}")
    try:
        df = pd.read_csv(SOURCE_DATA_PATH)
    except FileNotFoundError:
        print(f"오류: 원본 데이터 파일 '{SOURCE_DATA_PATH}'을 찾을 수 없습니다.")
        return

    # 필수 컬럼 확인
    required_cols = ['start_datetime', 'end_datetime', 'damage_area', 'latitude', 'longitude']
    if not all(col in df.columns for col in required_cols):
        print(f"오류: 원본 데이터에 필수 컬럼이 부족합니다. {required_cols}가 모두 필요합니다.")

    # 날짜/시간 컬럼을 datetime 객체로 변환
    df['start_datetime'] = pd.to_datetime(df['start_datetime'])
    df['end_datetime'] = pd.to_datetime(df['end_datetime'])

    # 총 진화 시간(시간 단위) 계산
    df['total_duration_hours'] = (df['end_datetime'] - df['start_datetime']).dt.total_seconds() / 3600
    # 0시간 또는 음수 시간인 이상치 데이터는 제외
    df = df[df['total_duration_hours'] > 0].copy()

    augmented_data = []
    
    print(f"2. 데이터 증강 시작 (총 {len(df)}개 산불 데이터)")
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="산불 데이터 처리 중"):
        total_duration = int(np.ceil(row['total_duration_hours']))
        final_damage_area = row['damage_area']
        
        # 1시간부터 총 진화 시간까지 1시간 간격으로 데이터 생성
        for duration_hours in range(1, total_duration + 1):
            
            current_timestamp = row['start_datetime'] + datetime.timedelta(hours=duration_hours)
            
            # 선형 보간법으로 중간 피해 면적 추정
            estimated_damage_area = (final_damage_area / total_duration) * duration_hours
            
            # 현재 시점의 날씨/FWI 피처 가져오기
            features = get_features_for_timestamp(row['latitude'], row['longitude'], current_timestamp)
            
            # API 요청 간 지연
            time.sleep(API_DELAY)

            if features is None:
                # print(f"  - 경고: {row['start_datetime']} 산불의 {duration_hours}시간 경과 시점 날씨 조회 실패. 건너뜁니다.")
                continue

            new_row = {
                # 원본 데이터의 주요 정보 유지
                'original_fire_id': index,
                'start_datetime': row['start_datetime'],
                'total_duration_hours': total_duration,
                'final_damage_area_ha': final_damage_area,
                
                # 증강된 시계열 정보
                'duration_hours': duration_hours,
                'estimated_damage_area': estimated_damage_area,
                'timestamp': current_timestamp,
                
                # 위치 정보
                'lat': row['latitude'],
                'lng': row['longitude'],
                
                # 날씨/FWI 피처
                **features
            }
            augmented_data.append(new_row)

    if not augmented_data:
        print("오류: 증강된 데이터가 생성되지 않았습니다. API 요청에 문제가 있었을 수 있습니다.")
        return

    print(f"3. 증강된 데이터프레임 생성 (총 {len(augmented_data)}개 시점 데이터)")
    augmented_df = pd.DataFrame(augmented_data)
    
    print(f"4. 새로운 데이터셋 저장: {OUTPUT_DATA_PATH}")
    augmented_df.to_csv(OUTPUT_DATA_PATH, index=False, encoding='utf-8-sig')
    
    print("작업 완료!")

if __name__ == "__main__":
    main()
