import pandas as pd
import datetime
import sys
import os
import numpy as np

# weather_db_manager.py 모듈을 임포트하기 위해 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from weather_db_manager import (
    get_db_connection, drop_all_tables, create_all_tables,
    insert_gee_features, insert_nasa_hourly_weather, insert_nasa_daily_precip
)

# CSV 파일 경로
CSV_FILE_PATH = "/Users/mmymacymac/Developer/Projects/eclipse/WildFire/dataset/gangwon_fire_data_full_merged.csv"

def populate_gee_features(df):
    print("\n[GEE_FEATURES] 데이터 삽입 시작...")
    # 위도, 경도를 소수점 6자리로 반올림하여 고유한 위치를 식별
    unique_locations = df[['latitude', 'longitude', 'ndvi_pre_fire_latest', 'elevation', 'slope', 'aspect']].copy()
    unique_locations['latitude'] = unique_locations['latitude'].round(6)
    unique_locations['longitude'] = unique_locations['longitude'].round(6)
    unique_locations = unique_locations.drop_duplicates(subset=['latitude', 'longitude'])
    
    gee_features_list = []
    for index, row in unique_locations.iterrows():
        try:
            gee_features_list.append({
                "lat": float(row['latitude']),
                "lon": float(row['longitude']),
                "ndvi_before": float(row['ndvi_pre_fire_latest']) if pd.notna(row['ndvi_pre_fire_latest']) else None,
                "treecover_pre_fire_5x5": None, # CSV에 없음
                "elevation_mean": float(row['elevation']) if pd.notna(row['elevation']) else None,
                "elevation_min": None, # CSV에 없음
                "elevation_std": None, # CSV에 없음
                "slope_mean": float(row['slope']) if pd.notna(row['slope']) else None,
                "slope_min": None, # CSV에 없음
                "slope_max": None, # CSV에 없음
                "slope_std": None, # CSV에 없음
                "aspect_mode": float(row['aspect']) if pd.notna(row['aspect']) else None,
                "aspect_std": None, # CSV에 없음
            })
        except Exception as e:
            print(f"❌ [GEE_FEATURES] 데이터 변환 오류 (행 {index}): {e} - 데이터: {row.to_dict()}")
            continue
    
    inserted_count = 0
    for i, gee_data in enumerate(gee_features_list):
        if i % 100 == 0: # 100개마다 진행 상황 출력
            print(f"[GEE_FEATURES] 처리 중: {i}/{len(gee_features_list)} ({(i/len(gee_features_list))*100:.2f}%) 완료")
        if insert_gee_features(gee_data):
            inserted_count += 1
    print(f"✅ [GEE_FEATURES] 총 {inserted_count}건의 GEE_FEATURES 데이터 삽입 완료 (중복 제외).")

def populate_nasa_hourly_weather(df):
    print("\n[NASA_HOURLY_WEATHER] 데이터 삽입 시작...")
    hourly_data_to_insert = []

    for index, row in df.iterrows():
        lat = round(float(row['latitude']), 6)
        lon = round(float(row['longitude']), 6)
        
        try:
            start_dt = datetime.datetime.strptime(row['start_datetime'], '%Y-%m-%d %H:%M:%S')
        except ValueError:
            print(f"⚠️ [NASA_HOURLY_WEATHER] start_datetime 파싱 오류: {row['start_datetime']} (행 {index}) - 건너뜀")
            continue

        for i in range(1, 60):
            current_dt = start_dt - datetime.timedelta(hours=(59 - i))
            
            try:
                hourly_data = {
                    "lat": lat,
                    "lon": lon,
                    "dt": current_dt,
                    "T2M": float(row[f't2m_{i}']) if pd.notna(row.get(f't2m_{i}')) else None,
                    "RH2M": float(row[f'rh2m_{i}']) if pd.notna(row.get(f'rh2m_{i}')) else None,
                    "WS2M": float(row[f'ws2m_{i}']) if pd.notna(row.get(f'ws2m_{i}')) else None,
                    "WD2M": float(row[f'wd2m_{i}']) if pd.notna(row.get(f'wd2m_{i}')) else None,
                    "PRECTOTCORR": float(row[f'prectotcorr_{i}']) if pd.notna(row.get(f'prectotcorr_{i}')) else None,
                    "PS": float(row[f'ps_{i}']) if pd.notna(row.get(f'ps_{i}')) else None,
                    "ALLSKY_SFC_SW_DWN": float(row[f'allsky_sfc_sw_dwn_{i}']) if pd.notna(row.get(f'allsky_sfc_sw_dwn_{i}')) else None,
                    "WS10M": float(row[f'ws10m_{i}']) if pd.notna(row.get(f'ws10m_{i}')) else None,
                    "WD10M": float(row[f'wd10m_{i}']) if pd.notna(row.get(f'wd10m_{i}')) else None,
                }
                hourly_data_to_insert.append(hourly_data)
            except KeyError as ke:
                print(f"⚠️ [NASA_HOURLY_WEATHER] 컬럼 없음 오류: {ke} (행 {index}) - 해당 컬럼 건너뜀")
                hourly_data = {
                    "lat": lat,
                    "lon": lon,
                    "dt": current_dt,
                    "T2M": None, "RH2M": None, "WS2M": None, "WD2M": None,
                    "PRECTOTCORR": None, "PS": None, "ALLSKY_SFC_SW_DWN": None,
                    "WS10M": None, "WD10M": None,
                }
                hourly_data_to_insert.append(hourly_data)
            except Exception as e:
                print(f"❌ [NASA_HOURLY_WEATHER] 데이터 변환 오류 (행 {index}, 시간 {i}): {e} - 데이터: {row.to_dict()}")
                continue
    
    insert_nasa_hourly_weather(hourly_data_to_insert)
    print(f"✅ [NASA_HOURLY_WEATHER] 데이터 삽입 완료.")

def populate_nasa_daily_precip(df):
    print("\n[NASA_DAILY_PRECIP] 데이터 삽입 시작...")
    daily_data_to_insert = []

    for index, row in df.iterrows():
        lat = round(float(row['latitude']), 6)
        lon = round(float(row['longitude']), 6)
        
        try:
            start_dt = datetime.datetime.strptime(row['start_datetime'], '%Y-%m-%d %H:%M:%S')
        except ValueError:
            print(f"⚠️ [NASA_DAILY_PRECIP] start_datetime 파싱 오류: {row['start_datetime']} (행 {index}) - 건너뜀")
            continue

        daily_precip_sums = {}
        for i in range(1, 60):
            current_dt = start_dt - datetime.timedelta(hours=(59 - i))
            current_date = current_dt.date()
            
            try:
                precip = float(row.get(f'prectotcorr_{i}', 0.0))
                if pd.isna(precip): # NaN 값 처리
                    precip = 0.0
            except Exception as e:
                print(f"⚠️ [NASA_DAILY_PRECIP] 강수량 데이터 변환 오류 (행 {index}, 시간 {i}): {e} - 값: {row.get(f'prectotcorr_{i}')}")
                precip = 0.0 # 오류 발생 시 0으로 처리
            
            daily_precip_sums[current_date] = daily_precip_sums.get(current_date, 0.0) + precip
        
        for date, total_precip in daily_precip_sums.items():
            daily_data_to_insert.append({
                "lat": lat,
                "lon": lon,
                "dt": datetime.datetime.combine(date, datetime.time.min), # 날짜만 필요하므로 시간은 00:00:00으로 설정
                "PRECTOTCORR": total_precip
            })
    
    # 중복 제거 (위도, 경도, 날짜 기준)
    unique_daily_data = []
    seen = set()
    for data in daily_data_to_insert:
        key = (data['lat'], data['lon'], data['dt'].date())
        if key not in seen:
            unique_daily_data.append(data)
            seen.add(key)

    insert_nasa_daily_precip(unique_daily_data)
    print(f"✅ [NASA_DAILY_PRECIP] 데이터 삽입 완료.")

if __name__ == "__main__":
    print("데이터베이스 초기화 및 테이블 생성...")
    drop_all_tables()
    create_all_tables()

    print(f"\n{CSV_FILE_PATH} 파일 읽기...")
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"✅ {len(df)}개의 데이터 행을 읽었습니다.")
    except FileNotFoundError:
        print(f"❌ 오류: {CSV_FILE_PATH} 파일을 찾을 수 없습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ CSV 파일 읽기 중 오류 발생: {e}")
        sys.exit(1)

    populate_gee_features(df)
    populate_nasa_hourly_weather(df)
    populate_nasa_daily_precip(df)

    print("\n모든 데이터 삽입 작업 완료.")