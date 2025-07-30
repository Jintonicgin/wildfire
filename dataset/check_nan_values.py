import pandas as pd
import sys

CSV_FILE_PATH = '/Users/mmymacymac/Developer/Projects/eclipse/WildFire/dataset/gangwon_fire_data_full_merged.csv'

try:
    df = pd.read_csv(CSV_FILE_PATH)
except FileNotFoundError:
    print(f"❌ 오류: {CSV_FILE_PATH} 파일을 찾을 수 없습니다.")
    sys.exit(1)
except Exception as e:
    print(f"❌ CSV 파일 읽기 중 오류 발생: {e}")
    sys.exit(1)

relevant_cols_gee = ['latitude', 'longitude', 'ndvi_pre_fire_latest', 'elevation', 'slope', 'aspect']
relevant_cols_hourly = []
for i in range(1, 60):
    for param in ['t2m', 'rh2m', 'ws2m', 'wd2m', 'prectotcorr', 'ps', 'allsky_sfc_sw_dwn', 'ws10m', 'wd10m']:
        relevant_cols_hourly.append(f'{param}_{i}')

relevant_cols_daily = [f'prectotcorr_{i}' for i in range(1, 60)]

print("--- NaN counts in GEE related columns ---")
# 실제 df에 존재하는 컬럼만 선택
existing_gee_cols = [col for col in relevant_cols_gee if col in df.columns]
print(df[existing_gee_cols].isnull().sum().to_markdown())

print("\n--- NaN counts in NASA Hourly Weather related columns (first 10 examples) ---")
existing_hourly_cols = [col for col in relevant_cols_hourly if col in df.columns]
print(df[existing_hourly_cols[:10]].isnull().sum().to_markdown())

print("\n--- NaN counts in NASA Daily Precip related columns (first 10 examples) ---")
existing_daily_cols = [col for col in relevant_cols_daily if col in df.columns]
print(df[existing_daily_cols[:10]].isnull().sum().to_markdown())
