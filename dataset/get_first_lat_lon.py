import pandas as pd
import sys

CSV_FILE_PATH = '/Users/mmymacymac/Developer/Projects/eclipse/WildFire/dataset/gangwon_fire_data_full_merged.csv'

try:
    df = pd.read_csv(CSV_FILE_PATH)
    if not df.empty:
        lat = df.loc[0, 'latitude']
        lon = df.loc[0, 'longitude']
        print(f"lat: {lat}, lon: {lon}")
    else:
        print("CSV 파일이 비어 있습니다.")
except FileNotFoundError:
    print(f"❌ 오류: {CSV_FILE_PATH} 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"❌ CSV 파일 읽기 중 오류 발생: {e}")
