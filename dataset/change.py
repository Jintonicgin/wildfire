import pandas as pd

# CSV 파일 불러오기
df = pd.read_csv('./gangwon_fire_data_merged_no_special.csv', encoding='utf-8-sig')

# 현재 컬럼 확인
print("현재 컬럼 순서:", df.columns.tolist())

df = df.drop(columns=["locgungu", "locmenu"])

# 원하는 컬럼 순서 지정 (예시)
# 예를 들어, 'fire_date'를 맨 앞으로, 'corrected_address' 다음에 위치시키고 싶다면:
new_order = [
    'fire_date',
    'start_time',
    'end_date',
    'end_time',
    'corrected_address',
    'latitude',
    'longitude',
    'fire_area',
    'temp_avg',
    'humidity',
    'wind_speed',
    'wind_dir',
    'precipitation',
]

# 없는 컬럼이 있을 수 있으니 실제 컬럼만 필터링
new_order = [col for col in new_order if col in df.columns]

# 재배치
df = df[new_order]

# 결과 확인
print("변경 후 컬럼 순서:", df.columns.tolist())

# 필요하면 재배치한 데이터프레임 CSV로 저장
df.to_csv('gangwon_fire_data.csv', index=False, encoding='utf-8-sig')
print("✅ 재배치된 CSV 파일 저장 완료: gangwon_fire_data.csv")
