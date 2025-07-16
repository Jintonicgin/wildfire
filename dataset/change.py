import pandas as pd

df = pd.read_csv('./gangwon_fire_data_merged_no_special.csv', encoding='utf-8-sig')

print("현재 컬럼 순서:", df.columns.tolist())

df = df.drop(columns=["locgungu", "locmenu"])

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

new_order = [col for col in new_order if col in df.columns]

df = df[new_order]

print("변경 후 컬럼 순서:", df.columns.tolist())

df.to_csv('gangwon_fire_data.csv', index=False, encoding='utf-8-sig')
print("✅ 재배치된 CSV 파일 저장 완료: gangwon_fire_data.csv")
