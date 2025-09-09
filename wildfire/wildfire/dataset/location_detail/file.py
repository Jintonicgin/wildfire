import pandas as pd

DEM_CSV_PATH = './gangwon_fire_data_with_dem.csv'
CLEANED_CSV_PATH = './gangwon_fire_data_cleaned.csv'

def split_and_trim_address(addr):
    parts = str(addr).split()
    if len(parts) < 3:
        return "", ""
    gungu = parts[1]
    eupmyeon = parts[2]

    # 읍면동 뒤에 한 글자 제거
    if len(eupmyeon) > 1:
        eupmyeon = eupmyeon[:-1]

    # 시군구 접미사 제거
    if len(gungu) > 1 and gungu[-1] in ['시', '군', '구']:
        gungu = gungu[:-1]

    return gungu, eupmyeon

# 데이터 불러오기
df_dem = pd.read_csv(DEM_CSV_PATH, encoding='utf-8-sig')
df_cleaned = pd.read_csv(CLEANED_CSV_PATH, encoding='utf-8-sig')

# df_dem에서 corrected_address 분리 + 접미사 제거하여 locgungu, locmenu 생성
df_dem[['locgungu', 'locmenu']] = df_dem['corrected_address'].apply(
    lambda x: pd.Series(split_and_trim_address(x))
)

# df_cleaned에는 locgungu, locmenu 그대로 사용 (접미사 제거 안 함)

# 병합 키
merge_keys = ['fire_date', 'locgungu', 'locmenu']

# 병합에 필요한 컬럼만 추출
df_cleaned_subset = df_cleaned[merge_keys + ['start_time', 'end_date', 'end_time']]

# 병합 수행
merged = pd.merge(
    df_dem,
    df_cleaned_subset,
    on=merge_keys,
    how='left'
)

# 결과 확인
print(merged[['corrected_address', 'fire_date', 'start_time', 'end_date', 'end_time']].head(30))

# 필요 시 CSV 저장
merged.to_csv('gangwon_fire_data_merged_no_special.csv', index=False, encoding='utf-8-sig')
print("✅ 병합 및 저장 완료 (특수케이스 없이, 모든 읍면동 뒤 한 글자 제거 적용)")