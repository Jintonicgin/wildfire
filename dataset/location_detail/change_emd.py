import pandas as pd

# 파일 경로
FIRE_CSV_PATH = "./location_detail/gangwon_fire_data_eupmyeon.csv"
ADMIN_CODE_XLSX_PATH = "./location_detail/hangjungdong_code.xlsx"

# 산불 데이터 로드
fire_df = pd.read_csv(FIRE_CSV_PATH, encoding="utf-8-sig")

# 행정동 코드 로드
admin_df = pd.read_excel(ADMIN_CODE_XLSX_PATH, engine='openpyxl')

# 컬럼명 공백 제거
admin_df.rename(columns=lambda x: x.strip(), inplace=True)

# 시군구에 접미사 붙이기 (없으면 기본 "시" 붙이기)
def normalize_gungu(name):
    if not isinstance(name, str) or not name.strip():
        return ""  # 빈 문자열 처리
    name = name.strip()
    if name.endswith(("시", "군", "구")):
        return name
    # 예외적으로 강원도 내 시군구에 맞게 접미사 붙이기 (필요시 더 확장 가능)
    # 간단히 '시' 붙이기 대신 실제 목록에 따라 처리하는게 좋음
    return name + "시"

def correct_eupmyeon(gungu, eupmyeon_raw):
    if not isinstance(eupmyeon_raw, str) or not eupmyeon_raw.strip():
        eupmyeon_raw = ""
    eupmyeon_raw = eupmyeon_raw.strip()

    matched = admin_df[
        (admin_df['시군구'] == gungu) &
        (admin_df['읍면동'].str.contains(eupmyeon_raw))
    ]

    if not matched.empty:
        return matched.iloc[0]['읍면동']

    # 매칭 실패 시 읍면동 접미사 붙여보기
    for suffix in ['읍', '면', '동']:
        trial_name = eupmyeon_raw + suffix
        matched = admin_df[
            (admin_df['시군구'] == gungu) &
            (admin_df['읍면동'] == trial_name)
        ]
        if not matched.empty:
            return matched.iloc[0]['읍면동']

    # 그래도 못 찾으면 원래값 반환
    return eupmyeon_raw

def correct_address(row):
    gungu = normalize_gungu(row['locgungu'])
    eupmyeon = correct_eupmyeon(gungu, row['locmenu'])
    return f"강원도 {gungu} {eupmyeon}".strip()

# 보정 컬럼 추가
fire_df['corrected_address'] = fire_df.apply(correct_address, axis=1)

# 결과 확인
print(fire_df[['location_name', 'corrected_address']].head(20))

# 필요 시 CSV로 저장
fire_df.to_csv("gangwon_fire_data_corrected.csv", index=False, encoding="utf-8-sig")
print("✅ 보정 완료된 CSV 저장: gangwon_fire_data_corrected.csv")