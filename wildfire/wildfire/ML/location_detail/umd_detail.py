import pandas as pd

# 파일 경로 설정
FIRE_CSV_PATH = "./gangwon_fire_fires.csv"  # 산불 데이터 CSV (locgungu, locmenu 포함)
ADMIN_CODE_XLSX_PATH = "./location_detail/hangjungdong_code.xlsx"  # 행정동 코드 엑셀

# 산불 데이터 로드
fire_df = pd.read_csv(FIRE_CSV_PATH, encoding="utf-8-sig")

# 행정동 코드 로드 및 컬럼 공백 제거
admin_df = pd.read_excel(ADMIN_CODE_XLSX_PATH, engine='openpyxl')
admin_df.rename(columns=lambda x: x.strip(), inplace=True)

SPECIAL_CASES = {
    ("영월군", "수주"): "무릉도원면",
    ("양구군", "남"): "국토정중앙면",
    ("홍천군", "동"): "영귀미면",
    ("홍천군", "서"): "서면",
    ("춘천시", "신동"): "신동면",
    ("춘천시", "동"): "동면",
    ("강릉시", "학"): "학동",
    ("강릉시", "강동"): "강동면",
    # 기타 필요 케이스 추가
}

def normalize_gungu(name):
    if not isinstance(name, str) or not name.strip():
        return None
    name = name.strip()

    if not hasattr(normalize_gungu, 'valid_gungu_names'):
        normalize_gungu.valid_gungu_names = set(admin_df['시군구'].unique())

    if name in normalize_gungu.valid_gungu_names:
        return name

    for suffix in ['시', '군', '구']:
        trial_name = name + suffix
        if trial_name in normalize_gungu.valid_gungu_names:
            return trial_name

    return name

def correct_eupmyeon(gungu, eupmyeon_raw):
    if not isinstance(eupmyeon_raw, str) or not eupmyeon_raw.strip():
        eupmyeon_raw = ""
    eupmyeon_raw = eupmyeon_raw.strip()

    if gungu is None:
        return eupmyeon_raw

    # 특별 케이스 우선 처리
    key = (gungu, eupmyeon_raw)
    if key in SPECIAL_CASES:
        return SPECIAL_CASES[key]

    # 완전 일치 검색
    matched = admin_df[(admin_df['시군구'] == gungu) & (admin_df['읍면동'] == eupmyeon_raw)]

    # 포함 검색 (부분 일치, 대소문자 무시)
    if matched.empty:
        matched = admin_df[(admin_df['시군구'] == gungu) & (admin_df['읍면동'].str.contains(eupmyeon_raw, case=False, na=False))]

    # 접미사 붙여서 재검색
    if matched.empty:
        for suffix in ['읍', '면', '동']:
            trial_name = eupmyeon_raw + suffix
            matched = admin_df[(admin_df['시군구'] == gungu) & (admin_df['읍면동'] == trial_name)]
            if not matched.empty:
                break

    if not matched.empty:
        return matched.iloc[0]['읍면동']

    return eupmyeon_raw

def correct_address(row):
    gungu = normalize_gungu(row['locgungu'])
    eupmyeon = correct_eupmyeon(gungu, row['locmenu'])

    if gungu is None:
        return f"강원도 {eupmyeon}".strip()
    return f"강원도 {gungu} {eupmyeon}".strip()

# 시군구가 없는 행 제거 (필요시)
fire_df = fire_df.dropna(subset=['locgungu'])

# 보정 주소 컬럼 추가
fire_df['corrected_address'] = fire_df.apply(correct_address, axis=1)

# 원본 주소 생성 (locgungu + locmenu 합침)
fire_df['original_address'] = ("강원도 " + fire_df['locgungu'].fillna('') + " " + fire_df['locmenu'].fillna('')).str.strip()

# 보정 안 된 행 인덱스 및 내용 출력
unmatched_indices = fire_df.index[
    fire_df['original_address'].str.lower().str.strip() == fire_df['corrected_address'].str.lower().str.strip()
].tolist()

print(f"✅ 보정 완료된 개수: {len(fire_df) - len(unmatched_indices)}")
print(f"⚠️ 보정 안 된 개수: {len(unmatched_indices)}")
print(f"⚠️ 보정 안 된 행 인덱스: {unmatched_indices}")
print(fire_df.loc[unmatched_indices, ['original_address', 'corrected_address']])

# 결과 확인 (상위 30개)
print(fire_df[['original_address', 'corrected_address']].head(30))

# 보정 결과 CSV 저장 (필요 시)
fire_df.to_csv("gangwon_fire_data_corrected.csv", index=False, encoding="utf-8-sig")
print("✅ 보정된 CSV 파일 저장 완료: gangwon_fire_data_corrected.csv")