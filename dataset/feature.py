import pandas as pd

# 1. 데이터 로드
df = pd.read_excel('./gangwon_fire_data_updated.xlsx')  # 경로에 맞게 조정

# 2. 날짜+시간 컬럼 합치기
df['fire_start_datetime'] = pd.to_datetime(df['fire_start_date'].astype(str) + ' ' + df['start_time'].astype(str))
df['fire_end_datetime'] = pd.to_datetime(df['fire_end_date'].astype(str) + ' ' + df['end_time'].astype(str))

# 3. 기후변수 평균/합산 함수
def average_climate_vars(df, var_prefix):
    cols = [col for col in df.columns if col.startswith(var_prefix)]
    return df[cols].mean(axis=1)

def sum_climate_vars(df, var_prefix):
    cols = [col for col in df.columns if col.startswith(var_prefix)]
    return df[cols].sum(axis=1)

# 4. 기후변수 통합
df['t2m_mean'] = average_climate_vars(df, 't2m_')
df['rh2m_mean'] = average_climate_vars(df, 'rh2m_')
df['ws2m_mean'] = average_climate_vars(df, 'ws2m_')
df['ws10m_mean'] = average_climate_vars(df, 'ws10m_')
df['wd2m_mean'] = average_climate_vars(df, 'wd2m_')
df['wd10m_mean'] = average_climate_vars(df, 'wd10m_')
df['prectotcorr_sum'] = sum_climate_vars(df, 'prectotcorr_')
df['ps_mean'] = average_climate_vars(df, 'ps_')
df['allsky_sfc_sw_dwn_sum'] = sum_climate_vars(df, 'allsky_sfc_sw_dwn_')

# 5. CSV로 저장
output_csv_path = './gangwon_fire_data_climate_aggregated.csv'
df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

print(f"✅ 기후변수 평균/합산 결과 CSV 저장 완료: {output_csv_path}")