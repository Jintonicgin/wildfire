import pandas as pd
import numpy as np

# 1. 기존 데이터 로드
df = pd.read_excel('./gangwon_fire_data_updated.xlsx')

# 2. 분석할 기후 변수 리스트
climate_vars = [
    't2m', 'rh2m', 'ws2m', 'ws10m', 'wd2m', 'wd10m',
    'prectotcorr', 'ps', 'allsky_sfc_sw_dwn'
]

max_segments = 10  # 최대 구간 수

# 3. 피처 생성용 빈 데이터프레임 준비
features = pd.DataFrame(index=df.index)

for var in climate_vars:
    var_cols = [f"{var}_{i}" for i in range(1, max_segments + 1) if f"{var}_{i}" in df.columns]

    if not var_cols:
        continue

    var_data = df[var_cols]

    if var == 'prectotcorr':
        features[f'{var}_sum'] = var_data.sum(axis=1)
        features[f'{var}_max'] = var_data.max(axis=1)
    elif var in ['wd2m', 'wd10m']:
        features[f'{var}_mean'] = var_data.mean(axis=1)
    else:
        features[f'{var}_mean'] = var_data.mean(axis=1)
        features[f'{var}_max'] = var_data.max(axis=1)
        features[f'{var}_min'] = var_data.min(axis=1)
        features[f'{var}_std'] = var_data.std(axis=1)

# 4. 누적강수량, 무강수일수 컬럼 추가
for days in [7]:
    rainfall_col = f"rainfall_cumulative_{days}d"
    dry_days_col = f"dry_days_{days}d"

    features['total_precip_7d'] = df[rainfall_col] if rainfall_col in df.columns else np.nan
    features['no_rain_days_7d'] = df[dry_days_col] if dry_days_col in df.columns else np.nan

# 5. 기존 데이터프레임에 병합 (컬럼 덮어쓰기 주의)
df = pd.concat([df, features], axis=1)

# 6. CSV 저장 (기존 데이터 뒤에 새 피처 포함)
df.to_csv('./gangwon_fire_data_with_features.csv', index=False, encoding='utf-8-sig')

print("✅ 기존 데이터에 피처 병합 후 CSV 저장 완료")