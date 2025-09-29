#!/usr/bin/env python3
"""
데이터 누수 제거 스크립트
미래 정보 및 화재 진행 중 데이터를 제거하여 클린한 훈련 데이터셋을 생성
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime

def identify_leakage_columns(df_columns):
    """데이터 누수가 있는 컬럼들을 식별"""
    leakage_columns = []
    
    # 1. 화재 종료 정보
    end_info_columns = ['endyear', 'endmonth', 'endday', 'endtime']
    leakage_columns.extend([col for col in end_info_columns if col in df_columns])
    
    # 2. 화재 종료 시점 날씨 (_end 접미사)
    end_weather_columns = [col for col in df_columns if col.endswith('_end') and col not in ['dt_end']]
    leakage_columns.extend(end_weather_columns)
    
    # 3. 화재 지속 시간
    duration_columns = ['fire_duration_hours']
    leakage_columns.extend([col for col in duration_columns if col in df_columns])
    
    # 4. 화재 진행 중 날씨 데이터 (3h~171h)
    # 시간별 데이터 패턴: t2m_3h, rh2m_6h, ws2m_9h 등
    time_pattern = re.compile(r'^(t2m|rh2m|ws2m|wd2m|ws10m|wd10m|prectotcorr|ps|allsky_sfc_sw_dwn|dt)_(\d+)h$')
    
    for col in df_columns:
        match = time_pattern.match(col)
        if match:
            hour = int(match.group(2))
            # 3h 이후의 모든 시간별 데이터는 화재 진행 중이므로 제거
            if hour >= 3:
                leakage_columns.append(col)
    
    return list(set(leakage_columns))

def identify_valid_features(df_columns):
    """사용 가능한 (데이터 누수가 없는) 피처들을 식별"""
    all_columns = set(df_columns)
    leakage_columns = set(identify_leakage_columns(df_columns))
    valid_columns = all_columns - leakage_columns
    
    # 화재 시작 시점 데이터 (0h)는 유지
    start_info_columns = ['startyear', 'startmonth', 'startday', 'starttime']
    weather_0h_pattern = re.compile(r'^(t2m|rh2m|ws2m|wd2m|ws10m|wd10m|prectotcorr|ps|allsky_sfc_sw_dwn|dt)_0h$')
    
    # 타겟 변수
    target_column = 'fire_area'
    
    valid_features = []
    
    for col in valid_columns:
        # 시작 정보
        if col in start_info_columns:
            valid_features.append(col)
        # 0시간 날씨 데이터
        elif weather_0h_pattern.match(col):
            valid_features.append(col)
        # 과거 데이터 (_past 접미사)
        elif '_past' in col:
            valid_features.append(col)
        # 지형/지리 정보
        elif col in ['ndvi_before', 'treecover_pre_fire_5x5', 'elevation_mean', 'elevation_std', 
                    'elevation_min', 'elevation_max', 'slope_mean', 'slope_std', 'slope_min', 
                    'slope_max', 'aspect_mode', 'aspect_std', 'aspect_north_ratio', 'aspect_south_ratio']:
            valid_features.append(col)
        # 위치 정보
        elif col in ['start_latitude', 'start_longitude']:
            valid_features.append(col)
        # 시간 정보
        elif col in ['fire_month', 'is_spring', 'is_summer', 'is_autumn', 'is_winter']:
            valid_features.append(col)
        # 과거 강수량 통계
        elif col.startswith('total_precip_') or col.startswith('dry_days_') or col == 'consecutive_dry_days_start':
            valid_features.append(col)
        # FWI 관련 (0시간 시점)
        elif col.endswith('_0h') and any(fwi_var in col for fwi_var in ['ffmc', 'dmc', 'dc', 'isi', 'bui', 'fwi']):
            valid_features.append(col)
        # 0-12시간 통계 (화재 시작 직후 짧은 시간이므로 허용)
        elif any(stat in col for stat in ['_0_12h', '_mean_0_12h']):
            valid_features.append(col)
        # 복합 지수들
        elif col in ['dry_windy_combo', 'hot_dry_combo', 'fuel_combo', 'slope_south_combo',
                    'potential_spread_index', 'terrain_var_effect', 'south_steep_effect',
                    'dry_to_rain_ratio_30d', 'ndvi_stress', 'high_wind_flag', 'low_humidity_flag', 
                    'extreme_hot_flag']:
            valid_features.append(col)
        # 새로 생성된 피처들 (0-12h 범위)
        elif col in ['min_humidity_0_12h_new', 'max_wind_0_12h_new', 'max_temp_0_12h_new', 'wd10m_var_0_12h_new']:
            valid_features.append(col)
        # 타겟 변수
        elif col == target_column:
            valid_features.append(col)
    
    return sorted(valid_features)

def main():
    print("데이터 누수 제거 작업 시작...")
    
    # 원본 데이터 로드
    input_file = '/Users/mmymacymac/Developer/Projects/WildFire_projects/wildfire/wildfire/ML/final_merged_feature_engineered.csv'
    print(f"원본 데이터 로딩: {input_file}")
    
    df = pd.read_csv(input_file)
    print(f"원본 데이터 크기: {df.shape}")
    print(f"원본 컬럼 수: {len(df.columns)}")
    
    # 데이터 누수 컬럼 식별
    leakage_columns = identify_leakage_columns(df.columns)
    print(f"\n제거할 데이터 누수 컬럼 수: {len(leakage_columns)}")
    print("제거할 컬럼들:")
    for i, col in enumerate(leakage_columns, 1):
        print(f"  {i:3d}. {col}")
    
    # 유효한 피처들 식별
    valid_features = identify_valid_features(df.columns)
    print(f"\n유지할 유효한 피처 수: {len(valid_features)}")
    
    # 클린 데이터셋 생성
    clean_df = df[valid_features].copy()
    print(f"\n클린 데이터 크기: {clean_df.shape}")
    
    # 결측치 확인
    missing_counts = clean_df.isnull().sum()
    missing_features = missing_counts[missing_counts > 0]
    if len(missing_features) > 0:
        print(f"\n결측치가 있는 피처들 ({len(missing_features)}개):")
        for feature, count in missing_features.items():
            percentage = (count / len(clean_df)) * 100
            print(f"  {feature}: {count} ({percentage:.1f}%)")
    
    # 클린 데이터셋 저장
    output_file = '/Users/mmymacymac/Developer/Projects/WildFire_projects/wildfire/wildfire/ML/clean_training_dataset.csv'
    clean_df.to_csv(output_file, index=False)
    print(f"\n클린 데이터셋 저장 완료: {output_file}")
    
    # 제거된 컬럼들을 별도 파일로 저장 (참조용)
    removed_info = {
        'original_columns': len(df.columns),
        'removed_columns': len(leakage_columns),
        'remaining_columns': len(valid_features),
        'removed_column_list': leakage_columns
    }
    
    import json
    info_file = '/Users/mmymacymac/Developer/Projects/WildFire_projects/wildfire/wildfire/ML/data_leakage_removal_info.json'
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(removed_info, f, indent=2, ensure_ascii=False)
    print(f"제거 정보 저장 완료: {info_file}")
    
    print("\n=== 데이터 누수 제거 요약 ===")
    print(f"원본 컬럼 수: {len(df.columns)}")
    print(f"제거된 컬럼 수: {len(leakage_columns)}")
    print(f"남은 컬럼 수: {len(valid_features)}")
    print(f"제거 비율: {len(leakage_columns)/len(df.columns)*100:.1f}%")
    print(f"원본 데이터 크기: {df.shape}")
    print(f"클린 데이터 크기: {clean_df.shape}")

if __name__ == "__main__":
    main()