import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
import math
import datetime
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, classification_report, mean_squared_error
from pathlib import Path
import re

# 경로 설정
sys.path.append(os.path.dirname(__file__))

# 출력 파일 경로
AREA_MODEL_PATH = "area_regressor_model_v4.joblib"
AREA_COLUMNS_PATH = "area_model_columns_v4.json"
AREA_SCALER_PATH = "area_model_scaler_v4.joblib"

SPEED_MODEL_PATH = "speed_classifier_model_v4.joblib"
SPEED_COLUMNS_PATH = "speed_model_columns_v4.json"
SPEED_SCALER_PATH = "speed_scaler_v4.joblib"

DIRECTION_MODEL_PATH = "direction_classifier_model_v4.joblib"
DIRECTION_COLUMNS_PATH = "direction_model_columns_v4.json"
DIRECTION_SCALER_PATH = "direction_scaler_v4.joblib"

def classify_speed(speed: float, thresholds=(0.014, 0.11)) -> int:
    """확산 속도 분류"""
    low, high = thresholds
    if speed <= low: return 0
    if speed <= high: return 1
    return 2

def convert_degree_to_direction(deg: float) -> int:
    """풍향을 8방향으로 분류"""
    if deg is None or pd.isna(deg) or deg == -999: return 0
    return int(math.floor(((float(deg) + 22.5) % 360) / 45))

def remove_temporal_leakage_columns(df):
    """시간적 누수 컬럼 제거"""
    print("🔧 시간적 누수 컬럼 제거 중...")
    
    original_cols = len(df.columns)
    
    # 1. 종료 시점 데이터 제거
    end_cols = [col for col in df.columns if 'end' in col.lower() and col not in ['endyear', 'endmonth', 'endday', 'endtime']]
    
    # 2. 미래 시점 데이터 제거 (15h 이상)
    future_cols = []
    for col in df.columns:
        matches = re.findall(r'_(\d+)h(?!_past)', col)
        for match in matches:
            hours = int(match)
            if hours >= 15:  # 15시간 이후는 미래 데이터
                future_cols.append(col)
                break
    
    # 3. 기타 제거할 컬럼들
    other_leakage_cols = ['endyear', 'endmonth', 'endday', 'endtime']
    
    # 4. 타겟 관련 파생변수 제거
    target_derived_cols = [col for col in df.columns if any(x in col.lower() for x in 
                          ['spread_speed', 'spread_direction', 'potential_spread'])]
    
    # 모든 누수 컬럼 합치기
    leakage_cols = set(end_cols + future_cols + other_leakage_cols + target_derived_cols)
    
    # 안전한 컬럼만 선택
    safe_cols = [col for col in df.columns if col not in leakage_cols]
    
    cleaned_df = df[safe_cols].copy()
    
    print(f"📊 컬럼 정리 결과:")
    print(f"   - 원본: {original_cols}개")
    print(f"   - 제거: {len(leakage_cols)}개 (종료시점: {len(end_cols)}, 미래시점: {len(future_cols)}, 파생변수: {len(target_derived_cols)}, 기타: {len(other_leakage_cols)})")
    print(f"   - 최종: {len(safe_cols)}개")
    
    return cleaned_df

def load_and_clean_existing_data():
    """기존 CSV 데이터 로드 및 정제"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_data_path = os.path.join(script_dir, "final_merged_feature_engineered.csv")
    
    if not os.path.exists(source_data_path):
        print(f"❌ 소스 데이터 파일을 찾을 수 없습니다: {source_data_path}")
        return None
    
    print(f"📁 기존 화재 데이터 로딩: {source_data_path}")
    fire_df = pd.read_csv(source_data_path, low_memory=False)
    fire_df.columns = [col.lower() for col in fire_df.columns]
    
    # 필수 컬럼 확인
    required_cols = ['startyear', 'startmonth', 'startday', 'fire_area', 'fire_duration_hours']
    missing_cols = [col for col in required_cols if col not in fire_df.columns]
    if missing_cols:
        print(f"❌ 필수 컬럼이 누락되었습니다: {missing_cols}")
        return None
    
    # 시간적 누수 제거
    cleaned_df = remove_temporal_leakage_columns(fire_df)
    
    # 유효한 데이터만 선택
    cleaned_df = cleaned_df.dropna(subset=['fire_area', 'fire_duration_hours'])
    cleaned_df = cleaned_df[cleaned_df['fire_area'] > 0]
    cleaned_df = cleaned_df[cleaned_df['fire_duration_hours'] > 0]
    
    print(f"🔥 처리 가능한 화재 데이터: {len(cleaned_df)}건")
    
    return cleaned_df

def create_interaction_features(df):
    """기후, 지형, 식생 간 상관관계 파생피처 생성"""
    print("\n🔗 상관관계 파생피처 생성 중...")
    
    df_enhanced = df.copy()
    created_features = []
    
    # 1. 기후 변수 식별
    climate_vars = {}
    climate_patterns = {
        'temperature': ['t2m_0h'],
        'humidity': ['rh2m_0h'], 
        'wind': ['ws10m_0h', 'ws2m_0h'],
        'pressure': ['ps_0h'],
        'precipitation': ['prectotcorr_0h'],
        'fwi': ['ffmc_0h', 'dmc_0h', 'dc_0h', 'isi_0h', 'bui_0h', 'fwi_0h']
    }
    
    for category, patterns in climate_patterns.items():
        climate_vars[category] = []
        for pattern in patterns:
            matching = [col for col in df.columns if pattern in col.lower()]
            climate_vars[category].extend(matching)
    
    # 2. 지형 변수 식별
    terrain_vars = {}
    terrain_patterns = {
        'elevation': ['elevation_mean', 'elevation_std'],
        'slope': ['slope_mean', 'slope_std'],
        'aspect': ['aspect_mode', 'aspect_std']
    }
    
    for category, patterns in terrain_patterns.items():
        terrain_vars[category] = []
        for pattern in patterns:
            matching = [col for col in df.columns if pattern in col.lower()]
            terrain_vars[category].extend(matching)
    
    # 3. 식생 변수 식별
    vegetation_vars = []
    vegetation_patterns = ['ndvi', 'treecover']
    for pattern in vegetation_patterns:
        matching = [col for col in df.columns if pattern in col.lower()]
        vegetation_vars.extend(matching)
    
    # 4. 상관관계 파생피처 생성
    
    # 4-1. 기후와 지형 상호작용
    if climate_vars['temperature'] and terrain_vars['elevation']:
        temp_col = climate_vars['temperature'][0]
        elev_col = terrain_vars['elevation'][0]
        if temp_col in df.columns and elev_col in df.columns:
            df_enhanced['temp_elevation_interaction'] = df_enhanced[temp_col] * df_enhanced[elev_col]
            created_features.append('temp_elevation_interaction')
    
    if climate_vars['wind'] and terrain_vars['slope']:
        wind_col = climate_vars['wind'][0]
        slope_col = terrain_vars['slope'][0]
        if wind_col in df.columns and slope_col in df.columns:
            df_enhanced['wind_slope_interaction'] = df_enhanced[wind_col] * df_enhanced[slope_col]
            created_features.append('wind_slope_interaction')
    
    # 4-2. 기후와 식생 상호작용
    if climate_vars['humidity'] and vegetation_vars:
        humidity_col = climate_vars['humidity'][0]
        ndvi_col = vegetation_vars[0]
        if humidity_col in df.columns and ndvi_col in df.columns:
            df_enhanced['humidity_vegetation_interaction'] = df_enhanced[humidity_col] * df_enhanced[ndvi_col]
            created_features.append('humidity_vegetation_interaction')
    
    if climate_vars['fwi'] and vegetation_vars:
        fwi_col = climate_vars['fwi'][-1]  # 종합 FWI 사용
        ndvi_col = vegetation_vars[0]
        if fwi_col in df.columns and ndvi_col in df.columns:
            df_enhanced['fwi_vegetation_interaction'] = df_enhanced[fwi_col] * df_enhanced[ndvi_col]
            created_features.append('fwi_vegetation_interaction')
    
    # 4-3. 지형과 식생 상호작용
    if terrain_vars['slope'] and vegetation_vars:
        slope_col = terrain_vars['slope'][0]
        ndvi_col = vegetation_vars[0]
        if slope_col in df.columns and ndvi_col in df.columns:
            df_enhanced['slope_vegetation_interaction'] = df_enhanced[slope_col] * df_enhanced[ndvi_col]
            created_features.append('slope_vegetation_interaction')
    
    # 4-4. 기후 변수들 간 상호작용
    if climate_vars['temperature'] and climate_vars['humidity']:
        temp_col = climate_vars['temperature'][0]
        humidity_col = climate_vars['humidity'][0]
        if temp_col in df.columns and humidity_col in df.columns:
            df_enhanced['temp_humidity_interaction'] = df_enhanced[temp_col] * (1 - df_enhanced[humidity_col]/100)
            created_features.append('temp_humidity_interaction')
    
    if climate_vars['wind'] and climate_vars['humidity']:
        wind_col = climate_vars['wind'][0]
        humidity_col = climate_vars['humidity'][0]
        if wind_col in df.columns and humidity_col in df.columns:
            df_enhanced['wind_humidity_interaction'] = df_enhanced[wind_col] * (1 - df_enhanced[humidity_col]/100)
            created_features.append('wind_humidity_interaction')
    
    # 4-5. 지형 변수들 간 상호작용
    if terrain_vars['elevation'] and terrain_vars['slope']:
        elev_col = terrain_vars['elevation'][0]
        slope_col = terrain_vars['slope'][0]
        if elev_col in df.columns and slope_col in df.columns:
            df_enhanced['elevation_slope_interaction'] = df_enhanced[elev_col] * df_enhanced[slope_col]
            created_features.append('elevation_slope_interaction')
    
    # 4-6. 복합 화재위험지수 생성
    if (climate_vars['temperature'] and climate_vars['humidity'] and 
        climate_vars['wind'] and vegetation_vars):
        temp_col = climate_vars['temperature'][0]
        humidity_col = climate_vars['humidity'][0]
        wind_col = climate_vars['wind'][0]
        ndvi_col = vegetation_vars[0]
        
        if all(col in df.columns for col in [temp_col, humidity_col, wind_col, ndvi_col]):
            # 복합 화재위험지수 = (온도 * 풍속) / (습도 * 식생지수)
            df_enhanced['composite_fire_risk'] = (
                (df_enhanced[temp_col] * df_enhanced[wind_col]) / 
                (df_enhanced[humidity_col] * (df_enhanced[ndvi_col] + 0.1))  # 0으로 나누기 방지
            )
            created_features.append('composite_fire_risk')
    
    # 4-7. 건조도 지수 생성
    dry_day_cols = [col for col in df.columns if 'dry_days' in col.lower()]
    if dry_day_cols and climate_vars['fwi']:
        dry_col = dry_day_cols[0]  # 첫 번째 건조일 변수
        fwi_col = climate_vars['fwi'][-1]
        if dry_col in df.columns and fwi_col in df.columns:
            df_enhanced['dryness_fire_index'] = df_enhanced[dry_col] * df_enhanced[fwi_col]
            created_features.append('dryness_fire_index')
    
    # 4-8. 계절별 상관관계 피처 생성
    if 'startmonth' in df.columns:
        # 계절 구분 (3-5월: 봄, 6-8월: 여름, 9-11월: 가을, 12-2월: 겨울)
        df_enhanced['season'] = df_enhanced['startmonth'].apply(
            lambda x: 'spring' if 3 <= x <= 5 else 'summer' if 6 <= x <= 8 else 'autumn' if 9 <= x <= 11 else 'winter'
        )
        
        # 계절별 더미 변수 생성
        for season in ['spring', 'summer', 'autumn', 'winter']:
            df_enhanced[f'is_{season}'] = (df_enhanced['season'] == season).astype(int)
            created_features.append(f'is_{season}')
        
        # 계절-기후 상호작용
        if climate_vars['temperature']:
            temp_col = climate_vars['temperature'][0]
            if temp_col in df.columns:
                # 여름 고온 위험도
                df_enhanced['summer_temp_risk'] = df_enhanced['is_summer'] * df_enhanced[temp_col]
                created_features.append('summer_temp_risk')
                
                # 봄/가을 온도 변화율
                df_enhanced['spring_autumn_temp'] = (df_enhanced['is_spring'] + df_enhanced['is_autumn']) * df_enhanced[temp_col]
                created_features.append('spring_autumn_temp')
        
        if climate_vars['humidity']:
            humidity_col = climate_vars['humidity'][0]
            if humidity_col in df.columns:
                # 겨울 저습도 위험
                df_enhanced['winter_dry_risk'] = df_enhanced['is_winter'] * (100 - df_enhanced[humidity_col])
                created_features.append('winter_dry_risk')
                
                # 가을 건조 위험
                df_enhanced['autumn_dry_risk'] = df_enhanced['is_autumn'] * (100 - df_enhanced[humidity_col])
                created_features.append('autumn_dry_risk')
        
        if climate_vars['wind']:
            wind_col = climate_vars['wind'][0]
            if wind_col in df.columns:
                # 봄 강풍 위험
                df_enhanced['spring_wind_risk'] = df_enhanced['is_spring'] * df_enhanced[wind_col]
                created_features.append('spring_wind_risk')
                
                # 겨울 강풍 위험  
                df_enhanced['winter_wind_risk'] = df_enhanced['is_winter'] * df_enhanced[wind_col]
                created_features.append('winter_wind_risk')
        
        # 계절-식생 상호작용
        if vegetation_vars:
            ndvi_col = vegetation_vars[0]
            if ndvi_col in df.columns:
                # 가을 낙엽기 위험 (낮은 NDVI + 가을)
                df_enhanced['autumn_vegetation_risk'] = df_enhanced['is_autumn'] * (1 - df_enhanced[ndvi_col])
                created_features.append('autumn_vegetation_risk')
                
                # 봄 신록기 보호효과
                df_enhanced['spring_vegetation_protection'] = df_enhanced['is_spring'] * df_enhanced[ndvi_col]
                created_features.append('spring_vegetation_protection')
        
        # 계절-FWI 상호작용
        if climate_vars['fwi']:
            fwi_col = climate_vars['fwi'][-1]
            if fwi_col in df.columns:
                # 가을 화재위험도 (가을은 화재 다발 계절)
                df_enhanced['autumn_fire_risk'] = df_enhanced['is_autumn'] * df_enhanced[fwi_col]
                created_features.append('autumn_fire_risk')
                
                # 봄 화재위험도 (건조한 봄철)
                df_enhanced['spring_fire_risk'] = df_enhanced['is_spring'] * df_enhanced[fwi_col]
                created_features.append('spring_fire_risk')
        
        # 계절-건조일수 상호작용
        if dry_day_cols:
            dry_col = dry_day_cols[0]
            if dry_col in df.columns:
                # 봄 가뭄 위험
                df_enhanced['spring_drought_risk'] = df_enhanced['is_spring'] * df_enhanced[dry_col]
                created_features.append('spring_drought_risk')
                
                # 가을 가뭄 위험
                df_enhanced['autumn_drought_risk'] = df_enhanced['is_autumn'] * df_enhanced[dry_col]
                created_features.append('autumn_drought_risk')
        
        # season 문자열 컬럼 제거 (수치형만 유지)
        if 'season' in df_enhanced.columns:
            df_enhanced = df_enhanced.drop('season', axis=1)
    
    # 4-9. 고급 피처 엔지니어링 추가
    
    # 통계적 변환 피처
    if climate_vars['temperature'] and climate_vars['humidity']:
        temp_col = climate_vars['temperature'][0]
        humidity_col = climate_vars['humidity'][0]
        if temp_col in df.columns and humidity_col in df.columns:
            # 체감온도 지수 (열지수 근사)
            df_enhanced['heat_index'] = df_enhanced[temp_col] + 0.5 * (df_enhanced[temp_col] - 10) * (df_enhanced[humidity_col] / 100)
            created_features.append('heat_index')
            
            # 증발산 잠재력 (간단한 근사)
            df_enhanced['evapotranspiration_potential'] = np.maximum(0, df_enhanced[temp_col] * (1 - df_enhanced[humidity_col] / 100))
            created_features.append('evapotranspiration_potential')
    
    # 다항식 피처 (중요 변수들)
    if climate_vars['fwi']:
        fwi_col = climate_vars['fwi'][-1]
        if fwi_col in df.columns:
            # FWI 제곱 (비선형 관계 포착)
            df_enhanced['fwi_squared'] = df_enhanced[fwi_col] ** 2
            created_features.append('fwi_squared')
            
            # FWI 로그 변환 (분포 정규화)
            df_enhanced['fwi_log'] = np.log1p(df_enhanced[fwi_col])
            created_features.append('fwi_log')
    
    # 비 관련 누적 피처
    if 'prectotcorr_0h' in df.columns:
        # 최근 강수 부족 지수
        recent_precip_cols = [col for col in df.columns if 'prec' in col.lower() and any(x in col for x in ['0h', '3h', '6h', '12h'])]
        if recent_precip_cols:
            recent_precip_sum = df_enhanced[recent_precip_cols].sum(axis=1)
            df_enhanced['recent_precipitation_deficit'] = np.maximum(0, 5 - recent_precip_sum)  # 5mm를 기준으로 부족분 계산
            created_features.append('recent_precipitation_deficit')
    
    # 지형 복잡도 지수
    if terrain_vars.get('elevation') and terrain_vars.get('slope'):
        elev_std_col = [col for col in terrain_vars['elevation'] if 'std' in col]
        slope_std_col = [col for col in terrain_vars['slope'] if 'std' in col]
        if elev_std_col and slope_std_col:
            elev_std = elev_std_col[0]
            slope_std = slope_std_col[0]
            if elev_std in df.columns and slope_std in df.columns:
                # 지형 복잡도 (고도 표준편차 + 경사 표준편차)
                df_enhanced['terrain_complexity'] = df_enhanced[elev_std] + df_enhanced[slope_std]
                created_features.append('terrain_complexity')
    
    # 극한 기상 플래그
    if climate_vars['temperature'] and climate_vars.get('wind_speed'):
        temp_col = climate_vars['temperature'][0]
        wind_col = climate_vars['wind_speed'][0]
        if temp_col in df.columns and wind_col in df.columns:
            # 고온 강풍 위험 플래그
            temp_threshold = df_enhanced[temp_col].quantile(0.8)
            wind_threshold = df_enhanced[wind_col].quantile(0.8)
            df_enhanced['extreme_fire_weather'] = ((df_enhanced[temp_col] > temp_threshold) & 
                                                 (df_enhanced[wind_col] > wind_threshold)).astype(int)
            created_features.append('extreme_fire_weather')
    
    # 순환 인코딩 (시간적 패턴)
    if 'month' in df.columns:
        # 월의 순환적 특성 (sin/cos 변환)
        df_enhanced['month_sin'] = np.sin(2 * np.pi * df_enhanced['month'] / 12)
        df_enhanced['month_cos'] = np.cos(2 * np.pi * df_enhanced['month'] / 12)
        created_features.extend(['month_sin', 'month_cos'])
    
    if 'day' in df.columns:
        # 일의 순환적 특성 (sin/cos 변환)
        df_enhanced['day_sin'] = np.sin(2 * np.pi * df_enhanced['day'] / 31)
        df_enhanced['day_cos'] = np.cos(2 * np.pi * df_enhanced['day'] / 31)
        created_features.extend(['day_sin', 'day_cos'])
    
    # 극값 지표 피처
    if climate_vars['temperature']:
        temp_col = climate_vars['temperature'][0]
        if temp_col in df.columns:
            # 극고온 지표 (임계값 30도)
            df_enhanced['extreme_heat_flag'] = (df_enhanced[temp_col] > 30).astype(int)
            created_features.append('extreme_heat_flag')
    
    if climate_vars['wind']:
        wind_col = climate_vars['wind'][0]
        if wind_col in df.columns:
            # 강풍 지표 (임계값 10m/s)
            df_enhanced['strong_wind_flag'] = (df_enhanced[wind_col] > 10).astype(int)
            created_features.append('strong_wind_flag')
    
    # 시간적 트렌드 피처
    if 'startyear' in df.columns and 'startmonth' in df.columns:
        # 년도별 트렌드 (기후변화)
        base_year = df_enhanced['startyear'].min()
        df_enhanced['years_since_base'] = df_enhanced['startyear'] - base_year
        created_features.append('years_since_base')
        
        # 월별 순환 피처 (사인, 코사인 변환)
        df_enhanced['month_sin'] = np.sin(2 * np.pi * df_enhanced['startmonth'] / 12)
        df_enhanced['month_cos'] = np.cos(2 * np.pi * df_enhanced['startmonth'] / 12)
        created_features.extend(['month_sin', 'month_cos'])
        
        # 일별 순환 피처
        if 'startday' in df.columns:
            df_enhanced['day_sin'] = np.sin(2 * np.pi * df_enhanced['startday'] / 31)
            df_enhanced['day_cos'] = np.cos(2 * np.pi * df_enhanced['startday'] / 31)
            created_features.extend(['day_sin', 'day_cos'])
    
    print(f"✅ 생성된 상관관계 파생피처: {len(created_features)}개")
    for feature in created_features:
        print(f"   - {feature}")
    
    # 무한값과 극한값 처리
    print("🔧 무한값 및 극한값 처리 중...")
    
    # 무한값을 NaN으로 변경
    df_enhanced = df_enhanced.replace([np.inf, -np.inf], np.nan)
    
    # 극한값 클리핑 (각 컬럼의 99.9%ile 및 0.1%ile로 제한)
    for col in created_features:
        if col in df_enhanced.columns:
            q99 = df_enhanced[col].quantile(0.999)
            q01 = df_enhanced[col].quantile(0.001)
            if pd.notna(q99) and pd.notna(q01):
                df_enhanced[col] = df_enhanced[col].clip(q01, q99)
    
    print(f"   - 무한값 제거 완료")
    print(f"   - 극한값 클리핑 완료")
    
    return df_enhanced, created_features

def prepare_data_for_models(cleaned_df):
    """모델별 데이터 준비 (누수 제거된 데이터 + 상관관계 파생피처 사용)"""
    print("\n📊 모델별 데이터 준비 중...")
    
    # 상관관계 파생피처 생성
    df_enhanced, interaction_features = create_interaction_features(cleaned_df)
    df_clean = df_enhanced.copy()
    
    # 타겟 변수 생성
    df_clean['spread_speed'] = df_clean['fire_area'] / df_clean['fire_duration_hours']
    df_clean['spread_speed_class'] = df_clean['spread_speed'].apply(classify_speed)
    
    # 풍향 데이터 확인 및 처리
    wind_direction_col = None
    for col in ['wd10m_0h', 'wd10m_0h_original']:
        if col in df_clean.columns:
            wind_direction_col = col
            break
    
    if wind_direction_col:
        df_clean['spread_direction_class'] = df_clean[wind_direction_col].apply(convert_degree_to_direction)
    else:
        print("⚠️ 풍향 데이터를 찾을 수 없어 방향 예측을 위해 0으로 설정합니다.")
        df_clean['spread_direction_class'] = 0
    
    # 수치형 컬럼만 선택
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    # 제외할 컬럼들 (타겟 및 파생변수)
    cols_to_exclude = [
        'fire_area', 'fire_duration_hours', 'spread_speed', 
        'spread_speed_class', 'spread_direction_class'
    ]
    
    # 풍향 원본 컬럼도 제외
    if wind_direction_col:
        cols_to_exclude.append(wind_direction_col)
    
    # 피처 컬럼들
    feature_cols = [col for col in numeric_cols if col not in cols_to_exclude]
    
    # 추가 정제: 위치 정보는 유지되나 개인정보성 데이터 확인
    sensitive_cols = [col for col in feature_cols if any(x in col.lower() for x in ['id', 'name', 'address'])]
    if sensitive_cols:
        print(f"⚠️ 민감정보 가능성 컬럼 제외: {sensitive_cols}")
        feature_cols = [col for col in feature_cols if col not in sensitive_cols]
    
    # NaN 값 처리
    X_all = df_clean[feature_cols].fillna(0)
    
    print(f"📋 기본 피처 수: {len(feature_cols) - len(interaction_features)}")
    print(f"📋 상관관계 파생피처 수: {len(interaction_features)}")
    print(f"📋 최종 피처 수: {len(feature_cols)}")
    print(f"📋 데이터 수: {len(df_clean)}")
    print(f"📋 데이터 대 피처 비율: {len(df_clean)/len(feature_cols):.2f}")
    
    if len(df_clean) < len(feature_cols) * 5:
        print("⚠️ 경고: 피처 수가 데이터 수에 비해 너무 많아 과적합 위험이 있습니다.")
    
    return df_clean, X_all, feature_cols

def select_important_features(X_all, feature_cols, max_features=30):
    """중요 피처 선택 (성능과 과적합 방지 균형)"""
    print(f"\n🔍 중요 피처 선택 중... (최대 {max_features}개)")
    
    # 1. 핵심 피처 선택 (도메인 지식 기반, 조금 더 포용적)
    essential_features = []
    
    # 1-1. 시간 정보 (연도도 포함 - 기후변화 트렌드)
    temporal_features = [f for f in feature_cols if any(x in f.lower() for x in ['startyear', 'startmonth', 'startday'])]
    essential_features.extend(temporal_features[:3])  # 연, 월, 일
    
    # 1-2. 핵심 기상 정보 (FWI 계열 확장)
    fire_weather = [f for f in feature_cols if any(x in f.lower() for x in 
                   ['fwi_0h', 'ffmc_0h', 'dmc_0h', 'dc_0h', 'isi_0h', 'bui_0h'])]
    essential_features.extend(fire_weather[:6])  # FWI 관련 6개
    
    current_weather = [f for f in feature_cols if any(x in f.lower() for x in 
                      ['t2m_0h', 'rh2m_0h', 'ws10m_0h', 'ps_0h'])]
    essential_features.extend(current_weather[:4])  # 기본 기상 4개
    
    # 1-3. 지형 정보 (확장)
    terrain = [f for f in feature_cols if any(x in f.lower() for x in 
              ['elevation_mean', 'elevation_std', 'slope_mean', 'slope_std', 'aspect_mode'])]
    essential_features.extend(terrain[:5])  # 지형 5개
    
    # 1-4. 식생 정보 (확장)
    vegetation = [f for f in feature_cols if any(x in f.lower() for x in ['ndvi', 'treecover'])]
    essential_features.extend(vegetation[:2])  # 식생 2개
    
    # 1-5. 과거 건조도 (확장)
    dry_history = [f for f in feature_cols if any(x in f.lower() for x in 
                  ['dry_days_7d', 'dry_days_14d', 'dry_days_30d'])]
    essential_features.extend(dry_history[:3])  # 건조도 3개
    
    # 1-6. 계절 정보 (모든 계절)
    seasonal = [f for f in feature_cols if any(x in f.lower() for x in 
               ['is_spring', 'is_summer', 'is_autumn', 'is_winter'])]
    essential_features.extend(seasonal[:4])  # 4계절
    
    # 1-7. 핵심 상호작용 피처 (중요한 것들)
    key_interactions = [f for f in feature_cols if any(x in f.lower() for x in 
                       ['composite_fire_risk', 'autumn_fire_risk', 'spring_fire_risk', 
                        'dryness_fire_index', 'temp_elevation_interaction'])]
    essential_features.extend(key_interactions[:5])  # 상호작용 5개
    
    # 중복 제거 및 존재하는 피처만 선택
    essential_features = [f for f in essential_features if f in feature_cols]
    essential_features = list(set(essential_features))
    
    print(f"   - 필수 피처: {len(essential_features)}개")
    
    # 2. 통계적 중요도 기반 추가 (적당히 제한적)
    remaining_features = [f for f in feature_cols if f not in essential_features]
    
    # 분산이 높은 피처 추가
    if len(essential_features) < max_features and remaining_features:
        from sklearn.feature_selection import VarianceThreshold
        
        # 분산이 너무 낮은 피처 제거 (기준 완화)
        selector = VarianceThreshold(threshold=0.001)  # 기준 완화
        X_remaining = X_all[remaining_features].fillna(0)
        
        try:
            X_filtered = selector.fit_transform(X_remaining)
            selected_remaining = np.array(remaining_features)[selector.get_support()]
            
            # 분산 기준으로 추가 선택
            variances = X_all[selected_remaining].var().sort_values(ascending=False)
            additional_count = min(max_features - len(essential_features), 10)  # 최대 10개 추가
            additional_features = variances.head(additional_count).index.tolist()
            
            essential_features.extend(additional_features)
            print(f"   - 추가 피처: {len(additional_features)}개")
            
        except Exception as e:
            print(f"   - 추가 피처 선택 실패: {e}")
    
    # 최종 피처 선택 (강제로 max_features 이하로 제한)
    selected_features = essential_features[:max_features]
    
    print(f"📊 최종 선택된 피처: {len(selected_features)}개")
    print("선택된 피처 목록:")
    for i, feature in enumerate(selected_features, 1):
        print(f"   {i:2d}. {feature}")
    
    return selected_features

def train_area_model(df_clean, X_all, feature_cols):
    """피해면적 예측 모델 학습 (과적합 방지 강화)"""
    print("\n🎯 피해면적 예측 모델 학습 시작...")
    
    # 데이터 수에 맞는 적정 피처 선택 (성능과 과적합 방지 균형)
    data_count = len(df_clean)
    max_features_area = min(40, data_count // 15)  # 보다 현실적으로 (데이터/15)
    max_features_area = max(25, max_features_area)  # 최소 25개는 보장
    
    print(f"📊 Area 모델 - 데이터 수: {data_count}, 최대 피처 수: {max_features_area}")
    
    # 피처 선택
    area_features = select_important_features(X_all, feature_cols, max_features_area)
    X_area_selected = X_all[area_features].fillna(0)
    
    # 타겟 준비 (로그 변환)
    y_area = np.log1p(df_clean['fire_area'])
    
    # 이상치 제거 (90% 분위수로 더 강화)
    area_90th = df_clean['fire_area'].quantile(0.90)
    mask = df_clean['fire_area'] <= area_90th
    X_area = X_area_selected[mask]
    y_area = y_area[mask]
    
    print(f"📊 이상치 제거 후 데이터 수: {len(X_area)}")
    
    # Train-Test Split으로 검증
    from sklearn.model_selection import train_test_split
    # 데이터 정제 (무한값, NaN, 극한값 처리)
    print("🔧 Area 모델 데이터 정제 중...")
    
    # 무한값 제거
    X_area = X_area.replace([np.inf, -np.inf], np.nan)
    
    # NaN 채우기 (0으로)
    X_area = X_area.fillna(0)
    
    # 극한값 클리핑 (각 컬럼별로 더 보수적인 99.5%ile 기준)
    for col in X_area.columns:
        if X_area[col].dtype in ['float64', 'int64']:
            q99 = X_area[col].quantile(0.995)
            q01 = X_area[col].quantile(0.005)
            if pd.notna(q99) and pd.notna(q01) and q99 != q01:
                # 더 안전한 클리핑
                safe_max = min(q99, 1e6)  # 최대 1백만으로 제한
                safe_min = max(q01, -1e6)  # 최소 -1백만으로 제한
                X_area[col] = X_area[col].clip(safe_min, safe_max)
    
    # NaN을 중위수로 채우기
    for col in X_area.columns:
        if X_area[col].isna().sum() > 0:
            median_val = X_area[col].median()
            if pd.notna(median_val):
                X_area[col] = X_area[col].fillna(median_val)
            else:
                X_area[col] = X_area[col].fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_area, y_area, test_size=0.2, random_state=42
    )
    
    # 스케일링
    scaler_area = RobustScaler()
    X_train_scaled = scaler_area.fit_transform(X_train)
    X_test_scaled = scaler_area.transform(X_test)
    
    # 스케일링 후에도 무한값 체크
    if not np.all(np.isfinite(X_train_scaled)) or not np.all(np.isfinite(X_test_scaled)):
        print("⚠️ 스케일링 후 무한값 발견, 추가 정제...")
        # 더 안전한 값으로 치환
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0, posinf=10, neginf=-10)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0, posinf=10, neginf=-10)
        
        # 추가 안전성 체크
        X_train_scaled = np.clip(X_train_scaled, -100, 100)
        X_test_scaled = np.clip(X_test_scaled, -100, 100)
    
    # 성능과 과적합 방지의 균형을 맞춘 하이퍼파라미터
    param_grid = {
        'n_estimators': [100, 200],         # 트리 수 적당히 증가
        'max_depth': [5, 8, 10],           # 깊이 완화
        'min_samples_leaf': [2, 5, 8],     # 리프 노드 완화
        'min_samples_split': [5, 10],      # 분할 기준 완화
        'max_features': ['sqrt', 0.5],     # 피처 선택 옵션 추가
        'max_samples': [0.8, 0.9]          # 배깅 비율 완화
    }
    
    cv_folds = 5
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # 앙상블 모델 구성: RandomForest + GradientBoosting
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import VotingRegressor
    
    # 1. RandomForest 모델
    rf_area = RandomForestRegressor(
        random_state=42, 
        n_jobs=-1,
        bootstrap=True,  # 배깅 활성화
        oob_score=True   # OOB 점수 계산
    )
    
    # 2. GradientBoosting 모델
    gb_area = GradientBoostingRegressor(
        random_state=42,
        learning_rate=0.08,
        max_depth=6,
        min_samples_split=15,
        min_samples_leaf=8,
        subsample=0.8
    )
    
    # 3. 투표 앙상블 모델
    ensemble_area = VotingRegressor([
        ('rf', rf_area),
        ('gb', gb_area)
    ], weights=[0.6, 0.4])
    
    print("🔧 하이퍼파라미터 튜닝 중 (앙상블 정규화 강화)...")
    
    # RandomForest 하이퍼파라미터 튜닝
    rf_grid_search = GridSearchCV(
        estimator=rf_area, 
        param_grid=param_grid, 
        cv=kf,
        scoring='neg_root_mean_squared_error', 
        verbose=1, 
        n_jobs=-1
    )
    rf_grid_search.fit(X_train_scaled, y_train)
    
    # RandomForest 최적 파라미터 추출
    best_rf_params = rf_grid_search.best_params_
    print(f"RF 최적 파라미터: {best_rf_params}")
    
    # 최적화된 앙상블 모델 생성
    optimized_rf = RandomForestRegressor(
        **best_rf_params,
        random_state=42, 
        n_jobs=-1,
        bootstrap=True,
        oob_score=True
    )
    
    optimized_gb = GradientBoostingRegressor(
        random_state=42,
        learning_rate=0.08,
        max_depth=min(6, best_rf_params.get('max_depth', 6)),
        min_samples_split=max(15, best_rf_params.get('min_samples_split', 15)),
        min_samples_leaf=max(8, best_rf_params.get('min_samples_leaf', 8)),
        subsample=0.8,
        n_estimators=150
    )
    
    # 최종 앙상블 모델
    best_model = VotingRegressor([
        ('rf', optimized_rf),
        ('gb', optimized_gb)
    ], weights=[0.6, 0.4])
    
    # 앙상블 모델 학습
    best_model.fit(X_train_scaled, y_train)
    
    # 훈련 및 테스트 성능 비교
    y_train_pred = best_model.predict(X_train_scaled)
    y_test_pred = best_model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # 교차 검증 점수
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='r2')
    
    print(f"✅ Area 앙상블 모델 성능:")
    print(f"   - 사용된 피처 수: {len(area_features)}")
    print(f"   - RF 최적 파라미터: {best_rf_params}")
    print(f"   - 훈련 R²: {train_r2:.4f}, 테스트 R²: {test_r2:.4f}")
    print(f"   - 훈련 RMSE: {train_rmse:.4f}, 테스트 RMSE: {test_rmse:.4f}")
    print(f"   - 교차검증 R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # OOB 점수는 개별 RF 모델을 학습해야 얻을 수 있음
    try:
        optimized_rf.fit(X_train_scaled, y_train)
        print(f"   - RF OOB Score: {optimized_rf.oob_score_:.4f}")
    except:
        print(f"   - RF OOB Score: 계산 불가")
    
    # 과적합 경고
    if train_r2 - test_r2 > 0.1:
        print("⚠️ 경고: 과적합 의심 (훈련-테스트 성능 차이 > 0.1)")
    
    if cv_scores.std() > 0.1:
        print("⚠️ 경고: 모델 불안정 (교차검증 표준편차 > 0.1)")
    
    return best_model, scaler_area, area_features

def train_speed_direction_models(df_clean, X_all, feature_cols):
    """속도/방향 분류 모델 학습 (과적합 방지 강화)"""
    print("\n🎯 속도/방향 분류 모델 학습 시작...")
    
    # 데이터 수에 맞는 적정 피처 수 결정 (성능 고려)
    data_count = len(df_clean)
    max_features = min(25, data_count // 25)  # 보다 현실적으로 (데이터/25)
    max_features = max(15, max_features)  # 최소 15개는 보장
    
    print(f"📊 데이터 수: {data_count}, 최대 피처 수: {max_features}")
    
    # 중요 피처 선택
    selected_features = select_important_features(X_all, feature_cols, max_features)
    X_core = X_all[selected_features].fillna(0)
    
    # 속도 모델 학습
    print("\n🚀 속도 분류 모델 학습...")
    y_speed = df_clean['spread_speed_class'].astype(int)
    
    # Train-Test Split
    from sklearn.model_selection import train_test_split
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_core, y_speed, test_size=0.2, random_state=42, stratify=y_speed
    )
    
    scaler_speed = RobustScaler()
    X_train_s_scaled = scaler_speed.fit_transform(X_train_s)
    X_test_s_scaled = scaler_speed.transform(X_test_s)
    
    # 앙상블 분류 모델 구성
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import VotingClassifier
    
    # RandomForest 분류기
    rf_speed = RandomForestClassifier(
        random_state=42, 
        n_jobs=-1,
        bootstrap=True,
        oob_score=True
    )
    
    # GradientBoosting 분류기  
    gb_speed = GradientBoostingClassifier(
        random_state=42,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        n_estimators=150
    )
    
    # 성능과 일반화의 균형을 맞춘 파라미터
    param_grid_clf = {
        'n_estimators': [100, 150],         # 트리 수 증가
        'max_depth': [5, 8, 12],           # 깊이 완화
        'min_samples_leaf': [2, 5],        # 리프 노드 완화
        'min_samples_split': [5, 10],      # 분할 기준 완화
        'max_features': ['sqrt', 0.5],     # 피처 선택 옵션
        'max_samples': [0.8, 0.9]          # 배깅 비율 완화
    }
    
    cv_folds = 5
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # RandomForest 하이퍼파라미터 튜닝
    rf_speed_tune = RandomForestClassifier(
        random_state=42, 
        class_weight='balanced',
        bootstrap=True,
        oob_score=True
    )
    
    grid_speed = GridSearchCV(
        rf_speed_tune, param_grid_clf, cv=kf, scoring='f1_weighted', n_jobs=-1, verbose=1
    )
    grid_speed.fit(X_train_s_scaled, y_train_s)
    
    # 최적화된 앙상블 모델 구성
    best_params_speed = grid_speed.best_params_
    print(f"Speed RF 최적 파라미터: {best_params_speed}")
    
    optimized_rf_speed = RandomForestClassifier(
        **best_params_speed,
        random_state=42,
        class_weight='balanced',
        bootstrap=True,
        oob_score=True
    )
    
    optimized_gb_speed = GradientBoostingClassifier(
        random_state=42,
        learning_rate=0.1,
        max_depth=min(5, best_params_speed.get('max_depth', 5)),
        min_samples_split=max(20, best_params_speed.get('min_samples_split', 20)),
        min_samples_leaf=max(10, best_params_speed.get('min_samples_leaf', 10)),
        subsample=0.8,
        n_estimators=150
    )
    
    # 속도 앙상블 모델
    best_speed_model = VotingClassifier([
        ('rf', optimized_rf_speed),
        ('gb', optimized_gb_speed)
    ], voting='soft', weights=[0.6, 0.4])
    
    best_speed_model.fit(X_train_s_scaled, y_train_s)
    
    # 훈련/테스트 성능 비교
    y_train_pred_s = best_speed_model.predict(X_train_s_scaled)
    y_test_pred_s = best_speed_model.predict(X_test_s_scaled)
    
    train_acc_s = (y_train_pred_s == y_train_s).mean()
    test_acc_s = (y_test_pred_s == y_test_s).mean()
    
    # 교차검증
    from sklearn.model_selection import cross_val_score
    cv_scores_s = cross_val_score(best_speed_model, X_train_s_scaled, y_train_s, cv=5)
    
    print(f"✅ Speed 앙상블 모델 성능:")
    print(f"   - RF 최적 파라미터: {best_params_speed}")
    print(f"   - 훈련 정확도: {train_acc_s:.4f}, 테스트 정확도: {test_acc_s:.4f}")
    print(f"   - 교차검증 정확도: {cv_scores_s.mean():.4f} ± {cv_scores_s.std():.4f}")
    
    # Speed RF OOB 점수
    try:
        optimized_rf_speed.fit(X_train_s_scaled, y_train_s)
        print(f"   - RF OOB Score: {optimized_rf_speed.oob_score_:.4f}")
    except:
        print(f"   - RF OOB Score: 계산 불가")
    print(classification_report(y_test_s, y_test_pred_s, zero_division=0))
    
    # 과적합 경고
    if train_acc_s - test_acc_s > 0.1:
        print("⚠️ 경고: Speed 모델 과적합 의심 (훈련-테스트 차이 > 0.1)")
    
    # 방향 모델 학습 (풍향 관련 피처 제외)
    print("\n🧭 방향 분류 모델 학습...")
    direction_features = [f for f in selected_features if not any(x in f.lower() for x in ['wd10m', 'wd2m'])]
    X_dir = X_all[direction_features].fillna(0)
    y_direction = df_clean['spread_direction_class'].astype(int)
    
    # 방향 데이터 불균형 확인
    direction_counts = pd.Series(y_direction).value_counts()
    print(f"방향 클래스 분포: {direction_counts.to_dict()}")
    
    # 소수 클래스 병합 (샘플이 5개 미만인 클래스)
    class_counts = pd.Series(y_direction).value_counts()
    small_classes = class_counts[class_counts < 5].index
    if len(small_classes) > 0:
        print(f"소수 클래스({small_classes.tolist()})를 가장 큰 클래스로 병합")
        most_common_class = class_counts.index[0]
        y_direction_adjusted = y_direction.copy()
        for small_class in small_classes:
            y_direction_adjusted[y_direction == small_class] = most_common_class
        y_direction = y_direction_adjusted
    
    # Train-Test Split
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
        X_dir, y_direction, test_size=0.2, random_state=42, stratify=y_direction
    )
    
    scaler_direction = RobustScaler()
    X_train_d_scaled = scaler_direction.fit_transform(X_train_d)
    X_test_d_scaled = scaler_direction.transform(X_test_d)
    
    # 방향 모델 앙상블 구성
    rf_direction_tune = RandomForestClassifier(
        random_state=42, 
        class_weight='balanced',
        bootstrap=True,
        oob_score=True
    )
    
    grid_direction = GridSearchCV(
        rf_direction_tune, param_grid_clf, cv=cv_folds, scoring='f1_weighted', n_jobs=-1, verbose=1
    )
    grid_direction.fit(X_train_d_scaled, y_train_d)
    
    # 최적화된 방향 앙상블 모델
    best_params_direction = grid_direction.best_params_
    print(f"Direction RF 최적 파라미터: {best_params_direction}")
    
    optimized_rf_direction = RandomForestClassifier(
        **best_params_direction,
        random_state=42,
        class_weight='balanced',
        bootstrap=True,
        oob_score=True
    )
    
    optimized_gb_direction = GradientBoostingClassifier(
        random_state=42,
        learning_rate=0.1,
        max_depth=min(5, best_params_direction.get('max_depth', 5)),
        min_samples_split=max(20, best_params_direction.get('min_samples_split', 20)),
        min_samples_leaf=max(10, best_params_direction.get('min_samples_leaf', 10)),
        subsample=0.8,
        n_estimators=150
    )
    
    # 방향 앙상블 모델
    best_direction_model = VotingClassifier([
        ('rf', optimized_rf_direction),
        ('gb', optimized_gb_direction)
    ], voting='soft', weights=[0.6, 0.4])
    
    best_direction_model.fit(X_train_d_scaled, y_train_d)
    
    # 성능 평가
    y_train_pred_d = best_direction_model.predict(X_train_d_scaled)
    y_test_pred_d = best_direction_model.predict(X_test_d_scaled)
    
    train_acc_d = (y_train_pred_d == y_train_d).mean()
    test_acc_d = (y_test_pred_d == y_test_d).mean()
    
    cv_scores_d = cross_val_score(best_direction_model, X_train_d_scaled, y_train_d, cv=5)
    
    print(f"✅ Direction 앙상블 모델 성능:")
    print(f"   - RF 최적 파라미터: {best_params_direction}")
    print(f"   - 훈련 정확도: {train_acc_d:.4f}, 테스트 정확도: {test_acc_d:.4f}")
    print(f"   - 교차검증 정확도: {cv_scores_d.mean():.4f} ± {cv_scores_d.std():.4f}")
    
    # Direction RF OOB 점수
    try:
        optimized_rf_direction.fit(X_train_d_scaled, y_train_d)
        print(f"   - RF OOB Score: {optimized_rf_direction.oob_score_:.4f}")
    except:
        print(f"   - RF OOB Score: 계산 불가")
    
    # 100% 정확도 경고
    if test_acc_d >= 0.95:
        print("🔴 경고: Direction 모델 성능이 의심스럽게 높음 (95%+)")
        print("    - 숨겨진 데이터 누수 가능성")
        print("    - 클래스 불균형으로 인한 허위 성능")
    
    # 과적합 경고
    if train_acc_d - test_acc_d > 0.1:
        print("⚠️ 경고: Direction 모델 과적합 의심")
    
    direction_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    print(classification_report(y_test_d, y_test_pred_d, 
                               target_names=direction_names[:len(np.unique(y_direction))], 
                               zero_division=0))
    
    return (best_speed_model, scaler_speed, selected_features,
            best_direction_model, scaler_direction, direction_features)

def main():
    """메인 실행 함수 (데이터 누수 방지)"""
    print("🚀 데이터 누수 방지 모델 학습 시작...")
    
    # 기존 정제된 데이터 로드
    cleaned_df = load_and_clean_existing_data()
    
    if cleaned_df is None or len(cleaned_df) < 10:
        print("❌ 충분한 데이터를 로드하지 못했습니다.")
        return
    
    # 샘플링 (전체 데이터 사용 권장, 테스트시에만 제한)
    if len(cleaned_df) > 200:
        # 전체 데이터 사용
        working_df = cleaned_df.copy()
        print(f"📊 전체 데이터 사용: {len(working_df)}건")
    else:
        # 작은 데이터셋인 경우 모두 사용
        working_df = cleaned_df.copy()
        print(f"📊 사용 가능한 모든 데이터 사용: {len(working_df)}건")
    
    # 모델별 데이터 준비
    df_clean, X_all, feature_cols = prepare_data_for_models(working_df)
    
    # Area 모델 학습
    area_model, area_scaler, area_features = train_area_model(df_clean, X_all, feature_cols)
    
    # Speed/Direction 모델 학습
    (speed_model, speed_scaler, speed_features,
     direction_model, direction_scaler, direction_features) = train_speed_direction_models(df_clean, X_all, feature_cols)
    
    # 모델 저장
    print("\n💾 모델 저장 중...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Area 모델
    joblib.dump(area_model, os.path.join(script_dir, AREA_MODEL_PATH))
    joblib.dump(area_scaler, os.path.join(script_dir, AREA_SCALER_PATH))
    with open(os.path.join(script_dir, AREA_COLUMNS_PATH), 'w') as f:
        json.dump(area_features, f, indent=2)
    
    # Speed 모델
    joblib.dump(speed_model, os.path.join(script_dir, SPEED_MODEL_PATH))
    joblib.dump(speed_scaler, os.path.join(script_dir, SPEED_SCALER_PATH))
    with open(os.path.join(script_dir, SPEED_COLUMNS_PATH), 'w') as f:
        json.dump(speed_features, f, indent=2)
    
    # Direction 모델
    joblib.dump(direction_model, os.path.join(script_dir, DIRECTION_MODEL_PATH))
    joblib.dump(direction_scaler, os.path.join(script_dir, DIRECTION_SCALER_PATH))
    with open(os.path.join(script_dir, DIRECTION_COLUMNS_PATH), 'w') as f:
        json.dump(direction_features, f, indent=2)
    
    print("🎉 앙상블 모델 학습 완료!")
    print(f"📊 Area 앙상블: RandomForest + GradientBoosting ({len(area_features)}개 피처)")
    print(f"🚀 Speed 앙상블: RandomForest + GradientBoosting ({len(speed_features)}개 피처)")
    print(f"🧭 Direction 앙상블: RandomForest + GradientBoosting ({len(direction_features)}개 피처)")
    print("\n✅ 주요 개선사항:")
    print("   - 🧠 고급 피처 엔지니어링 (체감온도, 증발산량, 순환 인코딩 등)")
    print("   - 🤝 다중 모델 앙상블 (RandomForest + GradientBoosting)")
    print("   - ⏰ 시간적 누수 제거 (종료시점, 미래데이터)")
    print("   - 🎯 타겟 누수 제거 (파생변수)")
    print("   - 🛡️ 강화된 과적합 방지 (정규화, 교차검증)")
    print("   - ⚙️ 하이퍼파라미터 최적화 (GridSearchCV)")
    print("   - 🗳️ 앙상블 가중 투표 (RF 60%, GB 40%)")
    print("   - 📊 성능 모니터링 (OOB, 교차검증, 경고 시스템)")
    
    print("\n🏗️ 모델 아키텍처:")
    print("   📈 Area: VotingRegressor (연속값 예측)")
    print("      └─ RandomForest (60%) + GradientBoosting (40%)")
    print("   🚀 Speed: VotingClassifier with soft voting (확률 기반)")
    print("      └─ RandomForest (60%) + GradientBoosting (40%)")
    print("   🧭 Direction: VotingClassifier with soft voting (확률 기반)")
    print("      └─ RandomForest (60%) + GradientBoosting (40%)")
    
    print("\n📈 성능 신뢰성 가이드:")
    print("   🟢 신뢰할 수 있는 지표:")
    print("      - 훈련-테스트 성능 차이 < 0.1")
    print("      - 교차검증 표준편차 < 0.1")
    print("      - OOB Score와 테스트 성능 유사")
    print("   🔴 의심해야 할 지표:")
    print("      - 95%+ 정확도 (특히 Direction)")
    print("      - 훈련-테스트 성능 차이 > 0.1")
    print("      - 교차검증 성능 급락")

if __name__ == "__main__":
    main()