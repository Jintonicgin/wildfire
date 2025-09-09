import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def generate_statistical_features(weather_features):
    """시간별 날씨 데이터로부터 통계 피처들을 생성합니다."""
    stats_features = {}
    
    # 파라미터별로 통계 계산
    weather_params = ['t2m', 'rh2m', 'ws10m', 'wd10m', 'prec', 'ps', 'solar', 'ws2m', 'wd2m']
    
    for param in weather_params:
        # 전체 기간 데이터 수집 (0~168시간)
        all_values = []
        day_values = []  # 24시간 데이터
        
        for hour in range(0, 169, 3):
            key = f"{param}_{hour}h_past"
            if key in weather_features:
                val = weather_features[key]
                if val is not None and pd.notna(val):
                    all_values.append(float(val))
                    if hour <= 24:  # 24시간 이내
                        day_values.append(float(val))
        
        # 전체 기간 통계
        if all_values:
            stats_features[f"{param}_max_past"] = max(all_values)
            stats_features[f"{param}_min_past"] = min(all_values)
            stats_features[f"{param}_mean_past"] = sum(all_values) / len(all_values)
            stats_features[f"{param}_std_past"] = (sum([(x - stats_features[f"{param}_mean_past"])**2 for x in all_values]) / len(all_values))**0.5
        else:
            for stat in ['max', 'min', 'mean', 'std']:
                stats_features[f"{param}_{stat}_past"] = -999
            
        # 24시간 통계  
        if day_values:
            stats_features[f"{param}_max_24h_past"] = max(day_values)
            stats_features[f"{param}_min_24h_past"] = min(day_values)
            stats_features[f"{param}_mean_24h_past"] = sum(day_values) / len(day_values)
            stats_features[f"{param}_std_24h_past"] = (sum([(x - stats_features[f"{param}_mean_24h_past"])**2 for x in day_values]) / len(day_values))**0.5
        else:
            for stat in ['max', 'min', 'mean', 'std']:
                stats_features[f"{param}_{stat}_24h_past"] = -999
    
    return stats_features

def generate_custom_features(weather_features, gee_features):
    """커스텀 화재 위험 피처들을 생성합니다."""
    custom_features = {}
    
    # 현재 시점 기상 데이터 (0h_past)
    t2m_0h = weather_features.get('t2m_0h_past', 20)
    rh2m_0h = weather_features.get('rh2m_0h_past', 50)
    ws10m_0h = weather_features.get('ws10m_0h_past', 5)
    wd10m_0h = weather_features.get('wd10m_0h_past', 180)
    prec_0h = weather_features.get('prec_0h_past', 0)
    
    # 지형 데이터
    slope_mean = gee_features.get('slope_mean', 5)
    slope_std = gee_features.get('slope_std', 2) 
    aspect_mode = gee_features.get('aspect_mode', 180)
    elevation_mean = gee_features.get('elevation_mean', 300)
    elevation_std = gee_features.get('elevation_std', 50)
    ndvi = gee_features.get('ndvi_before', 0.5)
    treecover = gee_features.get('treecover_pre_fire_5x5', 50)
    
    # 1. 지형-기후 상관관계 피처들
    
    # 1-1. 바람방향과 경사면 상호작용 (바람이 경사면을 타고 올라가는 효과)
    wind_slope_angle = abs(wd10m_0h - aspect_mode)
    if wind_slope_angle > 180:
        wind_slope_angle = 360 - wind_slope_angle
    # 바람이 경사면을 정면으로 타격할 때 최대 위험
    custom_features['wind_slope_interaction'] = float(ws10m_0h * slope_mean * (1 - wind_slope_angle/180))
    
    # 1-2. 고도-온도 관계 (고도가 높으면 온도 낮음)
    expected_temp_at_elevation = t2m_0h - (elevation_mean * 0.0065)  # 고도 100m당 0.65도 하락
    custom_features['elevation_temp_deviation'] = float(t2m_0h - expected_temp_at_elevation)
    
    # 1-3. 지형 복잡성 지수 (경사도 변동성과 고도 변동성)
    custom_features['terrain_complexity_index'] = float((slope_std / max(slope_mean, 1)) + (elevation_std / max(elevation_mean, 1)))
    
    # 1-4. 골짜기 효과 (낮은 고도 + 높은 습도)
    if elevation_mean < 200 and rh2m_0h > 70:
        custom_features['valley_moisture_trap'] = 1
    else:
        custom_features['valley_moisture_trap'] = 0
    
    # 2. 식생-기후 상관관계 피처들
    
    # 2-1. NDVI-온도 스트레스 (높은 온도 + 낮은 NDVI = 식생 스트레스)
    temp_stress = max(0, (t2m_0h - 25) / 15) if t2m_0h > 25 else 0
    ndvi_stress = max(0, (0.6 - ndvi) / 0.6) if ndvi < 0.6 else 0
    custom_features['vegetation_heat_stress'] = float(temp_stress * ndvi_stress)
    
    # 2-2. NDVI-강수량 관계 (가뭄 스트레스)
    total_precip_7d = weather_features.get('total_prec_7d_start_past', 0)
    drought_stress = max(0, (10 - total_precip_7d) / 10) if total_precip_7d < 10 else 0
    custom_features['vegetation_drought_stress'] = float(drought_stress * ndvi_stress)
    
    # 2-3. 산림 밀도와 습도 관계 (밀집된 산림은 습도 유지)
    if treecover > 70:
        expected_humidity = rh2m_0h + 10  # 산림은 습도를 높임
    else:
        expected_humidity = rh2m_0h
    custom_features['forest_humidity_effect'] = float(expected_humidity - rh2m_0h)
    
    # 3. 기존 복합 위험도 피처들 (개선된 버전)
    
    # 3-1. DRY_WINDY_COMBO - 건조하고 바람이 강한 조건 (지형 고려)
    dry_factor = max(0, (100 - rh2m_0h) / 100)
    wind_factor = min(ws10m_0h / 20.0, 1.0)
    terrain_factor = min(slope_mean / 30.0, 1.0)
    custom_features['dry_windy_combo'] = float(dry_factor * wind_factor * (1 + terrain_factor))
    
    # 3-2. FUEL_COMBO - 연료량과 건조도 결합 (개선)
    if ndvi < 0.3 and rh2m_0h < 40:
        fuel_score = 1.0  # 건조한 초지
    elif treecover > 50 and rh2m_0h < 50 and total_precip_7d < 5:
        fuel_score = 0.8  # 건조한 산림
    elif ndvi > 0.6 and rh2m_0h > 60:
        fuel_score = 0.2  # 건강한 식생
    else:
        fuel_score = 0.5  # 중간 상태
    custom_features['fuel_combo'] = float(fuel_score)
    
    # 3-3. POTENTIAL_SPREAD_INDEX - 확산 잠재력 지수 (지형 포함)
    spread_index = (wind_factor * 0.3 + dry_factor * 0.3 + temp_stress * 0.2 + terrain_factor * 0.2)
    custom_features['potential_spread_index'] = float(spread_index)
    
    # 3-4. TERRAIN_VAR_EFFECT - 지형 변동성 효과 (개선)
    custom_features['terrain_var_effect'] = float(terrain_factor * (1 + slope_std/10))
    
    # 4. 바람 패턴 분석
    wind_values = []
    for hour in range(0, 25, 3):
        key = f"ws10m_{hour}h_past"
        if key in weather_features and weather_features[key] is not None:
            wind_values.append(weather_features[key])
    
    if len(wind_values) > 3:
        wind_mean = sum(wind_values) / len(wind_values)
        wind_std = (sum([(x - wind_mean)**2 for x in wind_values]) / len(wind_values))**0.5
        custom_features['wind_steady_flag'] = 1 if wind_std < 2.0 else 0
        custom_features['wind_strength_consistency'] = float(wind_mean * (1 - wind_std/wind_mean) if wind_mean > 0 else 0)
    else:
        custom_features['wind_steady_flag'] = 0
        custom_features['wind_strength_consistency'] = 0
    
    # 5. 장기 기후 트렌드
    total_precip_30d = weather_features.get('total_prec_30d_start_past', 0)
    dry_days_30d = weather_features.get('dry_days_30d_start_past', 30)
    
    if total_precip_30d > 0:
        custom_features['dry_to_rain_ratio_30d'] = float(dry_days_30d / max(total_precip_30d, 0.1))
    else:
        custom_features['dry_to_rain_ratio_30d'] = 30.0  # 완전 건조
        
    # 6. 통합 위험도 스코어 (모든 요소 종합)
    base_risk = (dry_factor + wind_factor + temp_stress) / 3
    terrain_amplifier = 1 + terrain_factor * 0.5
    vegetation_modifier = 1 - (ndvi * 0.3) if ndvi > 0.5 else 1 + (0.5 - ndvi) * 0.5
    
    custom_features['integrated_fire_risk_score'] = float(base_risk * terrain_amplifier * vegetation_modifier)
    
    return custom_features
