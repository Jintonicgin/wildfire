#!/usr/bin/env python3
"""
화재 면적 예측이 어려운 이유 분석 및 새로운 접근법
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

def analyze_fire_area_problem():
    """화재 면적 예측 문제 분석"""
    print("🔍 화재 면적 예측이 어려운 이유 분석")
    print("=" * 50)
    
    # 데이터 로드
    df = pd.read_csv('final_merged_feature_engineered.csv', low_memory=False)
    fire_mask = (df['fire_area'] > 0) & df['fire_area'].notna()
    fire_df = df[fire_mask].copy()
    
    area_data = fire_df['fire_area']
    
    print(f"\\n📊 데이터 기본 정보:")
    print(f"   총 화재 건수: {len(area_data)}")
    print(f"   면적 범위: {area_data.min():.3f} ~ {area_data.max():.3f} ha")
    print(f"   평균: {area_data.mean():.3f} ha")
    print(f"   중앙값: {area_data.median():.3f} ha")
    
    # 1. 분포 문제 분석
    print("\\n🎯 문제 1: 극도로 왜곡된 분포")
    
    # 분위수별 분포
    quantiles = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
    cumulative_area = []
    
    for q in quantiles:
        threshold = area_data.quantile(q)
        count = (area_data <= threshold).sum()
        total_area = area_data[area_data <= threshold].sum()
        
        print(f"   {q*100:4.1f}% 화재: {threshold:8.3f}ha 이하, 총면적의 {total_area/area_data.sum()*100:4.1f}%")
    
    # 2. 큰 화재의 영향
    print("\\n🔥 문제 2: 소수의 대형 화재가 전체 면적 좌우")
    
    large_fires = area_data[area_data > area_data.quantile(0.99)]
    print(f"   상위 1% 화재({len(large_fires)}건)가 전체 면적의 {large_fires.sum()/area_data.sum()*100:.1f}% 차지")
    print(f"   이들 화재: {large_fires.values}")
    
    # 3. 예측 변수와의 상관관계 분석
    print("\\n📈 문제 3: 예측 변수와 면적의 상관관계")
    
    # 주요 예측 변수들
    predictors = ['fwi_0h', 'isi_0h', 'ws10m_0h', 'rh2m_0h', 't2m_0h', 'ffmc_0h']
    correlations = {}
    
    for pred in predictors:
        if pred in fire_df.columns:
            # 원본 상관관계
            corr_original = fire_df[pred].corr(fire_df['fire_area'])
            # 로그 변환 후 상관관계
            corr_log = fire_df[pred].corr(np.log1p(fire_df['fire_area']))
            
            correlations[pred] = {
                'original': corr_original,
                'log_transformed': corr_log
            }
            
            print(f"   {pred:10}: 원본 {corr_original:6.3f}, 로그변환 {corr_log:6.3f}")
    
    # 4. 시간적 패턴 분석
    print("\\n📅 문제 4: 시간적 패턴")
    
    if 'fire_month' in fire_df.columns:
        monthly_stats = fire_df.groupby('fire_month')['fire_area'].agg(['count', 'mean', 'median', 'max'])
        print("   월별 화재 통계:")
        for month in sorted(fire_df['fire_month'].unique()):
            if month in monthly_stats.index:
                stats = monthly_stats.loc[month]
                print(f"   {int(month):2d}월: {int(stats['count']):3d}건, 평균 {stats['mean']:6.2f}ha, 최대 {stats['max']:8.2f}ha")
    
    # 5. 지형적 패턴
    print("\\n🏔️ 문제 5: 지형적 요인")
    
    if 'elevation_mean' in fire_df.columns and 'slope_mean' in fire_df.columns:
        # 고도별
        fire_df['elevation_group'] = pd.cut(fire_df['elevation_mean'], bins=5, labels=['매우낮음', '낮음', '중간', '높음', '매우높음'])
        elev_stats = fire_df.groupby('elevation_group')['fire_area'].agg(['count', 'mean', 'max'])
        print("   고도별 화재 통계:")
        for group in elev_stats.index:
            if pd.notna(group):
                stats = elev_stats.loc[group]
                print(f"   {group:6}: {int(stats['count']):3d}건, 평균 {stats['mean']:6.2f}ha")
    
    return fire_df, correlations

def try_segmented_modeling(fire_df):
    """구간별 모델링 시도"""
    print("\\n🎯 새로운 접근법 1: 구간별 모델링")
    print("=" * 50)
    
    # 화재를 크기별로 구분하여 각각 다른 모델 적용
    area_data = fire_df['fire_area']
    
    # 구간 정의
    small_threshold = area_data.quantile(0.8)   # 하위 80%
    large_threshold = area_data.quantile(0.95)  # 상위 5%
    
    print(f"소형 화재: ≤ {small_threshold:.3f} ha")
    print(f"중형 화재: {small_threshold:.3f} ~ {large_threshold:.3f} ha") 
    print(f"대형 화재: > {large_threshold:.3f} ha")
    
    # 피처 준비
    features = ['fwi_0h', 'isi_0h', 'ws10m_0h', 'rh2m_0h', 't2m_0h', 'ffmc_0h', 
                'elevation_mean', 'slope_mean']
    available_features = [f for f in features if f in fire_df.columns]
    
    X = fire_df[available_features].fillna(fire_df[available_features].median())
    y = fire_df['fire_area']
    
    # 구간별 모델 성능 테스트
    segments = {
        'small': y <= small_threshold,
        'medium': (y > small_threshold) & (y <= large_threshold),
        'large': y > large_threshold
    }
    
    segment_results = {}
    
    for segment_name, mask in segments.items():
        if mask.sum() < 20:  # 최소 20개 샘플 필요
            print(f"   {segment_name}: 샘플 부족 ({mask.sum()}개)")
            continue
            
        print(f"\\n   {segment_name} 화재 모델 ({mask.sum()}개 샘플):")
        
        X_seg = X[mask]
        y_seg = y[mask]
        
        # 로그 변환
        y_seg_log = np.log1p(y_seg)
        
        # 분할
        if len(X_seg) > 50:
            X_train, X_test, y_train, y_test = train_test_split(
                X_seg, y_seg_log, test_size=0.3, random_state=42
            )
        else:
            # 샘플이 적으면 전체를 사용
            X_train, X_test = X_seg, X_seg
            y_train, y_test = y_seg_log, y_seg_log
        
        # 모델 훈련
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 예측
        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)  # 원래 스케일로
        y_test_original = np.expm1(y_test)
        
        # 평가
        r2 = r2_score(y_test_original, y_pred)
        rmse = np.sqrt(((y_test_original - y_pred) ** 2).mean())
        
        segment_results[segment_name] = {
            'r2': r2,
            'rmse': rmse,
            'n_samples': mask.sum(),
            'model': model
        }
        
        print(f"     R²: {r2:.4f}, RMSE: {rmse:.2f} ha")
    
    return segment_results

def try_duration_based_modeling(fire_df):
    """지속시간 기반 모델링"""
    print("\\n🎯 새로운 접근법 2: 지속시간 추정 후 면적 계산")
    print("=" * 50)
    
    # 가정: 면적 = 확산속도 × 지속시간 × 방향수
    # 확산속도와 지속시간을 별도로 모델링
    
    # 대략적인 지속시간 추정 (면적 기반)
    # 작은 화재: 몇 시간, 큰 화재: 며칠
    fire_df_copy = fire_df.copy()
    
    # 추정 지속시간 (시간 단위)
    area = fire_df_copy['fire_area']
    estimated_duration = np.where(area <= 1, np.sqrt(area) * 4,      # 1ha 이하: 최대 4시간
                         np.where(area <= 10, 4 + (area-1) * 2,      # 1-10ha: 4-22시간  
                                 22 + (area-10) * 0.5))              # 10ha 이상: 22시간+
    
    fire_df_copy['estimated_duration'] = estimated_duration
    
    # 확산속도 계산 (ha/hour)
    fire_df_copy['spread_rate'] = area / estimated_duration
    
    print(f"   추정 지속시간: {estimated_duration.min():.1f} ~ {estimated_duration.max():.1f} 시간")
    print(f"   확산속도: {fire_df_copy['spread_rate'].min():.3f} ~ {fire_df_copy['spread_rate'].max():.3f} ha/h")
    
    # 확산속도 예측 모델
    features = ['fwi_0h', 'isi_0h', 'ws10m_0h', 'rh2m_0h', 't2m_0h']
    available_features = [f for f in features if f in fire_df.columns]
    
    X = fire_df_copy[available_features].fillna(fire_df_copy[available_features].median())
    y_rate = fire_df_copy['spread_rate']
    y_duration = fire_df_copy['estimated_duration']
    
    print("\\n   확산속도 예측 모델:")
    X_train, X_test, y_rate_train, y_rate_test = train_test_split(
        X, np.log1p(y_rate), test_size=0.3, random_state=42
    )
    
    rate_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rate_model.fit(X_train, y_rate_train)
    
    y_rate_pred_log = rate_model.predict(X_test)
    y_rate_pred = np.expm1(y_rate_pred_log)
    y_rate_test_original = np.expm1(y_rate_test)
    
    rate_r2 = r2_score(y_rate_test_original, y_rate_pred)
    print(f"     확산속도 R²: {rate_r2:.4f}")
    
    print("\\n   지속시간 예측 모델:")
    X_train, X_test, y_dur_train, y_dur_test = train_test_split(
        X, np.log1p(y_duration), test_size=0.3, random_state=42
    )
    
    duration_model = RandomForestRegressor(n_estimators=100, random_state=42)
    duration_model.fit(X_train, y_dur_train)
    
    y_dur_pred_log = duration_model.predict(X_test)
    y_dur_pred = np.expm1(y_dur_pred_log)
    y_dur_test_original = np.expm1(y_dur_test)
    
    duration_r2 = r2_score(y_dur_test_original, y_dur_pred)
    print(f"     지속시간 R²: {duration_r2:.4f}")
    
    # 최종 면적 예측
    area_pred = y_rate_pred * y_dur_pred
    area_test = y_rate_test_original * y_dur_test_original
    
    final_r2 = r2_score(area_test, area_pred)
    print(f"\\n   🎯 최종 면적 예측 R²: {final_r2:.4f}")
    
    return final_r2, rate_r2, duration_r2

def try_probabilistic_approach(fire_df):
    """확률적 접근법"""
    print("\\n🎯 새로운 접근법 3: 확률적 면적 예측")
    print("=" * 50)
    
    # 기상 조건에 따른 확률 분포 예측
    # 작은 화재 확률, 중간 화재 확률, 큰 화재 확률을 각각 예측
    
    area = fire_df['fire_area']
    
    # 확률 구간 정의
    thresholds = [0.1, 1.0, 10.0, 100.0]  # ha 단위
    prob_categories = []
    
    for i, row in fire_df.iterrows():
        fire_area = row['fire_area']
        if fire_area <= thresholds[0]:
            prob_categories.append(0)  # 매우 작음
        elif fire_area <= thresholds[1]:
            prob_categories.append(1)  # 작음
        elif fire_area <= thresholds[2]:
            prob_categories.append(2)  # 중간
        elif fire_area <= thresholds[3]:
            prob_categories.append(3)  # 큼
        else:
            prob_categories.append(4)  # 매우 큼
    
    fire_df_prob = fire_df.copy()
    fire_df_prob['area_category'] = prob_categories
    
    # 각 카테고리별 분포
    category_counts = pd.Series(prob_categories).value_counts().sort_index()
    print("   면적 카테고리별 분포:")
    category_names = ['매우작음(≤0.1)', '작음(0.1-1)', '중간(1-10)', '큼(10-100)', '매우큼(>100)']
    
    expected_areas = []
    for i, count in category_counts.items():
        mask = fire_df_prob['area_category'] == i
        avg_area = fire_df_prob[mask]['fire_area'].mean()
        expected_areas.append(avg_area)
        print(f"   {category_names[i]:12}: {count:3d}건 ({count/len(fire_df_prob)*100:4.1f}%), 평균 {avg_area:.2f}ha")
    
    # 분류 모델로 카테고리 예측 후 기댓값 계산
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    features = ['fwi_0h', 'isi_0h', 'ws10m_0h', 'rh2m_0h', 't2m_0h', 'ffmc_0h']
    available_features = [f for f in features if f in fire_df.columns]
    
    X = fire_df_prob[available_features].fillna(fire_df_prob[available_features].median())
    y_category = fire_df_prob['area_category']
    
    X_train, X_test, y_cat_train, y_cat_test = train_test_split(
        X, y_category, test_size=0.3, random_state=42
    )
    
    # 분류 모델 훈련
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_cat_train)
    
    # 확률 예측
    y_proba = classifier.predict_proba(X_test)
    
    # 기댓값으로 면적 예측
    area_predictions = []
    for probs in y_proba:
        expected_area = sum(prob * expected_areas[i] for i, prob in enumerate(probs) if i < len(expected_areas))
        area_predictions.append(expected_area)
    
    # 실제 면적
    y_area_test = fire_df_prob.iloc[y_cat_test.index]['fire_area'].values
    
    # 평가
    prob_r2 = r2_score(y_area_test, area_predictions)
    prob_rmse = np.sqrt(((y_area_test - area_predictions) ** 2).mean())
    
    # 분류 정확도
    y_cat_pred = classifier.predict(X_test)
    cat_accuracy = accuracy_score(y_cat_test, y_cat_pred)
    
    print(f"\\n   분류 정확도: {cat_accuracy:.4f}")
    print(f"   🎯 확률적 면적 예측 R²: {prob_r2:.4f}")
    print(f"   RMSE: {prob_rmse:.2f} ha")
    
    return prob_r2, cat_accuracy

def main():
    """메인"""
    print("🔍 화재 면적 예측 근본 분석 및 새로운 접근법")
    print("=" * 60)
    
    # 1. 문제 분석
    fire_df, correlations = analyze_fire_area_problem()
    
    # 2. 구간별 모델링
    segment_results = try_segmented_modeling(fire_df)
    
    # 3. 지속시간 기반 모델링  
    duration_r2, rate_r2, dur_r2 = try_duration_based_modeling(fire_df)
    
    # 4. 확률적 접근법
    prob_r2, cat_acc = try_probabilistic_approach(fire_df)
    
    # 최종 결과 비교
    print("\\n" + "=" * 60)
    print("🏆 새로운 접근법 결과 비교")
    print("=" * 60)
    
    print(f"기존 모델 (단순 회귀):     R² = 17.9%")
    print(f"지속시간 기반 모델:        R² = {duration_r2:.1%}")
    print(f"확률적 접근법:           R² = {prob_r2:.1%}")
    
    if segment_results:
        print("\\n구간별 모델링 결과:")
        for segment, result in segment_results.items():
            print(f"   {segment:6} 화재: R² = {result['r2']:.1%} ({result['n_samples']}개 샘플)")
    
    # 최고 성능
    all_r2 = [0.179, duration_r2, prob_r2]
    method_names = ['기존', '지속시간', '확률적']
    
    best_idx = np.argmax(all_r2)
    best_r2 = all_r2[best_idx]
    best_method = method_names[best_idx]
    
    print(f"\\n🥇 최고 성능: {best_method} 방법 (R² = {best_r2:.1%})")
    
    if best_r2 > 0.4:
        print("✅ 목표 달성! 실용적 수준의 예측 성능")
    elif best_r2 > 0.25:
        print("📊 상당한 개선. 참고용으로 활용 가능")
    else:
        print("⚠️ 여전히 개선 필요. 화재 면적 예측은 근본적으로 어려운 문제")
        print("\\n💡 권장사항:")
        print("   1. 더 많은 실시간 데이터 필요 (위성, 드론 등)")
        print("   2. 진압 활동 정보 포함")
        print("   3. 미기상 데이터 활용")
        print("   4. 연료량 및 식생 상태 정보")

if __name__ == "__main__":
    main()