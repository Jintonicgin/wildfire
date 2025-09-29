#!/usr/bin/env python3
"""
데이터 품질 분석 및 개선점 찾기
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_data():
    """원본 데이터 로드 및 기본 분석"""
    print("📊 데이터 품질 분석 시작...")
    
    # 데이터 로드
    df = pd.read_csv('final_merged_feature_engineered.csv')
    print(f"원본 데이터: {df.shape}")
    
    # 기본 화재 데이터 필터링
    fire_data = df[(df['fire_area'] > 0) & (df['fire_area'].notna())].copy()
    print(f"화재 데이터: {fire_data.shape}")
    
    return df, fire_data

def analyze_target_distribution(fire_data):
    """타겟 변수 분포 분석"""
    print("\n🎯 타겟 변수 분석...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 화재 면적 분포
    axes[0, 0].hist(fire_data['fire_area'], bins=50, alpha=0.7, color='orange')
    axes[0, 0].set_title('화재 면적 분포')
    axes[0, 0].set_xlabel('면적 (ha)')
    axes[0, 0].set_ylabel('빈도')
    
    # 2. 로그 변환된 화재 면적
    log_area = np.log1p(fire_data['fire_area'])
    axes[0, 1].hist(log_area, bins=50, alpha=0.7, color='red')
    axes[0, 1].set_title('로그 변환된 화재 면적')
    axes[0, 1].set_xlabel('log(면적 + 1)')
    axes[0, 1].set_ylabel('빈도')
    
    # 3. 면적 vs 지속시간
    duration_col = None
    for col in ['fire_duration_hours', 'duration_hours', 'total_hours']:
        if col in fire_data.columns:
            duration_col = col
            break
    
    if duration_col:
        scatter_data = fire_data[[duration_col, 'fire_area']].dropna()
        axes[1, 0].scatter(scatter_data[duration_col], scatter_data['fire_area'], alpha=0.5)
        axes[1, 0].set_title('화재 면적 vs 지속시간')
        axes[1, 0].set_xlabel('지속시간 (시간)')
        axes[1, 0].set_ylabel('면적 (ha)')
    
    # 4. 월별 화재 패턴
    if 'startmonth' in fire_data.columns:
        monthly = fire_data.groupby('startmonth')['fire_area'].agg(['count', 'mean'])
        axes[1, 1].bar(monthly.index, monthly['mean'], alpha=0.7, color='green')
        axes[1, 1].set_title('월별 평균 화재 면적')
        axes[1, 1].set_xlabel('월')
        axes[1, 1].set_ylabel('평균 면적 (ha)')
    
    plt.tight_layout()
    plt.savefig('target_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 화재 면적 통계:")
    print(f"   - 평균: {fire_data['fire_area'].mean():.4f} ha")
    print(f"   - 중위수: {fire_data['fire_area'].median():.4f} ha") 
    print(f"   - 표준편차: {fire_data['fire_area'].std():.4f} ha")
    print(f"   - 최대값: {fire_data['fire_area'].max():.4f} ha")
    print(f"   - 왜도: {fire_data['fire_area'].skew():.4f}")

def find_important_features(fire_data):
    """중요한 피처 발견"""
    print("\n🔍 피처 중요도 분석...")
    
    # 수치형 컬럼만 선택
    numeric_cols = fire_data.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != 'fire_area']
    
    # 결측치와 무한값 처리
    X = fire_data[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # 결측치가 80% 이상인 컬럼 제거
    missing_pct = X.isnull().sum() / len(X)
    good_cols = missing_pct[missing_pct < 0.8].index.tolist()
    X = X[good_cols]
    
    # 극값 클리핑 (더 강력하게)
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            # 99%ile로 클리핑
            q99 = X[col].quantile(0.99)
            q01 = X[col].quantile(0.01)
            if pd.notna(q99) and pd.notna(q01) and q99 != q01:
                # 매우 안전한 범위로 제한
                safe_max = min(q99, 1e4)  # 최대 10,000
                safe_min = max(q01, -1e4)  # 최소 -10,000
                X[col] = X[col].clip(safe_min, safe_max)
    
    # 결측치를 중위수로 채우기
    X = X.fillna(X.median())
    
    # 최종 안전성 체크
    X = X.replace([np.inf, -np.inf], 0)
    
    # 여전히 문제가 있는 컬럼 제거
    problematic_cols = []
    for col in X.columns:
        if not np.all(np.isfinite(X[col])) or X[col].std() == 0:
            problematic_cols.append(col)
    
    if problematic_cols:
        print(f"   - 문제가 있는 컬럼 제거: {len(problematic_cols)}개")
        X = X.drop(columns=problematic_cols)
    
    y = fire_data['fire_area']
    
    print(f"   - 최종 분석 가능한 피처: {len(X.columns)}개")
    
    # 1. RandomForest 중요도
    print("   - RandomForest 피처 중요도 계산 중...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    rf_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 2. 상호정보량
    print("   - 상호정보량 계산 중...")
    mi_scores = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
    
    mi_importance = pd.DataFrame({
        'feature': X.columns,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    # 3. 상관계수 (절댓값)
    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    
    print("\n📊 Top 20 중요 피처 (RandomForest):")
    for i, row in rf_importance.head(20).iterrows():
        print(f"   {i+1:2d}. {row['feature']:<30} ({row['importance']:.4f})")
    
    print("\n📊 Top 20 중요 피처 (상호정보량):")
    for i, row in mi_importance.head(20).iterrows():
        print(f"   {i+1:2d}. {row['feature']:<30} ({row['mi_score']:.4f})")
    
    print("\n📊 Top 20 상관관계:")
    for i, (feature, corr) in enumerate(correlations.head(20).items(), 1):
        print(f"   {i:2d}. {feature:<30} ({corr:.4f})")
    
    return rf_importance, mi_importance, correlations, X, y

def analyze_data_leakage(fire_data):
    """데이터 누수 재분석"""
    print("\n🔍 데이터 누수 재분석...")
    
    # 시간 관련 컬럼 분석
    time_cols = [col for col in fire_data.columns if any(keyword in col.lower() 
                for keyword in ['end', 'finish', 'duration', 'total', 'final'])]
    
    # 미래 시점 컬럼
    future_cols = []
    for col in fire_data.columns:
        if any(pattern in col for pattern in ['_3h', '_6h', '_12h', '_24h', '_48h']):
            # 숫자 추출해서 미래 시점인지 확인
            if any(future_hour in col for future_hour in ['_3h', '_6h', '_12h', '_24h']):
                future_cols.append(col)
    
    print(f"   - 의심스러운 시간 관련 컬럼: {len(time_cols)}개")
    print(f"   - 미래 시점 컬럼: {len(future_cols)}개")
    
    # 높은 상관관계 (누수 의심)
    numeric_cols = fire_data.select_dtypes(include=[np.number]).columns
    high_corr_cols = []
    
    for col in numeric_cols:
        if col != 'fire_area':
            corr = fire_data[col].corr(fire_data['fire_area'])
            if abs(corr) > 0.7:  # 매우 높은 상관관계
                high_corr_cols.append((col, corr))
    
    print(f"\n⚠️ 누수 의심 컬럼들:")
    for col, corr in sorted(high_corr_cols, key=lambda x: abs(x[1]), reverse=True):
        print(f"   - {col}: {corr:.4f}")
    
    return time_cols, future_cols, high_corr_cols

def suggest_improvements(rf_importance, mi_importance, correlations):
    """개선 제안"""
    print("\n💡 성능 개선 제안:")
    
    # 1. 최고 성능 피처들
    top_rf = set(rf_importance.head(30)['feature'])
    top_mi = set(mi_importance.head(30)['feature'])
    top_corr = set(correlations.head(30).index)
    
    # 교집합 - 모든 방법에서 중요한 피처
    consensus_features = top_rf & top_mi & top_corr
    
    print(f"\n🎯 핵심 피처 ({len(consensus_features)}개):")
    for feature in sorted(consensus_features):
        rf_rank = rf_importance[rf_importance['feature'] == feature].index[0] + 1
        mi_rank = mi_importance[mi_importance['feature'] == feature].index[0] + 1
        corr_rank = list(correlations.index).index(feature) + 1
        print(f"   - {feature} (RF:{rf_rank}, MI:{mi_rank}, Corr:{corr_rank})")
    
    # 2. FWI 시스템 피처들
    fwi_features = [col for col in rf_importance['feature'] 
                   if any(fwi in col.lower() for fwi in ['fwi', 'ffmc', 'dmc', 'dc', 'isi', 'bui'])]
    
    print(f"\n🔥 FWI 시스템 피처 ({len(fwi_features)}개):")
    for feature in fwi_features[:10]:
        importance = rf_importance[rf_importance['feature'] == feature]['importance'].iloc[0]
        print(f"   - {feature}: {importance:.4f}")
    
    # 3. 기상 피처들  
    weather_features = [col for col in rf_importance['feature']
                       if any(weather in col.lower() for weather in ['t2m', 'rh2m', 'ws10m', 'ps', 'precip'])]
    
    print(f"\n🌤️ 기상 피처 ({len(weather_features)}개):")
    for feature in weather_features[:10]:
        importance = rf_importance[rf_importance['feature'] == feature]['importance'].iloc[0]
        print(f"   - {feature}: {importance:.4f}")
    
    return consensus_features, fwi_features, weather_features

def create_improved_features(fire_data, consensus_features):
    """개선된 피처 세트 생성"""
    print("\n🔧 개선된 피처 세트 생성...")
    
    # 기존 핵심 피처
    base_features = list(consensus_features)
    
    # 도메인 지식 기반 추가 피처
    domain_features = []
    for col in fire_data.columns:
        if any(keyword in col.lower() for keyword in [
            'fwi', 'ffmc', 'dmc', 'dc', 'isi', 'bui',  # Fire Weather Index
            'elevation', 'slope', 'aspect',             # 지형
            'ndvi', 'treecover',                        # 식생
            't2m_0h', 'rh2m_0h', 'ws10m_0h', 'ps_0h', # 현재 기상
            'dry_days', 'startmonth', 'startday'        # 시간/건조도
        ]):
            if col not in base_features and col != 'fire_area':
                domain_features.append(col)
    
    # 최종 피처 세트
    improved_features = base_features + domain_features[:20]  # 상위 20개 추가
    improved_features = [f for f in improved_features if f in fire_data.columns]
    
    print(f"✅ 개선된 피처 세트: {len(improved_features)}개")
    print("주요 피처들:")
    for i, feature in enumerate(improved_features[:15], 1):
        print(f"   {i:2d}. {feature}")
    
    return improved_features

def main():
    """메인 분석 함수"""
    print("🚀 데이터 품질 분석 및 개선안 도출...")
    
    # 1. 데이터 로드
    df, fire_data = load_and_analyze_data()
    
    # 2. 타겟 분석
    analyze_target_distribution(fire_data)
    
    # 3. 피처 중요도 분석
    rf_importance, mi_importance, correlations, X, y = find_important_features(fire_data)
    
    # 4. 데이터 누수 재분석
    time_cols, future_cols, high_corr_cols = analyze_data_leakage(fire_data)
    
    # 5. 개선 제안
    consensus_features, fwi_features, weather_features = suggest_improvements(
        rf_importance, mi_importance, correlations)
    
    # 6. 개선된 피처 세트
    improved_features = create_improved_features(fire_data, consensus_features)
    
    print("\n📋 분석 완료! 다음 단계:")
    print("1. target_analysis.png 확인")
    print("2. 핵심 피처들로 모델 재학습")
    print("3. 도메인 지식 기반 피처 엔지니어링 강화")
    print("4. 고급 모델 (XGBoost, LightGBM) 시도")

if __name__ == "__main__":
    main()