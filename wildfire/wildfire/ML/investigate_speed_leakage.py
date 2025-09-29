#!/usr/bin/env python3
"""
속도 모델 데이터 누출 조사
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_analyze():
    """속도 모델 데이터 누출 조사"""
    print("🔍 속도 모델 데이터 누출 조사...")
    
    # 데이터 로드
    df = pd.read_csv('final_merged_feature_engineered.csv')
    fire_mask = (df['fire_area'] > 0) & df['fire_area'].notna()
    fire_df = df[fire_mask].copy()
    
    # 속도 카테고리 생성 (동일한 로직)
    area_data = fire_df['fire_area'].copy()
    q40, q70 = area_data.quantile([0.4, 0.7])
    
    speed_cats = []
    for area in area_data:
        if area <= q40:
            speed_cats.append('slow')
        elif area <= q70:
            speed_cats.append('medium')  
        else:
            speed_cats.append('fast')
    
    fire_df['speed_category'] = speed_cats
    
    print(f"속도 카테고리 분포: {pd.Series(speed_cats).value_counts().to_dict()}")
    print(f"면적 기준: slow <= {q40:.2f}, medium <= {q70:.2f}, fast > {q70:.2f}")
    
    # 피처들
    features = ['t2m_0h', 'rh2m_0h', 'ws10m_0h', 'precip_0h', 'fwi_0h', 
                'ffmc_0h', 'dmc_0h', 'isi_0h', 'elevation_mean', 'slope_mean',
                'fire_area']  # <- 이것이 문제!
    
    # 가용한 피처만 사용
    available_features = [f for f in features if f in fire_df.columns]
    print(f"사용 가능한 피처: {available_features}")
    
    # 데이터 준비
    X = fire_df[available_features].copy()
    y = fire_df['speed_category'].copy()
    
    # 전처리
    for col in X.columns:
        if X[col].isna().sum() > 0:
            X[col] = X[col].fillna(X[col].median())
    
    # 간단한 모델로 피처 중요도 확인
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # 피처 중요도 분석
    importances = rf.feature_importances_
    feature_importance = list(zip(available_features, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🎯 피처 중요도:")
    for feat, imp in feature_importance:
        print(f"   {feat}: {imp:.4f}")
    
    # fire_area와 속도 카테고리의 관계 확인
    print(f"\n🔥 fire_area와 속도 카테고리 관계:")
    area_by_speed = fire_df.groupby('speed_category')['fire_area'].agg(['mean', 'median', 'min', 'max'])
    print(area_by_speed)
    
    # 상관관계 확인
    if 'fire_area' in available_features:
        corr_with_area = X.corrwith(X['fire_area']).abs().sort_values(ascending=False)
        print(f"\n📊 fire_area와의 상관관계:")
        for feat, corr in corr_with_area.items():
            if feat != 'fire_area':
                print(f"   {feat}: {corr:.4f}")
    
    # 시각화
    plt.figure(figsize=(12, 8))
    
    # 피처 중요도
    plt.subplot(2, 2, 1)
    features_plot = [f[:10] for f, _ in feature_importance[:10]]
    importances_plot = [imp for _, imp in feature_importance[:10]]
    plt.barh(features_plot, importances_plot)
    plt.title('피처 중요도')
    plt.xlabel('중요도')
    
    # 면적 분포
    plt.subplot(2, 2, 2)
    fire_df.boxplot(column='fire_area', by='speed_category', ax=plt.gca())
    plt.title('속도별 면적 분포')
    plt.suptitle('')
    
    # 면적 히스토그램
    plt.subplot(2, 2, 3)
    for speed in ['slow', 'medium', 'fast']:
        data = fire_df[fire_df['speed_category'] == speed]['fire_area']
        plt.hist(data, alpha=0.7, label=speed, bins=30)
    plt.xlabel('Fire Area (ha)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('면적별 분포')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('speed_leakage_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ 분석 완료: speed_leakage_analysis.png")
    
    # 누출 확인
    if 'fire_area' in available_features and importances[available_features.index('fire_area')] > 0.5:
        print(f"\n❌ 데이터 누출 발견!")
        print(f"   fire_area가 속도 분류에 {importances[available_features.index('fire_area')]:.1%} 기여")
        print(f"   속도 카테고리가 면적을 직접 사용해서 만들어졌기 때문!")
        
        return True
    
    return False

if __name__ == "__main__":
    has_leakage = load_and_analyze()
    if has_leakage:
        print(f"\n🔧 해결책: fire_area를 피처에서 제거하고 다른 방식으로 속도 정의")