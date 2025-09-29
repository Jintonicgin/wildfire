#!/usr/bin/env python3
"""
면적 예측 대신 면적 크기 분류 접근법
- 회귀 대신 분류로 접근
- 소형/중형/대형 화재로 분류
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

def load_and_analyze_area_data():
    """데이터 로드 및 면적 분포 분석"""
    print("🔥 화재 면적 분포 분석...")
    
    df = pd.read_csv('final_merged_feature_engineered.csv', low_memory=False)
    fire_mask = (df['fire_area'] > 0) & df['fire_area'].notna()
    fire_df = df[fire_mask].copy()
    
    area_data = fire_df['fire_area']
    
    print(f"   화재 데이터: {fire_df.shape}")
    print(f"\\n📊 면적 분포:")
    print(f"   평균: {area_data.mean():.2f} ha")
    print(f"   중앙값: {area_data.median():.2f} ha")
    print(f"   최댓값: {area_data.max():.2f} ha")
    
    # 분위수 확인
    quantiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    for q in quantiles:
        print(f"   {q*100:2.0f}%: {area_data.quantile(q):.2f} ha")
    
    return fire_df

def create_area_size_categories(df, method='adaptive'):
    """면적 크기 카테고리 생성"""
    print(f"\\n🎯 면적 크기 분류 ({method})...")
    
    area_data = df['fire_area']
    
    if method == 'adaptive':
        # 데이터 분포 기반 적응적 분류
        q75 = area_data.quantile(0.75)  # 상위 25%
        q95 = area_data.quantile(0.95)  # 상위 5%
        
        size_cats = []
        for area in area_data:
            if area <= q75:
                size_cats.append('small')      # 75% (소형)
            elif area <= q95:
                size_cats.append('medium')     # 20% (중형)
            else:
                size_cats.append('large')      # 5% (대형)
    
    elif method == 'fire_standard':
        # 화재 관리 표준 기준
        size_cats = []
        for area in area_data:
            if area <= 1.0:
                size_cats.append('small')      # 1ha 이하
            elif area <= 10.0:
                size_cats.append('medium')     # 1-10ha
            else:
                size_cats.append('large')      # 10ha 초과
    
    elif method == 'balanced':
        # 균형잡힌 3분할
        q33 = area_data.quantile(0.33)
        q67 = area_data.quantile(0.67)
        
        size_cats = []
        for area in area_data:
            if area <= q33:
                size_cats.append('small')
            elif area <= q67:
                size_cats.append('medium')
            else:
                size_cats.append('large')
    
    df['area_size_category'] = size_cats
    
    counts = pd.Series(size_cats).value_counts()
    percentages = pd.Series(size_cats).value_counts(normalize=True) * 100
    
    print(f"   분류 결과:")
    for cat in ['small', 'medium', 'large']:
        if cat in counts:
            print(f"   {cat:6}: {counts[cat]:3d}개 ({percentages[cat]:4.1f}%)")
    
    return df

def create_comprehensive_features(df):
    """포괄적 피처 생성"""
    print("\\n🎯 포괄적 피처 생성...")
    
    features = []
    
    # 1. 화재 위험 지수 (전체)
    fire_indices = ['fwi_0h', 'ffmc_0h', 'dmc_0h', 'dc_0h', 'isi_0h', 'bui_0h']
    for feat in fire_indices:
        if feat in df.columns:
            features.append(feat)
    
    # 2. 기상 조건
    weather_core = ['t2m_0h', 'rh2m_0h', 'ws10m_0h', 'wd10m_0h', 'precip_0h']
    for feat in weather_core:
        if feat in df.columns:
            features.append(feat)
    
    # 3. 지형 및 환경
    terrain_env = ['elevation_mean', 'slope_mean', 'ndvi_before']
    for feat in terrain_env:
        if feat in df.columns:
            features.append(feat)
    
    # 4. 시간적 요소
    temporal = ['fire_month', 'startday']
    for feat in temporal:
        if feat in df.columns:
            features.append(feat)
    
    # 5. 건조 조건
    dry_conditions = [
        'dry_days_7d_start', 'dry_days_14d_start', 'dry_days_30d_start',
        'consecutive_dry_days_start'
    ]
    for feat in dry_conditions:
        if feat in df.columns:
            features.append(feat)
    
    # 6. 복합 지수
    combo_indices = [
        'hot_dry_combo', 'dry_windy_combo', 'low_humidity_flag',
        'dry_to_rain_ratio_30d'
    ]
    for feat in combo_indices:
        if feat in df.columns:
            features.append(feat)
    
    # 7. 12시간 평균 및 최댓값
    avg_max_features = [
        'fwi_mean_0_12h', 'isi_mean_0_12h', 'max_temp_0_12h', 
        'max_wind_0_12h', 'min_humidity_0_12h'
    ]
    for feat in avg_max_features:
        if feat in df.columns:
            features.append(feat)
    
    # 8. 과거 기상 패턴 (선별적)
    past_hours = [3, 6, 12, 24]
    past_vars = ['t2m', 'rh2m', 'ws10m']
    
    for hour in past_hours:
        for var in past_vars:
            feat = f'{var}_{hour}h_past'
            if feat in df.columns:
                features.append(feat)
    
    # 9. 고급 파생 변수 생성
    derived_features = []
    
    # FWI 위험 등급
    if 'fwi_0h' in df.columns:
        df['fwi_risk_high'] = (df['fwi_0h'] > 21).astype(int)
        derived_features.append('fwi_risk_high')
    
    # 극한 기상 조건
    if 'ws10m_0h' in df.columns:
        df['extreme_wind'] = (df['ws10m_0h'] > 30).astype(int)
        derived_features.append('extreme_wind')
    
    if 'rh2m_0h' in df.columns:
        df['very_dry'] = (df['rh2m_0h'] < 20).astype(int)
        derived_features.append('very_dry')
    
    if 't2m_0h' in df.columns:
        df['very_hot'] = (df['t2m_0h'] > 35).astype(int)
        derived_features.append('very_hot')
    
    # 복합 위험 지수
    if all(col in df.columns for col in ['ws10m_0h', 'rh2m_0h']):
        df['wind_dry_risk'] = df['ws10m_0h'] * (100 - df['rh2m_0h'].fillna(50))
        derived_features.append('wind_dry_risk')
    
    features.extend(derived_features)
    
    # 사용 가능한 피처만 필터링
    available_features = [f for f in features if f in df.columns]
    
    print(f"   총 피처: {len(available_features)}개")
    print(f"   화재지수: {len([f for f in fire_indices if f in available_features])}개")
    print(f"   기상: {len([f for f in weather_core if f in available_features])}개")
    print(f"   파생변수: {len(derived_features)}개")
    
    return available_features

def train_classification_models(X_train, X_test, y_train, y_test):
    """분류 모델들 훈련"""
    print("\\n🤖 분류 모델 훈련...")
    
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=3,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=42,
            verbose=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    }
    
    results = {}
    best_model = None
    best_accuracy = 0
    
    for name, model in models.items():
        print(f"   {name} 훈련...")
        
        try:
            # 교차검증
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # 훈련 및 예측
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # 평가
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': accuracy,
                'model': model,
                'predictions': y_pred
            }
            
            print(f"     CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"     Test: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = (name, model, y_pred)
                
        except Exception as e:
            print(f"     {name} 실패: {e}")
    
    return results, best_model

def create_confusion_matrix_plot(y_true, y_pred, title, filename):
    """혼동행렬 시각화"""
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    
    labels = sorted(set(y_true) | set(y_pred))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    
    plt.title(f'{title}\\n정확도: {accuracy:.4f}')
    plt.xlabel('예측')
    plt.ylabel('실제')
    
    plt.tight_layout()
    plt.savefig(f'{filename}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 저장: {filename}_confusion_matrix.png")

def test_different_approaches(fire_df, features):
    """다양한 분류 방법 테스트"""
    print("\\n📊 다양한 분류 방법 비교...")
    
    approaches = ['adaptive', 'fire_standard', 'balanced']
    all_results = {}
    
    for approach in approaches:
        print(f"\\n=== {approach} 방법 ===")
        
        # 분류 생성
        df_copy = fire_df.copy()
        df_copy = create_area_size_categories(df_copy, method=approach)
        
        # 데이터 준비
        X = df_copy[features].copy()
        y = df_copy['area_size_category'].copy()
        
        # 전처리
        for col in X.columns:
            if X[col].isna().sum() > 0:
                X[col] = X[col].fillna(X[col].median())
        
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 모델 훈련
        results, best_model = train_classification_models(
            X_train_scaled, X_test_scaled, y_train, y_test
        )
        
        if best_model:
            best_name, model, y_pred = best_model
            
            # 결과 저장
            all_results[approach] = {
                'best_model': best_name,
                'accuracy': accuracy_score(y_test, y_pred),
                'results': results,
                'model_package': {
                    'model': model,
                    'scaler': scaler,
                    'features': features,
                    'approach': approach
                }
            }
            
            # 혼동행렬
            create_confusion_matrix_plot(
                y_test, y_pred, 
                f'{approach.title()} Area Classification',
                f'area_classification_{approach}'
            )
            
            # 분류 보고서
            print(f"\\n📊 분류 보고서 ({approach}):")
            print(classification_report(y_test, y_pred))
    
    return all_results

def main():
    """메인"""
    print("🎯 화재 면적 분류 접근법")
    print("=" * 50)
    
    # 데이터 분석
    fire_df = load_and_analyze_area_data()
    
    # 피처 생성
    features = create_comprehensive_features(fire_df)
    
    # 다양한 분류 방법 테스트
    all_results = test_different_approaches(fire_df, features)
    
    # 최종 결과 비교
    print("\\n" + "=" * 50)
    print("🏆 분류 방법별 최종 결과")
    print("=" * 50)
    
    for approach, result in all_results.items():
        print(f"{approach:12} | {result['best_model']:15} | 정확도: {result['accuracy']:.4f}")
    
    # 최고 성능 모델
    if all_results:
        best_approach = max(all_results.items(), key=lambda x: x[1]['accuracy'])
        approach_name, result = best_approach
        
        print(f"\\n🥇 최고 성능: {approach_name} 방법")
        print(f"   모델: {result['best_model']}")
        print(f"   정확도: {result['accuracy']:.4f}")
        
        # 모델 저장
        joblib.dump(result['model_package'], 'area_classification_best_model.joblib')
        print(f"\\n✅ 저장 완료: area_classification_best_model.joblib")
        
        # 성능 해석
        accuracy = result['accuracy']
        if accuracy > 0.7:
            print(f"\\n🎉 우수한 성능! 화재 크기 예측에 실용적으로 활용 가능")
        elif accuracy > 0.5:
            print(f"\\n📊 적절한 성능. 참고 정보로 활용 가능")
        else:
            print(f"\\n⚠️ 성능 부족. 추가 개선 필요")

if __name__ == "__main__":
    main()