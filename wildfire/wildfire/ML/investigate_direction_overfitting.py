#!/usr/bin/env python3
"""
방향 모델 100% 정확도 원인 조사
- 데이터 누출 검사
- 오버피팅 검증
- 실제 성능 평가
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

def load_direction_model():
    """방향 모델 로드"""
    print("🔍 방향 모델 로딩...")
    
    try:
        model = joblib.load('direction_model_retrained.joblib')
        scaler = joblib.load('direction_scaler_retrained.joblib')
        
        with open('direction_features_retrained.json', 'r') as f:
            features = json.load(f)
            
        with open('direction_retrained_summary.json', 'r') as f:
            summary = json.load(f)
            
        print(f"✅ 모델 로딩 완료")
        print(f"   - 피처 수: {len(features)}")
        print(f"   - 보고된 정확도: {summary['best_accuracy']:.4f}")
        
        return model, scaler, features, summary
        
    except FileNotFoundError as e:
        print(f"❌ 파일을 찾을 수 없습니다: {e}")
        return None, None, None, None

def investigate_data_leakage():
    """데이터 누출 조사"""
    print("🕵️ 데이터 누출 조사...")
    
    # 원본 데이터 로드
    try:
        df = pd.read_csv('final_merged_feature_engineered.csv')
        print(f"✅ 원본 데이터: {df.shape}")
        
        # 방향 관련 컬럼 확인
        direction_cols = [col for col in df.columns if 'direction' in col.lower()]
        wind_direction_cols = [col for col in df.columns if 'wd10m' in col.lower()]
        
        print(f"\n📊 방향 관련 컬럼 분석:")
        print(f"   - 'direction' 포함 컬럼: {len(direction_cols)}개")
        for col in direction_cols[:10]:  # 상위 10개만
            print(f"     • {col}")
            
        print(f"   - 바람 방향 컬럼: {len(wind_direction_cols)}개")
        for col in wind_direction_cols[:10]:  # 상위 10개만
            print(f"     • {col}")
        
        # 타겟 변수 확인
        if 'fire_spread_direction' in df.columns:
            target = df['fire_spread_direction']
            print(f"\n🎯 타겟 변수 분석:")
            print(f"   - 결측값: {target.isna().sum()}개")
            print(f"   - 유니크 값: {target.nunique()}개")
            print(f"   - 값 분포:\n{target.value_counts()}")
            
            return df, target
        else:
            print("❌ 타겟 변수 'fire_spread_direction'를 찾을 수 없습니다.")
            return df, None
            
    except FileNotFoundError:
        print("❌ 원본 데이터 파일을 찾을 수 없습니다.")
        return None, None

def check_perfect_correlation(df, target, features):
    """피처와 타겟의 완벽한 상관관계 확인"""
    print("🔍 완벽한 상관관계 검사...")
    
    if target is None or df is None:
        print("❌ 데이터가 없어 검사할 수 없습니다.")
        return
    
    # 타겟이 있는 데이터만 사용
    valid_data = df.dropna(subset=['fire_spread_direction'])
    print(f"   유효한 데이터: {len(valid_data)}개")
    
    suspicious_features = []
    
    for feature in features:
        if feature in df.columns:
            try:
                # 타겟과 피처의 관계 분석
                feature_data = valid_data[feature]
                target_data = valid_data['fire_spread_direction']
                
                # 범주형 피처인 경우
                if feature_data.dtype == 'object' or feature_data.nunique() < 20:
                    # 각 피처값에 대해 타겟의 분포 확인
                    cross_tab = pd.crosstab(feature_data, target_data)
                    
                    # 각 피처값이 하나의 타겟값만 가지는지 확인
                    perfect_mapping = True
                    for idx in cross_tab.index:
                        non_zero_targets = (cross_tab.loc[idx] > 0).sum()
                        if non_zero_targets > 1:
                            perfect_mapping = False
                            break
                    
                    if perfect_mapping:
                        suspicious_features.append({
                            'feature': feature,
                            'type': '완벽한 매핑',
                            'unique_values': feature_data.nunique(),
                            'sample_mapping': cross_tab.head()
                        })
                
                # 연속형 피처인 경우 (바람 방향 등)
                elif 'wd10m' in feature or 'direction' in feature.lower():
                    # 바람 방향과 화재 확산 방향의 관계
                    corr_data = pd.DataFrame({
                        'feature': feature_data,
                        'target': target_data.astype('category').cat.codes
                    }).dropna()
                    
                    if len(corr_data) > 10:
                        correlation = corr_data.corr().iloc[0, 1]
                        if abs(correlation) > 0.9:
                            suspicious_features.append({
                                'feature': feature,
                                'type': '높은 상관관계',
                                'correlation': correlation,
                                'unique_values': feature_data.nunique()
                            })
                        
            except Exception as e:
                continue
    
    print(f"\n⚠️ 의심스러운 피처: {len(suspicious_features)}개")
    for i, sf in enumerate(suspicious_features):
        print(f"   {i+1}. {sf['feature']}: {sf['type']}")
        if 'correlation' in sf:
            print(f"      상관관계: {sf['correlation']:.4f}")
        print(f"      유니크 값: {sf['unique_values']}개")
    
    return suspicious_features

def perform_proper_validation(df, target, features, model, scaler):
    """적절한 검증 수행"""
    print("✅ 적절한 검증 수행...")
    
    if target is None or df is None:
        print("❌ 데이터가 없어 검증할 수 없습니다.")
        return
    
    # 유효한 데이터만 사용
    valid_data = df.dropna(subset=['fire_spread_direction']).copy()
    
    # 피처 준비
    available_features = [f for f in features if f in valid_data.columns]
    X = valid_data[available_features].copy()
    y = valid_data['fire_spread_direction'].copy()
    
    print(f"   사용 가능한 피처: {len(available_features)}개")
    print(f"   데이터 크기: {X.shape}")
    print(f"   클래스 분포:\n{y.value_counts()}")
    
    # 결측치 처리
    X_processed = X.copy()
    for col in X_processed.columns:
        if X_processed[col].isna().sum() > 0:
            if X_processed[col].dtype == 'object':
                X_processed[col] = X_processed[col].fillna('unknown')
            else:
                X_processed[col] = X_processed[col].fillna(X_processed[col].median())
    
    # 범주형 변수 인코딩 (간단히)
    for col in X_processed.columns:
        if X_processed[col].dtype == 'object':
            X_processed[col] = X_processed[col].astype('category').cat.codes
    
    # 무한값 처리
    X_processed = X_processed.replace([np.inf, -np.inf], np.nan)
    for col in X_processed.columns:
        if X_processed[col].isna().sum() > 0:
            X_processed[col] = X_processed[col].fillna(0)
    
    print(f"\n🔄 교차 검증 수행...")
    
    try:
        # StratifiedKFold로 교차 검증
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_processed, y, cv=cv, scoring='accuracy')
        
        print(f"   교차 검증 점수: {cv_scores}")
        print(f"   평균 정확도: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # Train-Test Split 검증
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # 스케일링
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 새로운 훈련 및 예측
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"   홀드아웃 정확도: {test_accuracy:.4f}")
        
        # 상세 리포트
        print(f"\n📋 분류 리포트:")
        print(classification_report(y_test, y_pred))
        
        # 혼동행렬 시각화
        create_realistic_confusion_matrix(y_test, y_pred)
        
        return {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'holdout_accuracy': test_accuracy,
            'is_realistic': test_accuracy < 0.95  # 95% 이하면 현실적
        }
        
    except Exception as e:
        print(f"❌ 검증 중 오류: {e}")
        return None

def create_realistic_confusion_matrix(y_true, y_pred):
    """현실적인 혼동행렬 생성"""
    print("📊 현실적인 혼동행렬 생성...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(y_true.unique()),
                yticklabels=sorted(y_true.unique()))
    
    plt.title(f'현실적인 방향 모델 혼동행렬\n정확도: {accuracy_score(y_true, y_pred):.4f}')
    plt.xlabel('예측')
    plt.ylabel('실제')
    
    plt.tight_layout()
    plt.savefig('realistic_direction_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 시각화 저장: realistic_direction_confusion_matrix.png")

def analyze_feature_importance(model, features):
    """피처 중요도 분석"""
    print("📈 피처 중요도 분석...")
    
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # 중요도 정렬
            feature_importance = list(zip(features, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n🔝 상위 10개 중요한 피처:")
            for i, (feature, importance) in enumerate(feature_importance[:10]):
                print(f"   {i+1:2d}. {feature:30s}: {importance:.4f}")
            
            # 의심스럽게 높은 중요도 체크
            max_importance = max(importances)
            if max_importance > 0.5:
                print(f"\n⚠️ 매우 높은 중요도 피처 발견: {max_importance:.4f}")
                print(f"   이는 데이터 누출을 의심해야 합니다.")
            
            return feature_importance
        else:
            print("   모델에 feature_importances_ 속성이 없습니다.")
            return None
            
    except Exception as e:
        print(f"❌ 피처 중요도 분석 실패: {e}")
        return None

def main():
    """메인 함수"""
    print("🚨 방향 모델 100% 정확도 조사 시작")
    print("=" * 60)
    
    # 1. 모델 로드
    model, scaler, features, summary = load_direction_model()
    if model is None:
        return
    
    # 2. 데이터 누출 조사
    df, target = investigate_data_leakage()
    
    # 3. 완벽한 상관관계 확인
    suspicious_features = check_perfect_correlation(df, target, features)
    
    # 4. 피처 중요도 분석
    feature_importance = analyze_feature_importance(model, features)
    
    # 5. 적절한 검증 수행
    validation_results = perform_proper_validation(df, target, features, model, scaler)
    
    # 6. 최종 결론
    print("\n" + "=" * 60)
    print("🔍 조사 결과 요약")
    print("=" * 60)
    
    print(f"📊 의심스러운 피처: {len(suspicious_features) if suspicious_features else 0}개")
    
    if validation_results:
        print(f"📈 실제 교차검증 성능: {validation_results['cv_mean']:.4f} (±{validation_results['cv_std']:.4f})")
        print(f"📈 실제 홀드아웃 성능: {validation_results['holdout_accuracy']:.4f}")
        
        if validation_results['is_realistic']:
            print("✅ 성능이 현실적인 범위입니다.")
        else:
            print("⚠️ 여전히 의심스럽게 높은 성능입니다.")
    
    # 최종 판정
    if suspicious_features and len(suspicious_features) > 0:
        print("\n🚨 결론: 데이터 누출 가능성이 높습니다!")
        print("   - 의심스러운 피처들을 제거하고 모델을 재학습해야 합니다.")
    elif validation_results and validation_results['holdout_accuracy'] > 0.95:
        print("\n🚨 결론: 오버피팅 가능성이 높습니다!")
        print("   - 더 엄격한 정규화나 더 많은 데이터가 필요합니다.")
    else:
        print("\n✅ 결론: 모델이 실제로 좋은 성능을 보입니다.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()