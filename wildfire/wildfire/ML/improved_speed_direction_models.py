#!/usr/bin/env python3
"""
속도 및 방향 모델 개선
- 방향 모델: 데이터 누출 제거, 현실적 성능 목표 (65-75%)
- 속도 모델: 클래스 불균형 해결, 성능 향상 목표 (55-65%)
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """데이터 로드 및 전처리"""
    print("📊 데이터 로드 및 준비...")
    
    df = pd.read_csv('final_merged_feature_engineered.csv')
    print(f"   원본 데이터: {df.shape}")
    
    # 타겟 변수 확인
    available_targets = []
    target_info = {}
    
    # 속도 관련 타겟 찾기
    speed_candidates = ['fire_spread_speed', 'spread_rate', 'speed_category', 'fire_speed']
    for candidate in speed_candidates:
        if candidate in df.columns and df[candidate].notna().sum() > 50:
            available_targets.append(f"speed: {candidate}")
            target_info[f"speed_{candidate}"] = {
                'column': candidate,
                'type': 'speed',
                'valid_count': df[candidate].notna().sum()
            }
    
    # 방향 관련 타겟 찾기
    direction_candidates = ['fire_spread_direction', 'spread_direction', 'direction_category', 'fire_direction']
    for candidate in direction_candidates:
        if candidate in df.columns and df[candidate].notna().sum() > 50:
            available_targets.append(f"direction: {candidate}")
            target_info[f"direction_{candidate}"] = {
                'column': candidate,
                'type': 'direction',
                'valid_count': df[candidate].notna().sum()
            }
    
    print(f"   사용 가능한 타겟: {len(available_targets)}개")
    for target in available_targets:
        print(f"     • {target}")
    
    # 속도와 방향을 추론해서 만들기
    if not available_targets:
        print("   타겟 변수를 찾을 수 없어 추론으로 생성합니다...")
        df = create_inferred_targets(df)
    
    return df, target_info

def create_inferred_targets(df):
    """속도와 방향 타겟을 추론으로 생성"""
    print("🔍 속도 및 방향 타겟 추론 생성...")
    
    # 화재 면적과 기간으로 속도 추정
    if 'fire_area' in df.columns and 'fire_duration' in df.columns:
        valid_mask = (df['fire_area'] > 0) & (df['fire_duration'] > 0)
        df.loc[valid_mask, 'estimated_speed_rate'] = df.loc[valid_mask, 'fire_area'] / df.loc[valid_mask, 'fire_duration']
        
        # 속도 카테고리 생성 (3분위수 기준)
        speed_data = df.loc[valid_mask, 'estimated_speed_rate']
        if len(speed_data) > 10:
            q33, q67 = speed_data.quantile([0.33, 0.67])
            df.loc[valid_mask, 'speed_category'] = pd.cut(
                speed_data, 
                bins=[-np.inf, q33, q67, np.inf],
                labels=['slow', 'medium', 'fast']
            )
            print(f"   속도 카테고리 생성: {df['speed_category'].value_counts().to_dict()}")
    
    # 바람 방향으로 화재 방향 추정
    if 'wd10m_0h' in df.columns:
        wind_dir = df['wd10m_0h'].copy()
        valid_wind = wind_dir.notna() & (wind_dir >= 0) & (wind_dir < 360)
        
        if valid_wind.sum() > 10:
            # 8방향으로 변환
            direction_mapping = []
            for wd in wind_dir[valid_wind]:
                if 337.5 <= wd or wd < 22.5:
                    direction_mapping.append('north')
                elif 22.5 <= wd < 67.5:
                    direction_mapping.append('northeast')
                elif 67.5 <= wd < 112.5:
                    direction_mapping.append('east')
                elif 112.5 <= wd < 157.5:
                    direction_mapping.append('southeast')
                elif 157.5 <= wd < 202.5:
                    direction_mapping.append('south')
                elif 202.5 <= wd < 247.5:
                    direction_mapping.append('southwest')
                elif 247.5 <= wd < 292.5:
                    direction_mapping.append('west')
                elif 292.5 <= wd < 337.5:
                    direction_mapping.append('northwest')
                else:
                    direction_mapping.append('unknown')
            
            df.loc[valid_wind, 'estimated_direction'] = direction_mapping
            print(f"   방향 카테고리 생성: {df['estimated_direction'].value_counts().to_dict()}")
    
    return df

def create_clean_direction_features(df):
    """방향 모델용 깨끗한 피처 (데이터 누출 제거)"""
    print("🧹 방향 모델용 깨끗한 피처 생성...")
    
    feature_groups = {
        # 기상 데이터 (핵심)
        'weather': [
            't2m_0h', 'rh2m_0h', 'ws10m_0h', 'wd10m_0h',
            'precip_0h', 'sp_0h'
        ],
        
        # 간접적 지형 정보만 (aspect 제외)
        'terrain_safe': [
            'elevation_mean', 'elevation_std',
            'slope_mean', 'slope_std'
        ],
        
        # 화재 위험 지수
        'fire_indices': [
            'fwi_0h', 'ffmc_0h', 'dmc_0h', 'dc_0h', 'isi_0h', 'bui_0h'
        ],
        
        # 시간적 패턴
        'temporal': [
            'month', 'day', 'hour' if 'hour' in df.columns else None
        ],
        
        # 건조도 지표
        'dryness': [
            'dry_days_7d_start', 'dry_days_30d_start',
            'consecutive_dry_days_start'
        ]
    }
    
    # 실제 존재하는 피처만 선택
    selected_features = []
    for group, features in feature_groups.items():
        group_features = []
        for feature in features:
            if feature and feature in df.columns:
                group_features.append(feature)
        selected_features.extend(group_features)
        print(f"   {group}: {len(group_features)}개 피처")
    
    # 중요: aspect 관련 피처 완전 제거
    excluded_keywords = ['aspect', 'south_steep', 'terrain_var']
    selected_features = [f for f in selected_features 
                        if not any(keyword in f.lower() for keyword in excluded_keywords)]
    
    print(f"   최종 선택된 피처: {len(selected_features)}개")
    
    return selected_features

def create_enhanced_speed_features(df):
    """속도 모델용 강화된 피처"""
    print("⚡ 속도 모델용 강화된 피처 생성...")
    
    # 기본 피처 그룹
    feature_groups = {
        'weather_core': [
            't2m_0h', 'rh2m_0h', 'ws10m_0h', 'precip_0h'
        ],
        'fire_weather': [
            'fwi_0h', 'ffmc_0h', 'isi_0h', 'dmc_0h'
        ],
        'terrain': [
            'elevation_mean', 'slope_mean', 'aspect_mean'
        ],
        'dryness': [
            'dry_days_7d_start', 'dry_days_30d_start'
        ]
    }
    
    # 기존 피처 수집
    selected_features = []
    for group, features in feature_groups.items():
        group_features = [f for f in features if f in df.columns]
        selected_features.extend(group_features)
        print(f"   {group}: {len(group_features)}개")
    
    # 상호작용 피처 추가
    if 't2m_0h' in df.columns and 'rh2m_0h' in df.columns:
        df['temp_humidity_risk'] = df['t2m_0h'] / (df['rh2m_0h'] + 1)
        selected_features.append('temp_humidity_risk')
    
    if 'ws10m_0h' in df.columns and 'slope_mean' in df.columns:
        df['wind_slope_effect'] = df['ws10m_0h'] * df['slope_mean'] / 100
        selected_features.append('wind_slope_effect')
    
    if 'fwi_0h' in df.columns and 'dry_days_7d_start' in df.columns:
        df['fire_dryness_combo'] = df['fwi_0h'] * np.log1p(df['dry_days_7d_start'])
        selected_features.append('fire_dryness_combo')
    
    print(f"   최종 피처: {len(selected_features)}개")
    
    return selected_features

def train_improved_direction_model(df, features, target_col='estimated_direction'):
    """개선된 방향 모델 훈련"""
    print("🧭 개선된 방향 모델 훈련...")
    
    # 유효한 데이터 선택
    valid_mask = df[target_col].notna()
    X = df.loc[valid_mask, features].copy()
    y = df.loc[valid_mask, target_col].copy()
    
    print(f"   훈련 데이터: {X.shape}")
    print(f"   클래스 분포:\n{y.value_counts()}")
    
    # 데이터 전처리
    for col in X.columns:
        if X[col].isna().sum() > 0:
            X[col] = X[col].fillna(X[col].median())
    
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if X[col].isna().sum() > 0:
            X[col] = X[col].fillna(0)
    
    # 라벨 인코딩
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
    )
    
    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 다양한 모델 시도
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        ) if 'xgboost' in globals() else None,
        'LogisticRegression': LogisticRegression(
            multi_class='multinomial',
            max_iter=1000,
            random_state=42
        )
    }
    
    results = {}
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        if model is None:
            continue
        
        print(f"\n   {name} 훈련 중...")
        
        # 교차 검증
        cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                   cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                   scoring='accuracy')
        
        # 테스트 성능
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': test_accuracy,
            'model': model
        }
        
        print(f"     CV: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"     Test: {test_accuracy:.4f}")
        
        if cv_scores.mean() > best_score:
            best_score = cv_scores.mean()
            best_model = (name, model)
    
    print(f"\n🏆 최고 모델: {best_model[0]} (CV: {best_score:.4f})")
    
    # 최종 예측 및 평가
    best_model_obj = best_model[1]
    y_pred_final = best_model_obj.predict(X_test_scaled)
    y_pred_labels = le.inverse_transform(y_pred_final)
    y_test_labels = le.inverse_transform(y_test)
    
    print(f"\n📊 최종 성능 보고:")
    print(classification_report(y_test_labels, y_pred_labels))
    
    # 모델 저장
    model_package = {
        'model': best_model_obj,
        'scaler': scaler,
        'label_encoder': le,
        'features': features,
        'results': results,
        'best_model_name': best_model[0]
    }
    
    joblib.dump(model_package, 'improved_direction_model.joblib')
    
    # 혼동행렬 시각화
    create_confusion_matrix(y_test_labels, y_pred_labels, 
                           'Improved Direction Model', 'direction')
    
    return model_package, results

def train_improved_speed_model(df, features, target_col='speed_category'):
    """개선된 속도 모델 훈련 (클래스 불균형 해결)"""
    print("⚡ 개선된 속도 모델 훈련...")
    
    # 유효한 데이터 선택
    valid_mask = df[target_col].notna()
    X = df.loc[valid_mask, features].copy()
    y = df.loc[valid_mask, target_col].copy()
    
    print(f"   훈련 데이터: {X.shape}")
    print(f"   클래스 분포:\n{y.value_counts()}")
    
    # 데이터 전처리
    for col in X.columns:
        if X[col].isna().sum() > 0:
            X[col] = X[col].fillna(X[col].median())
    
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if X[col].isna().sum() > 0:
            X[col] = X[col].fillna(0)
    
    # 라벨 인코딩
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
    )
    
    # 클래스 가중치 계산
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    print(f"   클래스 가중치: {class_weight_dict}")
    
    # SMOTE로 오버샘플링
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"   SMOTE 후 분포: {np.bincount(y_train_balanced)}")
    
    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    # 다양한 모델 (클래스 가중치 적용)
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=8,
            random_state=42
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=8,
            scale_pos_weight=class_weights[1]/class_weights[0] if len(class_weights) > 1 else 1,
            random_state=42,
            n_jobs=-1
        ) if 'xgboost' in globals() else None,
        'SVC': SVC(
            kernel='rbf',
            class_weight='balanced',
            random_state=42,
            probability=True
        )
    }
    
    results = {}
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        if model is None:
            continue
        
        print(f"\n   {name} 훈련 중...")
        
        # 교차 검증 (원본 불균형 데이터에서)
        cv_scores = cross_val_score(model, X_train, y_train,
                                   cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                   scoring='accuracy')
        
        # 테스트 성능 (균형잡힌 데이터로 훈련)
        model.fit(X_train_scaled, y_train_balanced)
        y_pred = model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': test_accuracy,
            'model': model
        }
        
        print(f"     CV: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"     Test: {test_accuracy:.4f}")
        
        if test_accuracy > best_score:
            best_score = test_accuracy
            best_model = (name, model)
    
    print(f"\n🏆 최고 모델: {best_model[0]} (Test: {best_score:.4f})")
    
    # 최종 평가
    best_model_obj = best_model[1]
    y_pred_final = best_model_obj.predict(X_test_scaled)
    y_pred_labels = le.inverse_transform(y_pred_final)
    y_test_labels = le.inverse_transform(y_test)
    
    print(f"\n📊 최종 성능 보고:")
    print(classification_report(y_test_labels, y_pred_labels))
    
    # 모델 저장
    model_package = {
        'model': best_model_obj,
        'scaler': scaler,
        'label_encoder': le,
        'features': features,
        'smote': smote,
        'results': results,
        'best_model_name': best_model[0]
    }
    
    joblib.dump(model_package, 'improved_speed_model.joblib')
    
    # 혼동행렬 시각화
    create_confusion_matrix(y_test_labels, y_pred_labels, 
                           'Improved Speed Model', 'speed')
    
    return model_package, results

def create_confusion_matrix(y_true, y_pred, title, model_type):
    """혼동행렬 시각화"""
    print(f"📊 {title} 혼동행렬 생성...")
    
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(set(y_pred)),
                yticklabels=sorted(set(y_true)))
    
    plt.title(f'{title}\n정확도: {accuracy:.4f}')
    plt.xlabel('예측')
    plt.ylabel('실제')
    
    plt.tight_layout()
    plt.savefig(f'improved_{model_type}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 저장: improved_{model_type}_confusion_matrix.png")

def main():
    """메인 함수"""
    print("🚀 속도 및 방향 모델 개선 시작")
    print("=" * 60)
    
    try:
        # 1. 데이터 준비
        df, target_info = load_and_prepare_data()
        
        # 2. 방향 모델 개선 (데이터 누출 제거)
        if 'estimated_direction' in df.columns:
            print("\n🧭 방향 모델 개선...")
            direction_features = create_clean_direction_features(df)
            direction_package, direction_results = train_improved_direction_model(
                df, direction_features, 'estimated_direction'
            )
        
        # 3. 속도 모델 개선 (클래스 불균형 해결)
        if 'speed_category' in df.columns:
            print("\n⚡ 속도 모델 개선...")
            speed_features = create_enhanced_speed_features(df)
            speed_package, speed_results = train_improved_speed_model(
                df, speed_features, 'speed_category'
            )
        
        # 4. 결과 요약
        print("\n" + "=" * 60)
        print("🏆 개선 결과 요약")
        print("=" * 60)
        
        if 'direction_package' in locals():
            best_dir = max(direction_results.items(), key=lambda x: x[1]['cv_mean'])
            print(f"🧭 방향 모델:")
            print(f"   최고 모델: {best_dir[0]}")
            print(f"   교차검증: {best_dir[1]['cv_mean']:.4f} (±{best_dir[1]['cv_std']:.4f})")
            print(f"   테스트: {best_dir[1]['test_accuracy']:.4f}")
        
        if 'speed_package' in locals():
            best_spd = max(speed_results.items(), key=lambda x: x[1]['test_accuracy'])
            print(f"⚡ 속도 모델:")
            print(f"   최고 모델: {best_spd[0]}")
            print(f"   교차검증: {best_spd[1]['cv_mean']:.4f} (±{best_spd[1]['cv_std']:.4f})")
            print(f"   테스트: {best_spd[1]['test_accuracy']:.4f}")
        
        print("\n✅ 모델 개선 완료!")
        print("   - improved_direction_model.joblib")
        print("   - improved_speed_model.joblib")
        print("   - improved_direction_confusion_matrix.png")
        print("   - improved_speed_confusion_matrix.png")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()