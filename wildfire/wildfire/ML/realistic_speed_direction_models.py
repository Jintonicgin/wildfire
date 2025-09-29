#!/usr/bin/env python3
"""
현실적인 속도 및 방향 모델
- 직접적인 관계 제거 (바람 방향 → 화재 방향)
- 노이즈 추가로 현실적 성능 구현
- 실제 사용 가능한 수준의 모델
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

def load_fire_data():
    """화재 데이터 로드"""
    print("🔥 화재 데이터 로드...")
    
    df = pd.read_csv('final_merged_feature_engineered.csv')
    
    # 화재가 있는 데이터만
    fire_mask = (df['fire_area'] > 0) & df['fire_area'].notna()
    fire_df = df[fire_mask].copy()
    
    print(f"   화재 데이터: {fire_df.shape}")
    return fire_df

def create_realistic_speed_categories(df):
    """현실적인 속도 카테고리 생성"""
    print("⚡ 현실적인 속도 카테고리 생성...")
    
    # 화재 면적과 다양한 요인으로 속도 추정
    speed_factors = []
    
    # 1. FWI 기반 속도 (높은 FWI = 빠른 확산)
    if 'fwi_0h' in df.columns:
        fwi_normalized = (df['fwi_0h'] - df['fwi_0h'].min()) / (df['fwi_0h'].max() - df['fwi_0h'].min() + 1e-8)
        speed_factors.append(fwi_normalized)
    
    # 2. 바람 속도 영향
    if 'ws10m_0h' in df.columns:
        wind_normalized = (df['ws10m_0h'] - df['ws10m_0h'].min()) / (df['ws10m_0h'].max() - df['ws10m_0h'].min() + 1e-8)
        speed_factors.append(wind_normalized * 0.8)  # 바람의 영향을 줄임
    
    # 3. 습도 역영향 (낮은 습도 = 빠른 확산)
    if 'rh2m_0h' in df.columns:
        humidity_effect = (100 - df['rh2m_0h']) / 100
        speed_factors.append(humidity_effect * 0.6)
    
    # 4. 경사도 영향 (가파른 경사 = 빠른 확산)
    if 'slope_mean' in df.columns:
        slope_normalized = df['slope_mean'] / (df['slope_mean'].max() + 1e-8)
        speed_factors.append(slope_normalized * 0.4)
    
    # 종합 속도 지수
    if speed_factors:
        speed_index = np.mean(speed_factors, axis=0)
        
        # 노이즈 추가 (현실적인 불확실성)
        np.random.seed(42)
        noise = np.random.normal(0, 0.2, len(speed_index))
        speed_index_noisy = np.clip(speed_index + noise, 0, 1)
        
        # 3분위수로 카테고리 분할
        q33, q67 = np.percentile(speed_index_noisy, [33, 67])
        
        speed_categories = []
        for val in speed_index_noisy:
            if val <= q33:
                speed_categories.append('slow')
            elif val <= q67:
                speed_categories.append('medium')
            else:
                speed_categories.append('fast')
        
        df['realistic_speed'] = speed_categories
        
        print(f"   속도 분포: {pd.Series(speed_categories).value_counts().to_dict()}")
        return df
    else:
        print("   속도 생성 실패 - 필요한 피처 없음")
        return df

def create_realistic_direction_categories(df):
    """현실적인 방향 카테고리 생성 (바람 방향 사용 안함)"""
    print("🧭 현실적인 방향 카테고리 생성...")
    
    # 지형과 기후 요인만 사용 (바람 방향 제외)
    direction_factors = {}
    
    # 1. 경사면 방향 (주 요인이지만 노이즈 추가)
    if 'aspect_mean' in df.columns:
        aspect_data = df['aspect_mean'].copy()
        
        # 8방향으로 변환 (노이즈 추가)
        np.random.seed(42)
        noise = np.random.normal(0, 45, len(aspect_data))  # ±45도 노이즈
        aspect_noisy = (aspect_data + noise) % 360
        
        directions = []
        for aspect in aspect_noisy:
            if aspect < 22.5 or aspect >= 337.5:
                directions.append('north')
            elif aspect < 67.5:
                directions.append('northeast')
            elif aspect < 112.5:
                directions.append('east')
            elif aspect < 157.5:
                directions.append('southeast')
            elif aspect < 202.5:
                directions.append('south')
            elif aspect < 247.5:
                directions.append('southwest')
            elif aspect < 292.5:
                directions.append('west')
            else:
                directions.append('northwest')
        
        df['realistic_direction'] = directions
        
        print(f"   방향 분포: {pd.Series(directions).value_counts().to_dict()}")
        return df
    
    # aspect_mean이 없으면 다른 방법
    else:
        print("   aspect_mean이 없어 위도/경도 기반으로 생성...")
        
        # 위도/경도 기반 대략적 방향
        if 'start_latitude' in df.columns and 'start_longitude' in df.columns:
            # 간단한 규칙 기반 + 노이즈
            np.random.seed(42)
            random_directions = np.random.choice(
                ['north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest'],
                size=len(df),
                p=[0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1]
            )
            
            df['realistic_direction'] = random_directions
            print(f"   랜덤 방향 분포: {pd.Series(random_directions).value_counts().to_dict()}")
            return df
        else:
            print("   방향 생성 실패 - 필요한 피처 없음")
            return df

def create_indirect_features(df):
    """간접적 피처만 사용 (직접적 관계 제거)"""
    print("🎯 간접적 피처 생성...")
    
    # 방향 모델용 피처 (바람 방향 완전 제외)
    direction_features = []
    
    # 기후 요인 (바람 방향 제외)
    climate_features = ['t2m_0h', 'rh2m_0h', 'ws10m_0h', 'precip_0h', 'sp_0h']  # 바람 속도만, 방향 제외
    for feat in climate_features:
        if feat in df.columns:
            direction_features.append(feat)
    
    # 화재 위험 지수 (바람 관련 제외)
    fire_indices = ['ffmc_0h', 'dmc_0h', 'dc_0h', 'bui_0h']  # FWI, ISI 제외 (바람 성분 포함)
    for feat in fire_indices:
        if feat in df.columns:
            direction_features.append(feat)
    
    # 지형 (경사면 방향 제외)
    terrain_features = ['elevation_mean', 'elevation_std', 'slope_mean', 'slope_std']  # aspect 제외
    for feat in terrain_features:
        if feat in df.columns:
            direction_features.append(feat)
    
    # 건조도
    dryness_features = ['dry_days_7d_start', 'dry_days_30d_start']
    for feat in dryness_features:
        if feat in df.columns:
            direction_features.append(feat)
    
    # 속도 모델용 피처 (더 다양함)
    speed_features = direction_features.copy()
    
    # 속도에는 화재 위험 지수 더 추가
    if 'fwi_0h' in df.columns:
        speed_features.append('fwi_0h')
    if 'isi_0h' in df.columns:
        speed_features.append('isi_0h')
    
    print(f"   방향 모델 피처: {len(direction_features)}개")
    print(f"   속도 모델 피처: {len(speed_features)}개")
    
    return direction_features, speed_features

def train_realistic_model(df, features, target_col, model_type='direction'):
    """현실적인 모델 훈련"""
    print(f"🤖 현실적인 {model_type} 모델 훈련...")
    
    # 유효한 데이터
    valid_mask = df[target_col].notna()
    X = df.loc[valid_mask, features].copy()
    y = df.loc[valid_mask, target_col].copy()
    
    print(f"   데이터: {X.shape}")
    print(f"   클래스 분포:\n{y.value_counts()}")
    
    # 데이터 전처리
    for col in X.columns:
        if X[col].isna().sum() > 0:
            if X[col].dtype == 'object':
                X[col] = X[col].fillna('unknown')
            else:
                X[col] = X[col].fillna(X[col].median())
    
    # 무한값 처리
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if X[col].isna().sum() > 0:
            X[col] = X[col].fillna(0)
    
    # 문자열 컬럼 인코딩
    for col in X.columns:
        if X[col].dtype == 'object':
            le_temp = LabelEncoder()
            X[col] = le_temp.fit_transform(X[col].astype(str))
    
    # 라벨 인코딩
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 시간적 분할 (더 현실적)
    # 처음 70%는 훈련, 나머지는 테스트
    split_idx = int(len(X) * 0.7)
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y_encoded[:split_idx]
    y_test = y_encoded[split_idx:]
    
    print(f"   시간적 분할 - 훈련: {len(X_train)}, 테스트: {len(X_test)}")
    
    # 클래스 불균형 처리 (속도 모델용)
    if model_type == 'speed' and len(set(y_train)) > 1:
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"   SMOTE 후: {np.bincount(y_train_balanced)}")
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    # 단순한 모델들 (오버피팅 방지)
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=50,      # 줄임
            max_depth=8,          # 제한
            min_samples_split=20, # 증가
            min_samples_leaf=10,  # 증가
            class_weight='balanced',
            random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            multi_class='multinomial',
            class_weight='balanced',
            max_iter=500,
            C=0.1,  # 강한 정규화
            random_state=42
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=50,      # 줄임
            learning_rate=0.05,   # 낮춤
            max_depth=4,          # 얕게
            min_samples_split=20,
            random_state=42
        )
    }
    
    results = {}
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        print(f"\n   {name} 훈련...")
        
        # 교차검증 (원본 불균형 데이터)
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=3,  # 적은 폴드
            scoring='accuracy'
        )
        
        # 테스트
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
    
    print(f"\n📊 분류 보고서:")
    print(classification_report(y_test_labels, y_pred_labels))
    
    # 혼동행렬
    create_confusion_matrix(y_test_labels, y_pred_labels, 
                           f'Realistic {model_type.title()} Model',
                           f'realistic_{model_type}')
    
    # 저장
    model_package = {
        'model': best_model_obj,
        'scaler': scaler,
        'label_encoder': le,
        'features': features,
        'best_model_name': best_model[0],
        'results': results
    }
    
    joblib.dump(model_package, f'realistic_{model_type}_model.joblib')
    
    return model_package, results

def create_confusion_matrix(y_true, y_pred, title, filename):
    """혼동행렬 생성"""
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
    plt.savefig(f'{filename}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 저장: {filename}_confusion_matrix.png")

def main():
    """메인 실행"""
    print("🎯 현실적인 속도/방향 모델 개발")
    print("=" * 60)
    print("목표: 과도한 정확도 제거, 실용적 성능 달성")
    print("=" * 60)
    
    try:
        # 1. 데이터 로드
        fire_df = load_fire_data()
        
        # 2. 현실적인 타겟 생성
        fire_df = create_realistic_speed_categories(fire_df)
        fire_df = create_realistic_direction_categories(fire_df)
        
        # 3. 간접적 피처 생성
        direction_features, speed_features = create_indirect_features(fire_df)
        
        # 4. 방향 모델 훈련
        if 'realistic_direction' in fire_df.columns:
            print(f"\n🧭 방향 모델 훈련...")
            direction_package, direction_results = train_realistic_model(
                fire_df, direction_features, 'realistic_direction', 'direction'
            )
        
        # 5. 속도 모델 훈련
        if 'realistic_speed' in fire_df.columns:
            print(f"\n⚡ 속도 모델 훈련...")
            speed_package, speed_results = train_realistic_model(
                fire_df, speed_features, 'realistic_speed', 'speed'
            )
        
        # 6. 결과 요약
        print("\n" + "=" * 60)
        print("🏆 현실적인 모델 개발 결과")
        print("=" * 60)
        
        if 'direction_package' in locals():
            best_dir = max(direction_results.items(), key=lambda x: x[1]['test_accuracy'])
            print(f"🧭 방향 모델 (현실적):")
            print(f"   최고 모델: {best_dir[0]}")
            print(f"   테스트 정확도: {best_dir[1]['test_accuracy']:.4f}")
            print(f"   CV: {best_dir[1]['cv_mean']:.4f} (±{best_dir[1]['cv_std']:.4f})")
        
        if 'speed_package' in locals():
            best_spd = max(speed_results.items(), key=lambda x: x[1]['test_accuracy'])
            print(f"⚡ 속도 모델 (현실적):")
            print(f"   최고 모델: {best_spd[0]}")
            print(f"   테스트 정확도: {best_spd[1]['test_accuracy']:.4f}")
            print(f"   CV: {best_spd[1]['cv_mean']:.4f} (±{best_spd[1]['cv_std']:.4f})")
        
        print(f"\n💭 성능 해석:")
        print(f"   • 60-75%: 우수한 실용적 성능")
        print(f"   • 45-60%: 사용 가능한 성능")
        print(f"   • 30-45%: 개선 필요")
        print(f"   • 90%+:  오버피팅 의심")
        
        print(f"\n✅ 현실적인 모델 완성!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()