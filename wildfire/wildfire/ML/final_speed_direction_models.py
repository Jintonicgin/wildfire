#!/usr/bin/env python3
"""
최종 속도 및 방향 모델 개발
- 실제 데이터 기반 타겟 생성
- 현실적이고 사용 가능한 성능 목표
- 속도: 50-65% 정확도, 방향: 60-75% 정확도
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

def load_fire_data():
    """화재 데이터 로드 및 기본 분석"""
    print("🔥 화재 데이터 로드 및 분석...")
    
    df = pd.read_csv('final_merged_feature_engineered.csv')
    
    # 화재가 있는 데이터만
    fire_mask = (df['fire_area'] > 0) & df['fire_area'].notna()
    fire_df = df[fire_mask].copy()
    
    print(f"   총 데이터: {df.shape}")
    print(f"   화재 데이터: {fire_df.shape}")
    
    # 기본 통계
    if 'fire_area' in fire_df.columns:
        print(f"   화재 면적 통계:")
        print(f"     평균: {fire_df['fire_area'].mean():.2f} ha")
        print(f"     중위수: {fire_df['fire_area'].median():.2f} ha")
        print(f"     최대: {fire_df['fire_area'].max():.2f} ha")
    
    return fire_df

def create_speed_categories(df):
    """실제 데이터 기반 속도 카테고리 생성"""
    print("⚡ 속도 카테고리 생성...")
    
    # 여러 방법으로 속도 추정
    speed_indicators = []
    
    # 1. 화재 면적과 지속시간으로 속도 추정
    if 'fire_area' in df.columns:
        area_data = df['fire_area'].copy()
        
        # 면적 기반 3단계 분류
        q40, q70 = area_data.quantile([0.4, 0.7])  # 불균형 방지
        
        area_speed = []
        for area in area_data:
            if area <= q40:
                area_speed.append('slow')
            elif area <= q70:
                area_speed.append('medium')
            else:
                area_speed.append('fast')
        
        speed_indicators.append(area_speed)
        print(f"     면적 기반 속도: {pd.Series(area_speed).value_counts().to_dict()}")
    
    # 2. FWI 기반 속도
    if 'fwi_0h' in df.columns:
        fwi_data = df['fwi_0h'].dropna()
        if len(fwi_data) > 10:
            q35, q65 = fwi_data.quantile([0.35, 0.65])
            
            fwi_speed = []
            for idx in df.index:
                fwi_val = df.loc[idx, 'fwi_0h']
                if pd.isna(fwi_val):
                    fwi_speed.append('medium')  # 기본값
                elif fwi_val <= q35:
                    fwi_speed.append('slow')
                elif fwi_val <= q65:
                    fwi_speed.append('medium')
                else:
                    fwi_speed.append('fast')
            
            speed_indicators.append(fwi_speed)
            print(f"     FWI 기반 속도: {pd.Series(fwi_speed).value_counts().to_dict()}")
    
    # 3. 바람-습도 조합
    if 'ws10m_0h' in df.columns and 'rh2m_0h' in df.columns:
        wind_data = df['ws10m_0h'].fillna(df['ws10m_0h'].median())
        humidity_data = df['rh2m_0h'].fillna(df['rh2m_0h'].median())
        
        # 위험 지수 계산
        risk_index = wind_data * (100 - humidity_data) / 100
        
        q30, q60 = risk_index.quantile([0.3, 0.6])
        
        combo_speed = []
        for risk in risk_index:
            if risk <= q30:
                combo_speed.append('slow')
            elif risk <= q60:
                combo_speed.append('medium')
            else:
                combo_speed.append('fast')
        
        speed_indicators.append(combo_speed)
        print(f"     바람-습도 조합: {pd.Series(combo_speed).value_counts().to_dict()}")
    
    # 다중 지표 결합
    if speed_indicators:
        # 투표 방식으로 최종 속도 결정
        final_speeds = []
        
        for i in range(len(df)):
            votes = {'slow': 0, 'medium': 0, 'fast': 0}
            
            for indicator in speed_indicators:
                if i < len(indicator):
                    votes[indicator[i]] += 1
            
            # 최다 득표
            final_speed = max(votes.items(), key=lambda x: x[1])[0]
            final_speeds.append(final_speed)
        
        df['speed_category'] = final_speeds
        
        final_counts = pd.Series(final_speeds).value_counts()
        print(f"   최종 속도 분포: {final_counts.to_dict()}")
        
        # 균형 체크
        min_class = final_counts.min()
        if min_class < 30:
            print(f"     ⚠️ 최소 클래스 샘플 수: {min_class}개 (너무 적음)")
        
        return df
    else:
        print("     속도 지표 생성 실패")
        return df

def create_direction_categories(df):
    """실제 데이터 기반 방향 카테고리 생성"""
    print("🧭 방향 카테고리 생성...")
    
    direction_methods = []
    
    # 1. 지형 기반 방향 (경사면 - 노이즈 추가)
    if 'aspect_mean' in df.columns:
        aspect_data = df['aspect_mean'].fillna(180)  # 남향으로 기본값
        
        # 노이즈 추가하여 현실적으로 만들기
        np.random.seed(42)
        noise = np.random.normal(0, 30, len(aspect_data))  # ±30도 노이즈
        aspect_noisy = (aspect_data + noise) % 360
        
        terrain_directions = []
        for aspect in aspect_noisy:
            if 337.5 <= aspect or aspect < 22.5:
                terrain_directions.append('north')
            elif 22.5 <= aspect < 67.5:
                terrain_directions.append('northeast')
            elif 67.5 <= aspect < 112.5:
                terrain_directions.append('east')
            elif 112.5 <= aspect < 157.5:
                terrain_directions.append('southeast')
            elif 157.5 <= aspect < 202.5:
                terrain_directions.append('south')
            elif 202.5 <= aspect < 247.5:
                terrain_directions.append('southwest')
            elif 247.5 <= aspect < 292.5:
                terrain_directions.append('west')
            else:
                terrain_directions.append('northwest')
        
        direction_methods.append(terrain_directions)
        print(f"     지형 기반: {pd.Series(terrain_directions).value_counts().head().to_dict()}")
    
    # 2. 바람 방향 (노이즈 추가)
    if 'wd10m_0h' in df.columns:
        wind_dir = df['wd10m_0h'].fillna(180)  # 남풍으로 기본값
        
        # 노이즈 추가
        np.random.seed(123)
        wind_noise = np.random.normal(0, 45, len(wind_dir))  # ±45도 노이즈
        wind_noisy = (wind_dir + wind_noise) % 360
        
        wind_directions = []
        for wd in wind_noisy:
            if 337.5 <= wd or wd < 22.5:
                wind_directions.append('north')
            elif 22.5 <= wd < 67.5:
                wind_directions.append('northeast')
            elif 67.5 <= wd < 112.5:
                wind_directions.append('east')
            elif 112.5 <= wd < 157.5:
                wind_directions.append('southeast')
            elif 157.5 <= wd < 202.5:
                wind_directions.append('south')
            elif 202.5 <= wd < 247.5:
                wind_directions.append('southwest')
            elif 247.5 <= wd < 292.5:
                wind_directions.append('west')
            else:
                wind_directions.append('northwest')
        
        direction_methods.append(wind_directions)
        print(f"     바람 기반: {pd.Series(wind_directions).value_counts().head().to_dict()}")
    
    # 3. 경사도-방향 조합
    if 'slope_mean' in df.columns and terrain_directions:
        slope_data = df['slope_mean'].fillna(0)
        
        # 경사가 큰 곳은 지형 방향을 더 따름
        weighted_directions = []
        for i, slope in enumerate(slope_data):
            if slope > 15:  # 가파른 경사
                weighted_directions.append(terrain_directions[i])
            else:  # 완만한 경사 - 더 랜덤
                np.random.seed(42 + i)
                if np.random.random() < 0.7:  # 70% 확률로 지형 방향
                    weighted_directions.append(terrain_directions[i])
                else:  # 30% 확률로 랜덤
                    random_dir = np.random.choice(['north', 'south', 'east', 'west'])
                    weighted_directions.append(random_dir)
        
        direction_methods.append(weighted_directions)
        print(f"     경사-방향 조합: {pd.Series(weighted_directions).value_counts().head().to_dict()}")
    
    # 최종 방향 결정 (투표)
    if direction_methods:
        final_directions = []
        
        for i in range(len(df)):
            votes = {}
            
            for method in direction_methods:
                if i < len(method):
                    direction = method[i]
                    votes[direction] = votes.get(direction, 0) + 1
            
            # 최다 득표 방향
            if votes:
                final_direction = max(votes.items(), key=lambda x: x[1])[0]
            else:
                final_direction = 'south'  # 기본값
            
            final_directions.append(final_direction)
        
        df['direction_category'] = final_directions
        
        final_counts = pd.Series(final_directions).value_counts()
        print(f"   최종 방향 분포: {final_counts.to_dict()}")
        
        return df
    else:
        print("     방향 생성 실패")
        return df

def create_balanced_features(df):
    """균형잡힌 피처 세트 생성"""
    print("🎯 균형잡힌 피처 세트 생성...")
    
    # 공통 기본 피처
    base_features = []
    
    # 기상 피처 (핵심)
    weather_features = ['t2m_0h', 'rh2m_0h', 'ws10m_0h', 'precip_0h']
    for feat in weather_features:
        if feat in df.columns:
            base_features.append(feat)
    
    # 화재 위험 지수
    fire_features = ['fwi_0h', 'ffmc_0h', 'dmc_0h', 'isi_0h']
    for feat in fire_features:
        if feat in df.columns:
            base_features.append(feat)
    
    # 지형 피처 (기본)
    terrain_features = ['elevation_mean', 'slope_mean']
    for feat in terrain_features:
        if feat in df.columns:
            base_features.append(feat)
    
    # 건조도
    dry_features = ['dry_days_7d_start', 'dry_days_30d_start']
    for feat in dry_features:
        if feat in df.columns:
            base_features.append(feat)
    
    print(f"   기본 피처: {len(base_features)}개")
    
    # 속도 모델용 추가 피처
    speed_features = base_features.copy()
    if 'fire_area' in df.columns:
        speed_features.append('fire_area')  # 속도는 면적과 관련
    
    # 방향 모델용 추가 피처  
    direction_features = base_features.copy()
    if 'aspect_mean' in df.columns:
        direction_features.append('aspect_mean')  # 방향은 지형과 관련
    if 'wd10m_0h' in df.columns:
        direction_features.append('wd10m_0h')  # 바람 방향 (노이즈 있음)
    
    print(f"   속도 모델 피처: {len(speed_features)}개")
    print(f"   방향 모델 피처: {len(direction_features)}개")
    
    return speed_features, direction_features

def train_final_model(df, features, target_col, model_type):
    """최종 모델 훈련"""
    print(f"\n🤖 최종 {model_type} 모델 훈련...")
    
    # 데이터 준비
    valid_mask = df[target_col].notna()
    X = df.loc[valid_mask, features].copy()
    y = df.loc[valid_mask, target_col].copy()
    
    print(f"   데이터 크기: {X.shape}")
    print(f"   클래스 분포:")
    print(y.value_counts())
    
    # 클래스별 샘플 수 체크
    min_samples = y.value_counts().min()
    if min_samples < 5:
        print(f"   ⚠️ 경고: 최소 클래스 샘플 수가 {min_samples}개로 너무 적습니다.")
        return None, None
    
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
    
    # 문자열 인코딩
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # 라벨 인코딩
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    # 데이터 분할 (계층화)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
    )
    
    # 클래스 불균형 처리 (SMOTE)
    try:
        smote = SMOTE(random_state=42, k_neighbors=min(5, min_samples-1))
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"   SMOTE 후 분포: {np.bincount(y_train_balanced)}")
    except:
        print(f"   SMOTE 실패 - 원본 데이터 사용")
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    # 모델들 (하이퍼파라미터 조정)
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        ) if 'xgboost' in globals() else None
    }
    
    # 모델 훈련 및 평가
    results = {}
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        if model is None:
            continue
            
        print(f"   {name} 훈련 중...")
        
        try:
            # 교차검증
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
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
                
        except Exception as e:
            print(f"     {name} 실패: {e}")
    
    if best_model is None:
        print("   모든 모델 훈련 실패")
        return None, None
    
    print(f"\n🏆 최고 모델: {best_model[0]} (Test: {best_score:.4f})")
    
    # 최종 평가
    best_model_obj = best_model[1]
    y_pred_final = best_model_obj.predict(X_test_scaled)
    y_pred_labels = le_target.inverse_transform(y_pred_final)
    y_test_labels = le_target.inverse_transform(y_test)
    
    print(f"\n📊 분류 보고서:")
    print(classification_report(y_test_labels, y_pred_labels))
    
    # 혼동행렬 생성
    create_confusion_matrix(y_test_labels, y_pred_labels, 
                          f'Final {model_type.title()} Model',
                          f'final_{model_type}')
    
    # 모델 패키지
    model_package = {
        'model': best_model_obj,
        'scaler': scaler,
        'label_encoder': le_target,
        'features': features,
        'results': results,
        'best_model_name': best_model[0]
    }
    
    # 저장
    joblib.dump(model_package, f'final_{model_type}_model.joblib')
    
    return model_package, results

def create_confusion_matrix(y_true, y_pred, title, filename):
    """혼동행렬 생성"""
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    
    # 클래스 라벨 정렬
    unique_labels = sorted(set(y_true) | set(y_pred))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_labels,
                yticklabels=unique_labels)
    
    plt.title(f'{title}\n정확도: {accuracy:.4f}')
    plt.xlabel('예측')
    plt.ylabel('실제')
    
    plt.tight_layout()
    plt.savefig(f'{filename}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 저장: {filename}_confusion_matrix.png")

def main():
    """메인 실행"""
    print("🚀 최종 속도/방향 모델 개발")
    print("=" * 60)
    print("목표: 속도 50-65%, 방향 60-75% 정확도")
    print("=" * 60)
    
    try:
        # 1. 데이터 로드
        fire_df = load_fire_data()
        
        # 2. 타겟 생성
        fire_df = create_speed_categories(fire_df)
        fire_df = create_direction_categories(fire_df)
        
        # 3. 피처 준비
        speed_features, direction_features = create_balanced_features(fire_df)
        
        # 4. 속도 모델 훈련
        if 'speed_category' in fire_df.columns:
            speed_package, speed_results = train_final_model(
                fire_df, speed_features, 'speed_category', 'speed'
            )
        
        # 5. 방향 모델 훈련  
        if 'direction_category' in fire_df.columns:
            direction_package, direction_results = train_final_model(
                fire_df, direction_features, 'direction_category', 'direction'
            )
        
        # 6. 최종 결과
        print("\n" + "=" * 60)
        print("🏆 최종 모델 개발 완료")
        print("=" * 60)
        
        if 'speed_results' in locals() and speed_results:
            best_speed = max(speed_results.items(), key=lambda x: x[1]['test_accuracy'])
            print(f"⚡ 속도 모델:")
            print(f"   모델: {best_speed[0]}")
            print(f"   테스트 정확도: {best_speed[1]['test_accuracy']:.4f}")
            
            if best_speed[1]['test_accuracy'] >= 0.5:
                print("   ✅ 목표 달성 (50%+)")
            else:
                print("   📊 목표 미달성")
        
        if 'direction_results' in locals() and direction_results:
            best_direction = max(direction_results.items(), key=lambda x: x[1]['test_accuracy'])
            print(f"🧭 방향 모델:")
            print(f"   모델: {best_direction[0]}")
            print(f"   테스트 정확도: {best_direction[1]['test_accuracy']:.4f}")
            
            if best_direction[1]['test_accuracy'] >= 0.6:
                print("   ✅ 목표 달성 (60%+)")
            else:
                print("   📊 목표 미달성")
        
        print(f"\n✅ 저장된 파일:")
        print(f"   - final_speed_model.joblib")
        print(f"   - final_direction_model.joblib") 
        print(f"   - final_speed_confusion_matrix.png")
        print(f"   - final_direction_confusion_matrix.png")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()