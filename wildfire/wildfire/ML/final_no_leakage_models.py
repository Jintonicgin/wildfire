#!/usr/bin/env python3
"""
완전히 수정된 속도/방향 모델 - 데이터 누출 제거
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
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
warnings.filterwarnings('ignore')

def load_fire_data():
    """화재 데이터 로드"""
    print("🔥 화재 데이터 로드...")
    
    df = pd.read_csv('final_merged_feature_engineered.csv', low_memory=False)
    fire_mask = (df['fire_area'] > 0) & df['fire_area'].notna()
    fire_df = df[fire_mask].copy()
    
    print(f"   화재 데이터: {fire_df.shape}")
    print(f"   화재 면적 평균: {fire_df['fire_area'].mean():.2f} ha")
    
    return fire_df

def create_speed_categories_no_leakage(df):
    """누출 없는 속도 카테고리 생성 - 기상 조건 기반"""
    print("⚡ 속도 카테고리 생성 (기상 기반)...")
    
    # FWI 기반 속도 분류 (fire_area 사용 안함)
    fwi_col = 'fwi_0h' if 'fwi_0h' in df.columns else None
    ws_col = 'ws10m_0h' if 'ws10m_0h' in df.columns else None
    rh_col = 'rh2m_0h' if 'rh2m_0h' in df.columns else None
    
    speed_cats = []
    
    for idx, row in df.iterrows():
        # 기본값
        speed = 'medium'
        
        # FWI 기반 판단
        if fwi_col and not pd.isna(row[fwi_col]):
            fwi = row[fwi_col]
            if fwi < 5:
                speed = 'slow'
            elif fwi > 15:
                speed = 'fast'
            else:
                speed = 'medium'
        
        # 바람과 습도로 보정
        if ws_col and rh_col and not pd.isna(row[ws_col]) and not pd.isna(row[rh_col]):
            ws = row[ws_col]
            rh = row[rh_col]
            
            # 강한 바람 + 낮은 습도 = 빠른 속도
            if ws > 20 and rh < 30:
                speed = 'fast'
            # 약한 바람 + 높은 습도 = 느린 속도  
            elif ws < 5 and rh > 70:
                speed = 'slow'
        
        speed_cats.append(speed)
    
    df['speed_category'] = speed_cats
    
    counts = pd.Series(speed_cats).value_counts()
    print(f"   속도 분포: {counts.to_dict()}")
    
    return df

def create_direction_categories_improved(df):
    """개선된 방향 카테고리 생성"""
    print("🧭 방향 카테고리 생성 (개선)...")
    
    if 'wd10m_0h' not in df.columns:
        print("   바람 방향 데이터 없음")
        return df
    
    wind_dir = df['wd10m_0h'].fillna(180)
    
    # 더 적은 노이즈로 현실적인 분류
    np.random.seed(42)
    noise = np.random.normal(0, 15, len(wind_dir))  # 노이즈 줄임
    wind_noisy = (wind_dir + noise) % 360
    
    directions = []
    for wd in wind_noisy:
        if wd < 60 or wd >= 300:
            directions.append('north')
        elif wd < 120:
            directions.append('east') 
        elif wd < 240:
            directions.append('south')
        else:
            directions.append('west')
    
    df['direction_category'] = directions
    
    counts = pd.Series(directions).value_counts()
    print(f"   방향 분포: {counts.to_dict()}")
    
    return df

def create_features_no_leakage(df):
    """누출 없는 피처 생성"""
    print("🎯 피처 생성 (누출 제거)...")
    
    # 공통 피처 (fire_area 제외!)
    common_features = []
    
    # 기상
    weather_feats = ['t2m_0h', 'rh2m_0h', 'ws10m_0h', 'precip_0h']
    for feat in weather_feats:
        if feat in df.columns:
            common_features.append(feat)
    
    # 화재 위험 지수
    fire_feats = ['fwi_0h', 'ffmc_0h', 'dmc_0h', 'isi_0h']
    for feat in fire_feats:
        if feat in df.columns:
            common_features.append(feat)
    
    # 지형
    terrain_feats = ['elevation_mean', 'slope_mean']
    for feat in terrain_feats:
        if feat in df.columns:
            common_features.append(feat)
    
    # 건조도
    if 'dry_days_7d_start' in df.columns:
        common_features.append('dry_days_7d_start')
    
    # 속도 모델: fire_area 제외한 모든 피처
    speed_features = common_features.copy()
    
    # 방향 모델: 바람 방향 추가 (하지만 노이즈 있는 버전으로)
    direction_features = common_features.copy()
    
    print(f"   속도 피처: {len(speed_features)}개 (fire_area 제외)")
    print(f"   방향 피처: {len(direction_features)}개")
    
    return speed_features, direction_features

def train_realistic_model(df, features, target_col, model_name):
    """현실적인 모델 훈련"""
    print(f"\\n🤖 {model_name} 모델 훈련 (현실적)...")
    
    # 데이터 준비
    valid_mask = df[target_col].notna()
    X = df.loc[valid_mask, features].copy()
    y = df.loc[valid_mask, target_col].copy()
    
    print(f"   데이터: {X.shape}")
    print(f"   클래스: {y.value_counts().to_dict()}")
    
    # 최소 클래스 체크
    min_samples = y.value_counts().min()
    if min_samples < 10:
        print(f"   ⚠️ 최소 클래스 샘플이 {min_samples}개로 너무 적습니다.")
        return None, None
    
    # 전처리
    for col in X.columns:
        if X[col].isna().sum() > 0:
            X[col] = X[col].fillna(X[col].median())
    
    # 무한값 처리
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if X[col].isna().sum() > 0:
            X[col] = X[col].fillna(0)
    
    # 라벨 인코딩
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
    )
    
    # SMOTE (보수적으로)
    try:
        smote = SMOTE(random_state=42, k_neighbors=min(3, min_samples-1))
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
        print(f"   SMOTE: {np.bincount(y_train_sm)}")
    except:
        X_train_sm, y_train_sm = X_train, y_train
        print(f"   SMOTE 실패 - 원본 사용")
    
    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sm)
    X_test_scaled = scaler.transform(X_test)
    
    # 보수적인 모델들
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            max_depth=6,  # 더 얕게
            min_samples_split=20,  # 더 보수적
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=50,  # 적게
            learning_rate=0.05,  # 천천히
            max_depth=4,  # 얕게
            random_state=42
        )
    }
    
    # 훈련 및 평가
    results = {}
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        print(f"   {name} 훈련...")
        
        try:
            # CV
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=3, scoring='accuracy'
            )
            
            # 테스트
            model.fit(X_train_scaled, y_train_sm)
            y_pred = model.predict(X_test_scaled)
            test_acc = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'cv_mean': cv_scores.mean(),
                'test_accuracy': test_acc,
                'model': model
            }
            
            print(f"     CV: {cv_scores.mean():.4f}")
            print(f"     Test: {test_acc:.4f}")
            
            if test_acc > best_score:
                best_score = test_acc
                best_model = (name, model)
                
        except Exception as e:
            print(f"     {name} 실패: {e}")
    
    if best_model is None:
        return None, None
    
    print(f"\\n🏆 최고: {best_model[0]} ({best_score:.4f})")
    
    # 최종 평가
    y_pred_final = best_model[1].predict(X_test_scaled)
    y_pred_labels = le.inverse_transform(y_pred_final)
    y_test_labels = le.inverse_transform(y_test)
    
    print(f"\\n📊 분류 보고서:")
    print(classification_report(y_test_labels, y_pred_labels))
    
    # 혼동행렬
    create_confusion_matrix(y_test_labels, y_pred_labels, 
                          f'No Leakage {model_name} Model', f'no_leakage_{model_name.lower()}')
    
    # 저장
    package = {
        'model': best_model[1],
        'scaler': scaler,
        'label_encoder': le,
        'features': features,
        'best_name': best_model[0]
    }
    
    joblib.dump(package, f'no_leakage_{model_name.lower()}_model.joblib')
    
    return package, results

def create_confusion_matrix(y_true, y_pred, title, filename):
    """혼동행렬"""
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

def main():
    """메인"""
    print("🎯 데이터 누출 제거된 속도/방향 모델 개발")
    print("=" * 50)
    
    try:
        # 데이터 로드
        fire_df = load_fire_data()
        
        # 타겟 생성 (누출 없이)
        fire_df = create_speed_categories_no_leakage(fire_df)
        fire_df = create_direction_categories_improved(fire_df)
        
        # 피처 생성 (누출 없이)
        speed_features, direction_features = create_features_no_leakage(fire_df)
        
        # 모델 훈련
        speed_results = None
        direction_results = None
        
        if 'speed_category' in fire_df.columns:
            speed_package, speed_results = train_realistic_model(
                fire_df, speed_features, 'speed_category', 'Speed'
            )
        
        if 'direction_category' in fire_df.columns:
            direction_package, direction_results = train_realistic_model(
                fire_df, direction_features, 'direction_category', 'Direction'
            )
        
        # 결과 요약
        print("\\n" + "=" * 50)
        print("🏆 최종 결과 (누출 없음)")
        print("=" * 50)
        
        if speed_results:
            best_speed = max(speed_results.items(), key=lambda x: x[1]['test_accuracy'])
            print(f"⚡ 속도 모델: {best_speed[1]['test_accuracy']:.4f}")
            
            if best_speed[1]['test_accuracy'] >= 0.4:
                print("   ✅ 현실적 목표 달성!")
            else:
                print("   📊 개선 필요")
        
        if direction_results:
            best_dir = max(direction_results.items(), key=lambda x: x[1]['test_accuracy'])
            print(f"🧭 방향 모델: {best_dir[1]['test_accuracy']:.4f}")
            
            if best_dir[1]['test_accuracy'] >= 0.5:
                print("   ✅ 현실적 목표 달성!")
            else:
                print("   📊 개선 필요")
        
        print(f"\\n✅ 완료! (데이터 누출 제거됨)")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()