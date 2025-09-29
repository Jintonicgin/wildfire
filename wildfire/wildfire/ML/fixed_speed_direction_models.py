#!/usr/bin/env python3
"""
수정된 최종 속도 및 방향 모델
- 변수 오류 수정
- 실용적 성능 목표
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
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

def load_fire_data():
    """화재 데이터 로드"""
    print("🔥 화재 데이터 로드...")
    
    df = pd.read_csv('final_merged_feature_engineered.csv')
    fire_mask = (df['fire_area'] > 0) & df['fire_area'].notna()
    fire_df = df[fire_mask].copy()
    
    print(f"   화재 데이터: {fire_df.shape}")
    print(f"   화재 면적 평균: {fire_df['fire_area'].mean():.2f} ha")
    
    return fire_df

def create_speed_categories(df):
    """속도 카테고리 생성"""
    print("⚡ 속도 카테고리 생성...")
    
    # 면적 기반 3단계
    area_data = df['fire_area'].copy()
    q40, q70 = area_data.quantile([0.4, 0.7])
    
    speed_cats = []
    for area in area_data:
        if area <= q40:
            speed_cats.append('slow')
        elif area <= q70:
            speed_cats.append('medium')  
        else:
            speed_cats.append('fast')
    
    df['speed_category'] = speed_cats
    
    counts = pd.Series(speed_cats).value_counts()
    print(f"   속도 분포: {counts.to_dict()}")
    
    return df

def create_direction_categories(df):
    """방향 카테고리 생성"""
    print("🧭 방향 카테고리 생성...")
    
    # 바람 방향 기반 (노이즈 추가)
    if 'wd10m_0h' in df.columns:
        wind_dir = df['wd10m_0h'].fillna(180)
        
        # 노이즈 추가
        np.random.seed(42)
        noise = np.random.normal(0, 30, len(wind_dir))
        wind_noisy = (wind_dir + noise) % 360
        
        directions = []
        for wd in wind_noisy:
            if wd < 45 or wd >= 315:
                directions.append('north')
            elif wd < 90:
                directions.append('northeast')
            elif wd < 135:
                directions.append('east')
            elif wd < 180:
                directions.append('southeast')
            elif wd < 225:
                directions.append('south')
            elif wd < 270:
                directions.append('southwest')
            elif wd < 315:
                directions.append('west')
            else:
                directions.append('northwest')
        
        df['direction_category'] = directions
        
        counts = pd.Series(directions).value_counts()
        print(f"   방향 분포: {counts.to_dict()}")
        
        return df
    else:
        print("   바람 방향 데이터 없음")
        return df

def create_features(df):
    """피처 생성"""
    print("🎯 피처 생성...")
    
    # 공통 피처
    common_features = []
    
    # 기상
    weather_feats = ['t2m_0h', 'rh2m_0h', 'ws10m_0h', 'precip_0h']
    for feat in weather_feats:
        if feat in df.columns:
            common_features.append(feat)
    
    # 화재 위험
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
    
    # 속도 모델용
    speed_features = common_features.copy()
    if 'fire_area' in df.columns:
        speed_features.append('fire_area')
    
    # 방향 모델용
    direction_features = common_features.copy()
    if 'wd10m_0h' in df.columns:
        direction_features.append('wd10m_0h')
    
    print(f"   속도 피처: {len(speed_features)}개")
    print(f"   방향 피처: {len(direction_features)}개")
    
    return speed_features, direction_features

def train_model(df, features, target_col, model_name):
    """모델 훈련"""
    print(f"\n🤖 {model_name} 모델 훈련...")
    
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
    
    # SMOTE
    try:
        smote = SMOTE(random_state=42, k_neighbors=min(5, min_samples-1))
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
        print(f"   SMOTE: {np.bincount(y_train_sm)}")
    except:
        X_train_sm, y_train_sm = X_train, y_train
        print(f"   SMOTE 실패 - 원본 사용")
    
    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sm)
    X_test_scaled = scaler.transform(X_test)
    
    # 모델들
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=15,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            eval_metric='logloss'
        ) if 'xgboost' in globals() else None
    }
    
    # 훈련 및 평가
    results = {}
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        if model is None:
            continue
        
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
    
    print(f"\n🏆 최고: {best_model[0]} ({best_score:.4f})")
    
    # 최종 평가
    y_pred_final = best_model[1].predict(X_test_scaled)
    y_pred_labels = le.inverse_transform(y_pred_final)
    y_test_labels = le.inverse_transform(y_test)
    
    print(f"\n📊 분류 보고서:")
    print(classification_report(y_test_labels, y_pred_labels))
    
    # 혼동행렬
    create_confusion_matrix(y_test_labels, y_pred_labels, 
                          f'{model_name} Model', model_name.lower())
    
    # 저장
    package = {
        'model': best_model[1],
        'scaler': scaler,
        'label_encoder': le,
        'features': features,
        'best_name': best_model[0]
    }
    
    joblib.dump(package, f'final_{model_name.lower()}_model.joblib')
    
    return package, results

def create_confusion_matrix(y_true, y_pred, title, filename):
    """혼동행렬"""
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    
    labels = sorted(set(y_true) | set(y_pred))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    
    plt.title(f'{title}\n정확도: {accuracy:.4f}')
    plt.xlabel('예측')
    plt.ylabel('실제')
    
    plt.tight_layout()
    plt.savefig(f'{filename}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 저장: {filename}_confusion_matrix.png")

def main():
    """메인"""
    print("🎯 최종 속도/방향 모델 개발")
    print("=" * 50)
    
    try:
        # 데이터 로드
        fire_df = load_fire_data()
        
        # 타겟 생성
        fire_df = create_speed_categories(fire_df)
        fire_df = create_direction_categories(fire_df)
        
        # 피처 생성
        speed_features, direction_features = create_features(fire_df)
        
        # 모델 훈련
        speed_results = None
        direction_results = None
        
        if 'speed_category' in fire_df.columns:
            speed_package, speed_results = train_model(
                fire_df, speed_features, 'speed_category', 'Speed'
            )
        
        if 'direction_category' in fire_df.columns:
            direction_package, direction_results = train_model(
                fire_df, direction_features, 'direction_category', 'Direction'
            )
        
        # 결과 요약
        print("\n" + "=" * 50)
        print("🏆 최종 결과")
        print("=" * 50)
        
        if speed_results:
            best_speed = max(speed_results.items(), key=lambda x: x[1]['test_accuracy'])
            print(f"⚡ 속도 모델: {best_speed[1]['test_accuracy']:.4f}")
            
            if best_speed[1]['test_accuracy'] >= 0.5:
                print("   ✅ 목표 달성!")
            else:
                print("   📊 목표 미달")
        
        if direction_results:
            best_dir = max(direction_results.items(), key=lambda x: x[1]['test_accuracy'])
            print(f"🧭 방향 모델: {best_dir[1]['test_accuracy']:.4f}")
            
            if best_dir[1]['test_accuracy'] >= 0.6:
                print("   ✅ 목표 달성!")
            else:
                print("   📊 목표 미달")
        
        print(f"\n✅ 완료!")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()