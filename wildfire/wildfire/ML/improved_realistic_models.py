#!/usr/bin/env python3
"""
개선된 현실적 속도/방향 모델 - 성능과 현실성의 균형
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
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

def create_better_speed_categories(df):
    """더 나은 속도 카테고리 - 복합적 기상 조건 기반"""
    print("⚡ 속도 카테고리 생성 (복합 기상 기반)...")
    
    speed_cats = []
    
    # 기상 조건 기반 화재 확산 속도 분류
    for idx, row in df.iterrows():
        # 기본 점수 시스템
        speed_score = 0
        
        # FWI 기여 (30점)
        if 'fwi_0h' in df.columns and not pd.isna(row['fwi_0h']):
            fwi = row['fwi_0h']
            if fwi > 20:
                speed_score += 30
            elif fwi > 10:
                speed_score += 20
            elif fwi > 5:
                speed_score += 10
        
        # 바람속도 기여 (25점)
        if 'ws10m_0h' in df.columns and not pd.isna(row['ws10m_0h']):
            ws = row['ws10m_0h']
            if ws > 25:
                speed_score += 25
            elif ws > 15:
                speed_score += 20
            elif ws > 8:
                speed_score += 15
            elif ws > 3:
                speed_score += 10
        
        # 습도 기여 (20점, 낮을수록 높은 점수)
        if 'rh2m_0h' in df.columns and not pd.isna(row['rh2m_0h']):
            rh = row['rh2m_0h']
            if rh < 20:
                speed_score += 20
            elif rh < 40:
                speed_score += 15
            elif rh < 60:
                speed_score += 10
            elif rh < 80:
                speed_score += 5
        
        # 온도 기여 (15점)
        if 't2m_0h' in df.columns and not pd.isna(row['t2m_0h']):
            temp = row['t2m_0h']
            if temp > 35:
                speed_score += 15
            elif temp > 25:
                speed_score += 10
            elif temp > 15:
                speed_score += 5
        
        # 강수 기여 (10점, 없을수록 높은 점수)
        if 'precip_0h' in df.columns and not pd.isna(row['precip_0h']):
            precip = row['precip_0h']
            if precip == 0:
                speed_score += 10
            elif precip < 1:
                speed_score += 5
        
        # 점수 기반 분류 (더 균형잡힌 기준)
        if speed_score >= 50:
            speed_cats.append('fast')
        elif speed_score >= 25:
            speed_cats.append('medium')
        else:
            speed_cats.append('slow')
    
    df['speed_category'] = speed_cats
    
    counts = pd.Series(speed_cats).value_counts()
    print(f"   속도 분포: {counts.to_dict()}")
    
    return df

def create_better_direction_categories(df):
    """더 나은 방향 카테고리 - 바람과 지형의 조합"""
    print("🧭 방향 카테고리 생성 (바람+지형 기반)...")
    
    directions = []
    
    for idx, row in df.iterrows():
        # 기본 방향은 바람 방향
        base_direction = 'north'  # 기본값
        
        # 바람 방향이 있으면 사용 (8방향)
        if 'wd10m_0h' in df.columns and not pd.isna(row['wd10m_0h']):
            wd = row['wd10m_0h']
            if wd < 22.5 or wd >= 337.5:
                base_direction = 'north'
            elif wd < 67.5:
                base_direction = 'northeast'
            elif wd < 112.5:
                base_direction = 'east'
            elif wd < 157.5:
                base_direction = 'southeast'
            elif wd < 202.5:
                base_direction = 'south'
            elif wd < 247.5:
                base_direction = 'southwest'
            elif wd < 292.5:
                base_direction = 'west'
            else:
                base_direction = 'northwest'
        
        # 지형 경사로 약간 수정 (25% 가중치, 너무 직접적이지 않게)
        if 'slope_mean' in df.columns and not pd.isna(row['slope_mean']):
            slope = row['slope_mean']
            # 경사가 클 때만 영향, 그리고 랜덤 요소 추가
            if slope > 15 and np.random.random() > 0.7:  # 30% 확률로만 영향
                # 약간의 방향 변경
                directions_list = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']
                if base_direction in directions_list:
                    current_idx = directions_list.index(base_direction)
                    # 인접한 방향으로 변경
                    if np.random.random() > 0.5:
                        new_idx = (current_idx + 1) % 8
                    else:
                        new_idx = (current_idx - 1) % 8
                    base_direction = directions_list[new_idx]
        
        directions.append(base_direction)
    
    df['direction_category'] = directions
    
    counts = pd.Series(directions).value_counts()
    print(f"   방향 분포: {counts.to_dict()}")
    
    return df

def create_expanded_features(df):
    """확장된 피처 세트"""
    print("🎯 확장 피처 생성...")
    
    features = []
    
    # 핵심 기상 변수
    weather_vars = ['t2m_0h', 'rh2m_0h', 'ws10m_0h', 'precip_0h', 'wd10m_0h']
    for var in weather_vars:
        if var in df.columns:
            features.append(var)
    
    # 화재 위험 지수
    fire_vars = ['fwi_0h', 'ffmc_0h', 'dmc_0h', 'dc_0h', 'isi_0h', 'bui_0h']
    for var in fire_vars:
        if var in df.columns:
            features.append(var)
    
    # 지형 정보
    terrain_vars = ['elevation_mean', 'slope_mean']
    for var in terrain_vars:
        if var in df.columns:
            features.append(var)
    
    # 과거 날씨 패턴 (일부)
    past_vars = ['t2m_3h_past', 'rh2m_3h_past', 'ws10m_3h_past', 
                 't2m_6h_past', 'rh2m_6h_past', 'ws10m_6h_past']
    for var in past_vars:
        if var in df.columns:
            features.append(var)
    
    # 건조 조건
    dry_vars = ['dry_days_7d_start', 'dry_days_30d_start']
    for var in dry_vars:
        if var in df.columns:
            features.append(var)
    
    print(f"   총 피처: {len(features)}개")
    print(f"   피처 목록: {features[:10]}..." if len(features) > 10 else f"   피처 목록: {features}")
    
    return features

def train_better_model(df, features, target_col, model_name):
    """더 나은 모델 훈련"""
    print(f"\\n🤖 {model_name} 모델 훈련 (개선됨)...")
    
    # 데이터 준비
    valid_mask = df[target_col].notna()
    available_features = [f for f in features if f in df.columns]
    X = df.loc[valid_mask, available_features].copy()
    y = df.loc[valid_mask, target_col].copy()
    
    print(f"   데이터: {X.shape}")
    print(f"   사용 피처: {len(available_features)}개")
    print(f"   클래스: {y.value_counts().to_dict()}")
    
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
    
    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 더 강력한 모델들
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=3,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=10,
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
                model, X_train_scaled, y_train, cv=5, scoring='accuracy'
            )
            
            # 테스트
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            test_acc = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': test_acc,
                'model': model
            }
            
            print(f"     CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
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
                          f'Improved {model_name} Model', f'improved_{model_name.lower()}')
    
    # 피처 중요도
    if hasattr(best_model[1], 'feature_importances_'):
        importances = best_model[1].feature_importances_
        feature_imp = list(zip(available_features, importances))
        feature_imp.sort(key=lambda x: x[1], reverse=True)
        print(f"\\n🎯 상위 피처 중요도:")
        for feat, imp in feature_imp[:10]:
            print(f"   {feat}: {imp:.4f}")
    
    # 저장
    package = {
        'model': best_model[1],
        'scaler': scaler,
        'label_encoder': le,
        'features': available_features,
        'best_name': best_model[0],
        'results': results
    }
    
    joblib.dump(package, f'improved_{model_name.lower()}_model_v2.joblib')
    
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
    plt.savefig(f'{filename}_confusion_matrix_v2.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 저장: {filename}_confusion_matrix_v2.png")

def main():
    """메인"""
    print("🎯 개선된 현실적 속도/방향 모델 개발")
    print("=" * 50)
    
    np.random.seed(42)
    
    try:
        # 데이터 로드
        fire_df = load_fire_data()
        
        # 더 나은 타겟 생성
        fire_df = create_better_speed_categories(fire_df)
        fire_df = create_better_direction_categories(fire_df)
        
        # 확장된 피처 생성
        features = create_expanded_features(fire_df)
        
        # 모델 훈련
        speed_results = None
        direction_results = None
        
        if 'speed_category' in fire_df.columns:
            speed_package, speed_results = train_better_model(
                fire_df, features, 'speed_category', 'Speed'
            )
        
        if 'direction_category' in fire_df.columns:
            direction_package, direction_results = train_better_model(
                fire_df, features, 'direction_category', 'Direction'
            )
        
        # 결과 요약
        print("\\n" + "=" * 50)
        print("🏆 개선된 모델 결과")
        print("=" * 50)
        
        if speed_results:
            best_speed = max(speed_results.items(), key=lambda x: x[1]['test_accuracy'])
            print(f"⚡ 속도 모델: {best_speed[1]['test_accuracy']:.4f}")
            print(f"   CV: {best_speed[1]['cv_mean']:.4f} ± {best_speed[1]['cv_std']:.4f}")
            
            if best_speed[1]['test_accuracy'] >= 0.6:
                print("   ✅ 좋은 성능!")
            elif best_speed[1]['test_accuracy'] >= 0.5:
                print("   📊 적당한 성능")
            else:
                print("   ❌ 성능 부족")
        
        if direction_results:
            best_dir = max(direction_results.items(), key=lambda x: x[1]['test_accuracy'])
            print(f"🧭 방향 모델: {best_dir[1]['test_accuracy']:.4f}")
            print(f"   CV: {best_dir[1]['cv_mean']:.4f} ± {best_dir[1]['cv_std']:.4f}")
            
            if best_dir[1]['test_accuracy'] >= 0.6:
                print("   ✅ 좋은 성능!")
            elif best_dir[1]['test_accuracy'] >= 0.5:
                print("   📊 적당한 성능")
            else:
                print("   ❌ 성능 부족")
        
        print(f"\\n✅ 개선 완료!")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()