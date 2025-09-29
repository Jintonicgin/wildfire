#!/usr/bin/env python3
"""
모든 모델 종합 테스트 - 실제 성능 측정
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

def load_test_data():
    """테스트 데이터 로드"""
    print("📊 테스트 데이터 로드...")
    
    df = pd.read_csv('final_merged_feature_engineered.csv', low_memory=False)
    fire_mask = (df['fire_area'] > 0) & df['fire_area'].notna()
    fire_df = df[fire_mask].copy()
    
    print(f"   테스트 데이터: {fire_df.shape}")
    return fire_df

def test_area_model(fire_df):
    """면적 모델 테스트"""
    print("\\n🔥 면적 모델 테스트...")
    
    model_files = [
        'area_model_retrained.joblib',
        'improved_area_model_v2.joblib'
    ]
    
    results = {}
    
    for model_file in model_files:
        try:
            print(f"   {model_file} 테스트...")
            package = joblib.load(model_file)
            
            model = package['model']
            scaler = package['scaler'] 
            features = package['features']
            
            # 데이터 준비
            available_features = [f for f in features if f in fire_df.columns]
            X = fire_df[available_features].copy()
            y = fire_df['fire_area'].copy()
            
            # 전처리
            for col in X.columns:
                if X[col].isna().sum() > 0:
                    X[col] = X[col].fillna(X[col].median())
            
            X = X.replace([np.inf, -np.inf], np.nan)
            for col in X.columns:
                if X[col].isna().sum() > 0:
                    X[col] = X[col].fillna(0)
            
            # 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            # 스케일링
            X_test_scaled = scaler.transform(X_test)
            
            # 예측
            y_pred = model.predict(X_test_scaled)
            
            # 평가
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            results[model_file] = {
                'r2': r2,
                'rmse': rmse,
                'mse': mse,
                'mean_actual': y_test.mean(),
                'mean_pred': y_pred.mean()
            }
            
            print(f"     R²: {r2:.4f}")
            print(f"     RMSE: {rmse:.2f} ha")
            print(f"     실제 평균: {y_test.mean():.2f} ha")
            print(f"     예측 평균: {y_pred.mean():.2f} ha")
            
        except Exception as e:
            print(f"     {model_file} 실패: {e}")
    
    return results

def test_speed_model(fire_df):
    """속도 모델 테스트"""
    print("\\n⚡ 속도 모델 테스트...")
    
    model_files = [
        'improved_speed_model_v2.joblib',
        'speed_model_retrained.joblib',
        'independent_speed_model.joblib'
    ]
    
    results = {}
    
    for model_file in model_files:
        try:
            print(f"   {model_file} 테스트...")
            package = joblib.load(model_file)
            
            model = package['model']
            scaler = package['scaler']
            label_encoder = package['label_encoder']
            features = package['features']
            
            # 데이터 준비
            available_features = [f for f in features if f in fire_df.columns]
            X = fire_df[available_features].copy()
            
            # 속도 카테고리 생성 (모델에 따라)
            if 'improved' in model_file:
                y = create_improved_speed_categories(fire_df)
            elif 'independent' in model_file:
                y = create_independent_speed_categories(fire_df)
            else:
                y = create_basic_speed_categories(fire_df)
            
            # 전처리
            for col in X.columns:
                if X[col].isna().sum() > 0:
                    X[col] = X[col].fillna(X[col].median())
            
            X = X.replace([np.inf, -np.inf], np.nan)
            for col in X.columns:
                if X[col].isna().sum() > 0:
                    X[col] = X[col].fillna(0)
            
            # 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # 스케일링
            X_test_scaled = scaler.transform(X_test)
            
            # 라벨 인코딩
            y_test_encoded = label_encoder.transform(y_test)
            
            # 예측
            y_pred_encoded = model.predict(X_test_scaled)
            y_pred = label_encoder.inverse_transform(y_pred_encoded)
            
            # 평가
            accuracy = accuracy_score(y_test, y_pred)
            
            results[model_file] = {
                'accuracy': accuracy,
                'class_distribution': pd.Series(y_test).value_counts().to_dict(),
                'predictions': pd.Series(y_pred).value_counts().to_dict()
            }
            
            print(f"     정확도: {accuracy:.4f}")
            print(f"     실제 분포: {pd.Series(y_test).value_counts().to_dict()}")
            
        except Exception as e:
            print(f"     {model_file} 실패: {e}")
    
    return results

def test_direction_model(fire_df):
    """방향 모델 테스트"""
    print("\\n🧭 방향 모델 테스트...")
    
    model_files = [
        'improved_direction_model_v2.joblib',
        'direction_model_retrained.joblib',
        'independent_direction_model.joblib'
    ]
    
    results = {}
    
    for model_file in model_files:
        try:
            print(f"   {model_file} 테스트...")
            package = joblib.load(model_file)
            
            model = package['model']
            scaler = package['scaler']
            label_encoder = package['label_encoder']
            features = package['features']
            
            # 데이터 준비
            available_features = [f for f in features if f in fire_df.columns]
            X = fire_df[available_features].copy()
            
            # 방향 카테고리 생성 (모델에 따라)
            if 'improved' in model_file:
                y = create_improved_direction_categories(fire_df)
            elif 'independent' in model_file:
                y = create_independent_direction_categories(fire_df)
            else:
                y = create_basic_direction_categories(fire_df)
            
            # 전처리
            for col in X.columns:
                if X[col].isna().sum() > 0:
                    X[col] = X[col].fillna(X[col].median())
            
            X = X.replace([np.inf, -np.inf], np.nan)
            for col in X.columns:
                if X[col].isna().sum() > 0:
                    X[col] = X[col].fillna(0)
            
            # 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # 스케일링
            X_test_scaled = scaler.transform(X_test)
            
            # 라벨 인코딩
            y_test_encoded = label_encoder.transform(y_test)
            
            # 예측
            y_pred_encoded = model.predict(X_test_scaled)
            y_pred = label_encoder.inverse_transform(y_pred_encoded)
            
            # 평가
            accuracy = accuracy_score(y_test, y_pred)
            
            results[model_file] = {
                'accuracy': accuracy,
                'num_classes': len(set(y_test)),
                'class_distribution': pd.Series(y_test).value_counts().to_dict(),
                'predictions': pd.Series(y_pred).value_counts().to_dict()
            }
            
            print(f"     정확도: {accuracy:.4f}")
            print(f"     클래스 수: {len(set(y_test))}개")
            print(f"     실제 분포: {pd.Series(y_test).value_counts().to_dict()}")
            
        except Exception as e:
            print(f"     {model_file} 실패: {e}")
    
    return results

def create_improved_speed_categories(df):
    """개선된 속도 카테고리"""
    speed_cats = []
    for idx, row in df.iterrows():
        speed_score = 0
        
        if 'fwi_0h' in df.columns and not pd.isna(row['fwi_0h']):
            fwi = row['fwi_0h']
            if fwi > 20: speed_score += 30
            elif fwi > 10: speed_score += 20
            elif fwi > 5: speed_score += 10
        
        if 'ws10m_0h' in df.columns and not pd.isna(row['ws10m_0h']):
            ws = row['ws10m_0h']
            if ws > 25: speed_score += 25
            elif ws > 15: speed_score += 20
            elif ws > 8: speed_score += 15
            elif ws > 3: speed_score += 10
        
        if 'rh2m_0h' in df.columns and not pd.isna(row['rh2m_0h']):
            rh = row['rh2m_0h']
            if rh < 20: speed_score += 20
            elif rh < 40: speed_score += 15
            elif rh < 60: speed_score += 10
            elif rh < 80: speed_score += 5
        
        if speed_score >= 50: speed_cats.append('fast')
        elif speed_score >= 25: speed_cats.append('medium')
        else: speed_cats.append('slow')
    
    return speed_cats

def create_improved_direction_categories(df):
    """개선된 방향 카테고리 (8방향)"""
    np.random.seed(42)
    directions = []
    for idx, row in df.iterrows():
        if 'wd10m_0h' in df.columns and not pd.isna(row['wd10m_0h']):
            wd = row['wd10m_0h']
            if wd < 22.5 or wd >= 337.5: base_direction = 'north'
            elif wd < 67.5: base_direction = 'northeast'
            elif wd < 112.5: base_direction = 'east'
            elif wd < 157.5: base_direction = 'southeast'
            elif wd < 202.5: base_direction = 'south'
            elif wd < 247.5: base_direction = 'southwest'
            elif wd < 292.5: base_direction = 'west'
            else: base_direction = 'northwest'
        else:
            base_direction = 'north'
        directions.append(base_direction)
    return directions

def create_independent_speed_categories(df):
    """독립적 속도 카테고리"""
    np.random.seed(42)
    speed_cats = []
    for idx, row in df.iterrows():
        hash_val = hash(str(idx)) % 100
        if hash_val < 35: speed_cats.append('slow')
        elif hash_val < 70: speed_cats.append('medium')
        else: speed_cats.append('fast')
    return speed_cats

def create_independent_direction_categories(df):
    """독립적 방향 카테고리"""
    np.random.seed(42)
    directions = []
    for idx, row in df.iterrows():
        hash_val = hash(str(idx * 2)) % 4
        if hash_val == 0: directions.append('north')
        elif hash_val == 1: directions.append('east')
        elif hash_val == 2: directions.append('south')
        else: directions.append('west')
    return directions

def create_basic_speed_categories(df):
    """기본 속도 카테고리"""
    area_data = df['fire_area'].copy()
    q33, q67 = area_data.quantile([0.33, 0.67])
    speed_cats = []
    for area in area_data:
        if area <= q33: speed_cats.append('slow')
        elif area <= q67: speed_cats.append('medium')
        else: speed_cats.append('fast')
    return speed_cats

def create_basic_direction_categories(df):
    """기본 방향 카테고리"""
    if 'wd10m_0h' not in df.columns:
        return ['north'] * len(df)
    
    wind_dir = df['wd10m_0h'].fillna(180)
    directions = []
    for wd in wind_dir:
        if wd < 45 or wd >= 315: directions.append('north')
        elif wd < 90: directions.append('northeast')
        elif wd < 135: directions.append('east')
        elif wd < 180: directions.append('southeast')
        elif wd < 225: directions.append('south')
        elif wd < 270: directions.append('southwest')
        elif wd < 315: directions.append('west')
        else: directions.append('northwest')
    return directions

def create_summary_report(area_results, speed_results, direction_results):
    """종합 보고서 생성"""
    print("\\n" + "=" * 60)
    print("🏆 모델 성능 종합 보고서")
    print("=" * 60)
    
    # 면적 모델
    print("\\n🔥 면적 모델 (회귀)")
    print("-" * 30)
    for model, result in area_results.items():
        print(f"{model:25} | R²: {result['r2']:.4f} | RMSE: {result['rmse']:.2f} ha")
    
    if area_results:
        best_area = max(area_results.items(), key=lambda x: x[1]['r2'])
        print(f"\\n🏆 최고 면적 모델: {best_area[0]} (R²: {best_area[1]['r2']:.4f})")
    
    # 속도 모델
    print("\\n⚡ 속도 모델 (분류)")
    print("-" * 30)
    for model, result in speed_results.items():
        print(f"{model:25} | 정확도: {result['accuracy']:.4f}")
    
    if speed_results:
        best_speed = max(speed_results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\\n🏆 최고 속도 모델: {best_speed[0]} (정확도: {best_speed[1]['accuracy']:.4f})")
    
    # 방향 모델
    print("\\n🧭 방향 모델 (분류)")
    print("-" * 30)
    for model, result in direction_results.items():
        print(f"{model:25} | 정확도: {result['accuracy']:.4f} ({result['num_classes']}방향)")
    
    if direction_results:
        best_direction = max(direction_results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\\n🏆 최고 방향 모델: {best_direction[0]} (정확도: {best_direction[1]['accuracy']:.4f})")
    
    print("\\n" + "=" * 60)

def main():
    """메인"""
    print("🎯 종합 모델 성능 테스트")
    print("=" * 50)
    
    # 데이터 로드
    fire_df = load_test_data()
    
    # 각 모델 테스트
    area_results = test_area_model(fire_df)
    speed_results = test_speed_model(fire_df)
    direction_results = test_direction_model(fire_df)
    
    # 종합 보고서
    create_summary_report(area_results, speed_results, direction_results)

if __name__ == "__main__":
    main()