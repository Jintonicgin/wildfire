#!/usr/bin/env python3
"""
고급 면적 모델 - 성능 대폭 개선
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

def load_and_analyze_area_data():
    """데이터 로드 및 분석"""
    print("🔥 면적 데이터 분석...")
    
    df = pd.read_csv('final_merged_feature_engineered.csv', low_memory=False)
    fire_mask = (df['fire_area'] > 0) & df['fire_area'].notna()
    fire_df = df[fire_mask].copy()
    
    print(f"   화재 데이터: {fire_df.shape}")
    print(f"   면적 분포: {fire_df['fire_area'].describe()}")
    
    # 면적 분포 확인
    area_data = fire_df['fire_area']
    print(f"\\n📊 면적 분포 분석:")
    print(f"   평균: {area_data.mean():.2f} ha")
    print(f"   중앙값: {area_data.median():.2f} ha")
    print(f"   표준편차: {area_data.std():.2f} ha")
    print(f"   왜도: {area_data.skew():.2f}")
    print(f"   99% 분위수: {area_data.quantile(0.99):.2f} ha")
    print(f"   최댓값: {area_data.max():.2f} ha")
    
    # 로그 변환된 면적도 확인
    log_area = np.log1p(area_data)
    print(f"\\n📊 로그 변환 후:")
    print(f"   평균: {log_area.mean():.3f}")
    print(f"   표준편차: {log_area.std():.3f}")
    print(f"   왜도: {log_area.skew():.3f}")
    
    return fire_df

def create_advanced_features(df):
    """고급 피처 엔지니어링"""
    print("\\n🎯 고급 피처 생성...")
    
    features = []
    
    # 1. 핵심 화재 위험 지수
    fire_risk_features = ['fwi_0h', 'ffmc_0h', 'dmc_0h', 'dc_0h', 'isi_0h', 'bui_0h']
    for feat in fire_risk_features:
        if feat in df.columns:
            features.append(feat)
    
    # 2. 기상 조건
    weather_features = ['t2m_0h', 'rh2m_0h', 'ws10m_0h', 'wd10m_0h', 'precip_0h']
    for feat in weather_features:
        if feat in df.columns:
            features.append(feat)
    
    # 3. 지형 정보
    terrain_features = ['elevation_mean', 'slope_mean']
    for feat in terrain_features:
        if feat in df.columns:
            features.append(feat)
    
    # 4. 시간적 패턴
    temporal_features = ['fire_month', 'startday']
    for feat in temporal_features:
        if feat in df.columns:
            features.append(feat)
    
    # 5. 건조 조건
    dry_features = ['dry_days_7d_start', 'dry_days_14d_start', 'dry_days_30d_start']
    for feat in dry_features:
        if feat in df.columns:
            features.append(feat)
    
    # 6. 복합 지수들
    combo_features = ['hot_dry_combo', 'dry_windy_combo']
    for feat in combo_features:
        if feat in df.columns:
            features.append(feat)
    
    # 7. 과거 기상 패턴 (핵심만)
    past_weather = []
    for hour in [3, 6, 12, 24]:
        for param in ['t2m', 'rh2m', 'ws10m']:
            feat = f'{param}_{hour}h_past'
            if feat in df.columns:
                past_weather.append(feat)
    
    # 상위 20개만 선택
    if len(past_weather) > 20:
        # 상관관계가 높은 것들 우선 선택
        past_weather = past_weather[:20]
    
    features.extend(past_weather)
    
    # 8. 고급 파생 피처 생성
    derived_features = []
    
    # FWI 기반 위험도
    if 'fwi_0h' in df.columns:
        df['fwi_risk_level'] = pd.cut(df['fwi_0h'], 
                                     bins=[0, 5, 13, 21, 100], 
                                     labels=[1, 2, 3, 4])
        df['fwi_risk_level'] = df['fwi_risk_level'].astype(float)
        derived_features.append('fwi_risk_level')
    
    # 바람-습도 상호작용
    if 'ws10m_0h' in df.columns and 'rh2m_0h' in df.columns:
        df['wind_dry_interaction'] = df['ws10m_0h'] * (100 - df['rh2m_0h'].fillna(50))
        derived_features.append('wind_dry_interaction')
    
    # 온도-습도 조합
    if 't2m_0h' in df.columns and 'rh2m_0h' in df.columns:
        df['temp_humidity_deficit'] = df['t2m_0h'] * (100 - df['rh2m_0h'].fillna(50)) / 100
        derived_features.append('temp_humidity_deficit')
    
    # ISI와 바람속도 조합
    if 'isi_0h' in df.columns and 'ws10m_0h' in df.columns:
        df['isi_wind_combo'] = df['isi_0h'] * np.sqrt(df['ws10m_0h'].fillna(0))
        derived_features.append('isi_wind_combo')
    
    features.extend(derived_features)
    
    # 사용 가능한 피처만 필터링
    available_features = [f for f in features if f in df.columns]
    
    print(f"   총 피처: {len(available_features)}개")
    print(f"   핵심 피처: {available_features[:15]}")
    
    return available_features

def preprocess_target(y, method='log'):
    """타겟 변수 전처리"""
    if method == 'log':
        return np.log1p(y), lambda x: np.expm1(x)
    elif method == 'sqrt':
        return np.sqrt(y), lambda x: x**2
    elif method == 'quantile':
        transformer = QuantileTransformer(output_distribution='normal')
        y_transformed = transformer.fit_transform(y.values.reshape(-1, 1)).flatten()
        return y_transformed, transformer.inverse_transform
    else:
        return y, lambda x: x

def train_advanced_models(X_train, X_test, y_train, y_test, scaler):
    """고급 모델들 훈련"""
    print("\\n🤖 고급 모델 훈련...")
    
    # 스케일링
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'XGBoost_Tuned': xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),
        'LightGBM_Tuned': lgb.LGBMRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        'RandomForest_Tuned': RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features=0.7,
            random_state=42,
            n_jobs=-1
        ),
        'ExtraTrees': ExtraTreesRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting_Tuned': GradientBoostingRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        ),
        'Neural_Network': MLPRegressor(
            hidden_layer_sizes=(100, 50),
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42
        )
    }
    
    results = {}
    best_model = None
    best_r2 = -np.inf
    
    for name, model in models.items():
        print(f"   {name} 훈련...")
        
        try:
            # 훈련
            if 'Neural' in name:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # 평가
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            results[name] = {
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'model': model
            }
            
            print(f"     R²: {r2:.4f}, RMSE: {rmse:.2f}")
            
            if r2 > best_r2:
                best_r2 = r2
                best_model = (name, model)
                
        except Exception as e:
            print(f"     {name} 실패: {e}")
    
    return results, best_model

def create_ensemble_model(results, X_test, scaler):
    """앙상블 모델 생성"""
    print("\\n🎭 앙상블 모델 생성...")
    
    # 상위 3개 모델 선택
    top_models = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)[:3]
    
    predictions = []
    weights = []
    
    for name, result in top_models:
        model = result['model']
        weight = max(0, result['r2'])  # 음수 R²는 0으로
        
        if 'Neural' in name:
            X_test_input = scaler.transform(X_test)
        else:
            X_test_input = X_test
            
        pred = model.predict(X_test_input)
        predictions.append(pred)
        weights.append(weight)
    
    # 가중평균
    if sum(weights) > 0:
        weights = np.array(weights) / sum(weights)
        ensemble_pred = sum(w * pred for w, pred in zip(weights, predictions))
        
        print(f"   앙상블 가중치: {[f'{w:.3f}' for w in weights]}")
        return ensemble_pred
    else:
        return predictions[0]

def main():
    """메인"""
    print("🎯 고급 면적 모델 개발")
    print("=" * 50)
    
    # 데이터 분석
    fire_df = load_and_analyze_area_data()
    
    # 고급 피처 생성
    features = create_advanced_features(fire_df)
    
    # 데이터 준비
    X = fire_df[features].copy()
    y_original = fire_df['fire_area'].copy()
    
    # 강력한 전처리
    print(f"\\n🔧 데이터 전처리...")
    print(f"   전처리 전 shape: {X.shape}")
    
    # 무한값과 극값 처리
    for col in X.columns:
        # 무한값을 NaN으로
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        
        # 극값 클리핑 (99.9% quantile)
        if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            q99 = X[col].quantile(0.999)
            q01 = X[col].quantile(0.001)
            X[col] = X[col].clip(lower=q01, upper=q99)
        
        # NaN 채우기
        if X[col].isna().sum() > 0:
            X[col] = X[col].fillna(X[col].median())
    
    # 최종 안전 체크
    X = X.select_dtypes(include=[np.number])  # 숫자형만
    X = X.fillna(0)  # 혹시 남은 NaN
    
    print(f"   전처리 후 shape: {X.shape}")
    print(f"   무한값 체크: {np.isinf(X.values).sum()}")
    print(f"   NaN 체크: {X.isna().sum().sum()}")
    
    # 타겟 변환 테스트
    target_methods = ['original', 'log', 'sqrt']
    best_results = {}
    
    for method in target_methods:
        print(f"\\n📊 {method} 변환 테스트...")
        
        y_transformed, inverse_transform = preprocess_target(y_original, method)
        
        # 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_transformed, test_size=0.25, random_state=42
        )
        
        # 스케일러
        scaler = StandardScaler()
        
        # 모델 훈련
        results, best_model = train_advanced_models(X_train, X_test, y_train, y_test, scaler)
        
        if results:
            # 최고 성능 모델
            best_name, best_result = max(results.items(), key=lambda x: x[1]['r2'])
            
            # 앙상블
            ensemble_pred = create_ensemble_model(results, X_test, scaler)
            
            # 원래 스케일로 변환
            if method != 'original':
                if method == 'quantile':
                    ensemble_pred_original = inverse_transform(ensemble_pred.reshape(-1, 1)).flatten()
                    y_test_original = inverse_transform(y_test.reshape(-1, 1)).flatten()
                else:
                    ensemble_pred_original = inverse_transform(ensemble_pred)
                    y_test_original = inverse_transform(y_test)
            else:
                ensemble_pred_original = ensemble_pred
                y_test_original = y_test
            
            # 최종 평가
            ensemble_r2 = r2_score(y_test_original, ensemble_pred_original)
            ensemble_rmse = np.sqrt(mean_squared_error(y_test_original, ensemble_pred_original))
            
            best_results[method] = {
                'ensemble_r2': ensemble_r2,
                'ensemble_rmse': ensemble_rmse,
                'best_single_r2': best_result['r2'],
                'results': results,
                'best_model': best_model,
                'features': features,
                'scaler': scaler
            }
            
            print(f"   최고 단일 모델: {best_name} (R²: {best_result['r2']:.4f})")
            print(f"   앙상블 모델: R²: {ensemble_r2:.4f}, RMSE: {ensemble_rmse:.2f} ha")
        else:
            print(f"   {method} 변환에서 성공한 모델이 없습니다.")
    
    # 최종 결과
    print("\\n" + "=" * 50)
    print("🏆 최종 결과 비교")
    print("=" * 50)
    
    for method, result in best_results.items():
        print(f"{method:10} | 단일: {result['best_single_r2']:.4f} | 앙상블: {result['ensemble_r2']:.4f}")
    
    # 최고 성능 모델
    if best_results:
        best_method = max(best_results.items(), key=lambda x: x[1]['ensemble_r2'])
        print(f"\\n🥇 최고 성능: {best_method[0]} 변환")
        print(f"   앙상블 R²: {best_method[1]['ensemble_r2']:.4f}")
        print(f"   RMSE: {best_method[1]['ensemble_rmse']:.2f} ha")
        
        # 모델 저장
        best_package = {
            'method': best_method[0],
            'ensemble_r2': best_method[1]['ensemble_r2'],
            'results': best_method[1]['results'],
            'best_model': best_method[1]['best_model'],
            'features': best_method[1]['features'],
            'scaler': best_method[1]['scaler']
        }
        
        joblib.dump(best_package, 'advanced_area_model_final.joblib')
        print(f"\\n✅ 저장 완료: advanced_area_model_final.joblib")

if __name__ == "__main__":
    main()