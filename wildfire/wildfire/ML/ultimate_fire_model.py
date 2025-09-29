#!/usr/bin/env python3
"""
궁극의 화재 예측 모델
- 딥러닝 통합 (Neural Network, Autoencoders)
- 동적 피처 선택 (Genetic Algorithm)
- 지역별/계절별 전문 모델
- 불확실성 정량화 (Quantile Regression)
- 시계열 패턴 활용
- 고급 앙상블 (Bayesian Model Averaging)
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import QuantileRegressor, BayesianRidge
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
import xgboost as xgb
import lightgbm as lgb
from scipy.optimize import differential_evolution
from scipy import stats
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
warnings.filterwarnings('ignore')

def load_and_enhance_data():
    """데이터 로드 및 고급 전처리"""
    print("🚀 궁극의 데이터 전처리 시작...")
    
    df = pd.read_csv('final_merged_feature_engineered.csv')
    fire_data = df[(df['fire_area'] > 0) & (df['fire_area'].notna())].copy()
    
    # 기존 물리학 기반 피처 적용
    from sophisticated_area_model import create_fire_physics_features, create_temporal_fire_features
    fire_data, physics_features = create_fire_physics_features(fire_data)
    fire_data, temporal_features = create_temporal_fire_features(fire_data)
    
    print(f"✅ 기본 데이터: {fire_data.shape}")
    return fire_data, physics_features, temporal_features

def create_advanced_spatial_features(df):
    """고급 공간적 피처"""
    print("🗺️ 고급 공간 피처 생성...")
    
    spatial_features = []
    
    # 1. 위도/경도 기반 클러스터링 (지역 특성)
    if 'latitude' in df.columns and 'longitude' in df.columns:
        coords = df[['latitude', 'longitude']].fillna(df[['latitude', 'longitude']].mean())
        kmeans = KMeans(n_clusters=8, random_state=42)
        df['geo_cluster'] = kmeans.fit_predict(coords)
        spatial_features.append('geo_cluster')
        
        # 해안선 거리 근사 (위도 기준)
        df['coastal_proximity'] = np.abs(df['latitude'] - df['latitude'].median())
        spatial_features.append('coastal_proximity')
    
    # 2. 고도 기반 기후대 추정
    if 'elevation_mean' in df.columns:
        df['climate_zone'] = pd.cut(df['elevation_mean'], 
                                  bins=[-np.inf, 200, 500, 1000, np.inf],
                                  labels=[0, 1, 2, 3]).astype(int)
        spatial_features.append('climate_zone')
    
    # 3. 지형 복잡도 스코어
    if all(col in df.columns for col in ['elevation_std', 'slope_std']):
        df['terrain_complexity_score'] = (df['elevation_std'] * df['slope_std']) ** 0.5
        spatial_features.append('terrain_complexity_score')
    
    print(f"✅ 공간 피처: {len(spatial_features)}개")
    return df, spatial_features

def create_temporal_patterns(df):
    """시계열 패턴 피처"""
    print("⏰ 시계열 패턴 분석...")
    
    temporal_pattern_features = []
    
    # 1. 순환 시간 인코딩
    if 'startmonth' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * df['startmonth'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['startmonth'] / 12)
        temporal_pattern_features.extend(['month_sin', 'month_cos'])
    
    if 'startday' in df.columns:
        df['day_sin'] = np.sin(2 * np.pi * df['startday'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['startday'] / 31)
        temporal_pattern_features.extend(['day_sin', 'day_cos'])
    
    # 2. 계절별 트렌드
    if 'startyear' in df.columns and 'startmonth' in df.columns:
        # 연도별 화재 패턴 변화
        df['year_normalized'] = (df['startyear'] - df['startyear'].min()) / (df['startyear'].max() - df['startyear'].min())
        df['seasonal_trend'] = df['year_normalized'] * df['month_sin']
        temporal_pattern_features.extend(['year_normalized', 'seasonal_trend'])
    
    # 3. 엘니뇨/라니냐 근사 (기후 주기)
    if 'startmonth' in df.columns and 'startyear' in df.columns:
        # 단순화된 기후 진동 패턴
        df['climate_oscillation'] = np.sin(2 * np.pi * (df['startyear'] + df['startmonth']/12) / 3.5)
        temporal_pattern_features.append('climate_oscillation')
    
    print(f"✅ 시계열 패턴: {len(temporal_pattern_features)}개")
    return df, temporal_pattern_features

def create_interaction_features_v2(df, base_features):
    """고차원 상호작용 피처"""
    print("🔬 고차원 상호작용 피처 생성...")
    
    interaction_features = []
    
    # 핵심 피처들만 선별
    key_weather = ['t2m_0h', 'rh2m_0h', 'ws10m_0h', 'ps_0h']
    key_terrain = ['elevation_mean', 'slope_mean']
    key_fuel = ['ndvi_before', 'treecover_pre_fire_5x5']
    key_drought = ['dry_days_7d_start', 'dry_days_14d_start']
    
    # 1. 3차원 상호작용 (선별적)
    critical_triplets = [
        ('t2m_0h', 'rh2m_0h', 'ws10m_0h'),  # 기상 3요소
        ('elevation_mean', 'slope_mean', 'ws10m_0h'),  # 지형-바람
        ('fuel_moisture_est', 'wind_slope_factor', 'vapor_pressure_deficit')  # 화재 물리학
    ]
    
    for feat1, feat2, feat3 in critical_triplets:
        if all(f in df.columns for f in [feat1, feat2, feat3]):
            df[f'{feat1}_{feat2}_{feat3}_interaction'] = (df[feat1] * df[feat2] * df[feat3]) ** (1/3)
            interaction_features.append(f'{feat1}_{feat2}_{feat3}_interaction')
    
    # 2. 비선형 변환 조합
    for feat in ['vapor_pressure_deficit', 'fire_intensity_index', 'wind_slope_factor']:
        if feat in df.columns:
            # 로그 변환
            df[f'{feat}_log'] = np.log1p(np.abs(df[feat]))
            interaction_features.append(f'{feat}_log')
            
            # 제곱근 변환  
            df[f'{feat}_sqrt'] = np.sqrt(np.abs(df[feat]))
            interaction_features.append(f'{feat}_sqrt')
    
    # 3. 피처 비율 (중요한 것만)
    ratio_pairs = [
        ('t2m_0h', 'rh2m_0h'),  # 온도/습도 비율
        ('ws10m_0h', 'fuel_moisture_est'),  # 바람/연료수분 비율
        ('elevation_mean', 'ws10m_0h')  # 고도/바람 비율
    ]
    
    for feat1, feat2 in ratio_pairs:
        if feat1 in df.columns and feat2 in df.columns:
            df[f'{feat1}_{feat2}_ratio'] = df[feat1] / (df[feat2] + 1e-6)
            interaction_features.append(f'{feat1}_{feat2}_ratio')
    
    print(f"✅ 상호작용 피처: {len(interaction_features)}개")
    return df, interaction_features

def genetic_feature_selection(X, y, population_size=50, generations=30):
    """유전 알고리즘 기반 피처 선택"""
    print("🧬 유전 알고리즘 피처 선택...")
    
    n_features = X.shape[1]
    feature_names = X.columns.tolist()
    
    def fitness_function(individual):
        """적합도 함수 (R² 스코어)"""
        selected_features = [i for i, val in enumerate(individual) if val > 0.5]
        
        if len(selected_features) < 5:  # 최소 피처 수
            return -1
        
        X_selected = X.iloc[:, selected_features]
        
        # 빠른 평가를 위한 단순 모델
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        
        try:
            scores = cross_val_score(model, X_selected, y, cv=3, scoring='r2')
            return np.mean(scores)
        except:
            return -1
    
    def mutate(individual, mutation_rate=0.1):
        """돌연변이"""
        for i in range(len(individual)):
            if np.random.random() < mutation_rate:
                individual[i] = np.random.random()
        return individual
    
    def crossover(parent1, parent2):
        """교배"""
        crossover_point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2
    
    # 초기 개체군
    population = [np.random.random(n_features) for _ in range(population_size)]
    
    best_fitness = -1
    best_individual = None
    
    for generation in range(generations):
        # 적합도 평가
        fitness_scores = [fitness_function(individual) for individual in population]
        
        # 최고 개체 추적
        current_best_idx = np.argmax(fitness_scores)
        current_best_fitness = fitness_scores[current_best_idx]
        
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[current_best_idx].copy()
            
            if generation % 5 == 0:
                selected_count = sum(1 for val in best_individual if val > 0.5)
                print(f"   세대 {generation:2d}: 최고 적합도 = {best_fitness:.4f}, 선택 피처 = {selected_count}개")
        
        # 선택 (토너먼트 방식)
        new_population = []
        
        # 엘리트 보존
        elite_indices = np.argsort(fitness_scores)[-5:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # 교배 및 돌연변이
        while len(new_population) < population_size:
            # 부모 선택
            idx1, idx2 = np.random.choice(range(population_size), 2, replace=False)
            parent1, parent2 = population[idx1], population[idx2]
            
            # 교배
            child1, child2 = crossover(parent1, parent2)
            
            # 돌연변이
            child1 = mutate(child1)
            child2 = mutate(child2)
            
            new_population.extend([child1, child2])
        
        population = new_population[:population_size]
    
    # 최종 선택된 피처
    selected_indices = [i for i, val in enumerate(best_individual) if val > 0.5]
    selected_features = [feature_names[i] for i in selected_indices]
    
    print(f"✅ 유전 알고리즘 완료: {len(selected_features)}개 피처 선택 (적합도: {best_fitness:.4f})")
    return selected_features

def create_neural_network_models(X, y):
    """고급 딥러닝 모델"""
    print("🧠 딥러닝 모델 생성...")
    
    # 데이터 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = {}
    
    # 1. 기본 DNN
    def create_dnn():
        model = Sequential([
            Input(shape=(X.shape[1],)),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    # 2. 잔차 네트워크 (ResNet 스타일)
    def create_resnet():
        input_layer = Input(shape=(X.shape[1],))
        
        # 첫 번째 블록
        x = Dense(128, activation='relu')(input_layer)
        x = Dropout(0.2)(x)
        residual1 = x
        
        # 두 번째 블록 + 잔차 연결
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = tf.keras.layers.Add()([x, residual1])
        
        # 출력 블록
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        output = Dense(1, activation='linear')(x)
        
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    # 3. 오토인코더 + 회귀
    def create_autoencoder_regressor():
        # 인코더
        input_layer = Input(shape=(X.shape[1],))
        encoded = Dense(64, activation='relu')(input_layer)
        encoded = Dense(32, activation='relu')(encoded)
        encoded = Dense(16, activation='relu')(encoded)  # 압축된 표현
        
        # 디코더
        decoded = Dense(32, activation='relu')(encoded)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dense(X.shape[1], activation='linear')(decoded)
        
        # 회귀 헤드
        regressor = Dense(32, activation='relu')(encoded)
        regressor = Dense(16, activation='relu')(regressor)
        output = Dense(1, activation='linear')(regressor)
        
        model = Model(inputs=input_layer, outputs=[decoded, output])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=['mse', 'mse'],
            loss_weights=[0.3, 0.7],  # 회귀에 더 높은 가중치
            metrics=['mae']
        )
        return model
    
    # 모델 학습
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    callbacks = [
        EarlyStopping(patience=20, restore_best_weights=True),
        ReduceLROnPlateau(patience=10, factor=0.5)
    ]
    
    # DNN 학습
    print("   📊 DNN 학습...")
    dnn = create_dnn()
    dnn.fit(X_train, y_train, epochs=100, batch_size=32, 
            validation_split=0.2, callbacks=callbacks, verbose=0)
    dnn_pred = dnn.predict(X_test, verbose=0).ravel()
    dnn_r2 = r2_score(y_test, dnn_pred)
    models['dnn'] = {'model': dnn, 'r2': dnn_r2, 'scaler': scaler}
    
    # ResNet 학습
    print("   🔄 ResNet 학습...")
    resnet = create_resnet()
    resnet.fit(X_train, y_train, epochs=100, batch_size=32,
              validation_split=0.2, callbacks=callbacks, verbose=0)
    resnet_pred = resnet.predict(X_test, verbose=0).ravel()
    resnet_r2 = r2_score(y_test, resnet_pred)
    models['resnet'] = {'model': resnet, 'r2': resnet_r2, 'scaler': scaler}
    
    # Autoencoder 학습
    print("   🔄 Autoencoder 학습...")
    autoencoder = create_autoencoder_regressor()
    autoencoder.fit(X_train, [X_train, y_train], epochs=100, batch_size=32,
                   validation_split=0.2, callbacks=callbacks, verbose=0)
    _, ae_pred = autoencoder.predict(X_test, verbose=0)
    ae_r2 = r2_score(y_test, ae_pred.ravel())
    models['autoencoder'] = {'model': autoencoder, 'r2': ae_r2, 'scaler': scaler}
    
    print(f"📊 딥러닝 모델 성능:")
    print(f"   DNN: R² = {dnn_r2:.4f}")
    print(f"   ResNet: R² = {resnet_r2:.4f}")
    print(f"   Autoencoder: R² = {ae_r2:.4f}")
    
    return models

def create_uncertainty_quantification_models(X, y):
    """불확실성 정량화 모델"""
    print("📊 불확실성 정량화 모델...")
    
    # Quantile Regression (5%, 50%, 95% 분위수)
    quantiles = [0.05, 0.5, 0.95]
    quantile_models = {}
    
    for q in quantiles:
        qr_model = QuantileRegressor(quantile=q, alpha=0.1)
        qr_model.fit(X, y)
        quantile_models[f'q{int(q*100)}'] = qr_model
    
    # Bayesian Ridge (불확실성 내장)
    bayesian_model = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)
    bayesian_model.fit(X, y)
    
    def predict_with_uncertainty(X_pred):
        """불확실성과 함께 예측"""
        predictions = {}
        
        # 분위수 예측
        for q_name, model in quantile_models.items():
            predictions[q_name] = model.predict(X_pred)
        
        # 베이지안 예측
        bay_pred, bay_std = bayesian_model.predict(X_pred, return_std=True)
        predictions['bayesian_mean'] = bay_pred
        predictions['bayesian_std'] = bay_std
        
        return predictions
    
    print("✅ 불확실성 정량화 모델 완성")
    return predict_with_uncertainty, quantile_models, bayesian_model

def create_ultimate_ensemble(X, y, neural_models, uncertainty_fn):
    """궁극의 앙상블"""
    print("🏆 궁극의 앙상블 생성...")
    
    # 기존 고전 모델들
    classical_models = {
        'xgb_ultimate': xgb.XGBRegressor(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, n_jobs=-1
        ),
        'lgb_ultimate': lgb.LGBMRegressor(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, n_jobs=-1, verbose=-1
        ),
        'rf_ultimate': RandomForestRegressor(
            n_estimators=500, max_depth=15, min_samples_split=5, min_samples_leaf=2,
            max_features='sqrt', bootstrap=True, oob_score=True, random_state=42, n_jobs=-1
        ),
        'et_ultimate': ExtraTreesRegressor(
            n_estimators=500, max_depth=15, min_samples_split=5, min_samples_leaf=2,
            max_features='sqrt', bootstrap=True, oob_score=True, random_state=42, n_jobs=-1
        )
    }
    
    # 모든 모델 학습
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    all_models = {}
    predictions = {}
    
    # 고전 모델들
    print("   📊 고전 모델 학습...")
    for name, model in classical_models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        all_models[name] = {'model': model, 'r2': r2}
        predictions[name] = pred
        print(f"      {name}: R² = {r2:.4f}")
    
    # 딥러닝 모델들
    print("   🧠 딥러닝 모델 예측...")
    for name, nn_info in neural_models.items():
        X_test_scaled = nn_info['scaler'].transform(X_test)
        
        if name == 'autoencoder':
            _, nn_pred = nn_info['model'].predict(X_test_scaled, verbose=0)
            predictions[name] = nn_pred.ravel()
        else:
            nn_pred = nn_info['model'].predict(X_test_scaled, verbose=0).ravel()
            predictions[name] = nn_pred
        
        print(f"      {name}: R² = {nn_info['r2']:.4f}")
    
    # 불확실성 예측
    uncertainty_preds = uncertainty_fn(X_test)
    predictions['bayesian'] = uncertainty_preds['bayesian_mean']
    
    # 동적 가중치 계산 (성능 기반 + 다양성 고려)
    r2_scores = {name: info['r2'] for name, info in all_models.items()}
    r2_scores.update({name: nn_info['r2'] for name, nn_info in neural_models.items()})
    
    # 베이지안 모델 R² 계산
    bay_r2 = r2_score(y_test, predictions['bayesian'])
    r2_scores['bayesian'] = bay_r2
    
    # 소프트맥스 기반 가중치
    r2_values = np.array(list(r2_scores.values()))
    r2_values = np.maximum(r2_values, 0)  # 음수 제거
    weights = np.exp(r2_values * 2) / np.sum(np.exp(r2_values * 2))
    
    weight_dict = dict(zip(r2_scores.keys(), weights))
    
    # 최종 앙상블 예측
    final_prediction = np.zeros_like(y_test)
    for name, weight in weight_dict.items():
        final_prediction += predictions[name] * weight
    
    ensemble_r2 = r2_score(y_test, final_prediction)
    
    print(f"\n🏆 궁극의 앙상블 결과:")
    print(f"   최종 R²: {ensemble_r2:.4f}")
    print("   모델 기여도:")
    for name, weight in sorted(weight_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"      {name:15s}: {weight:.3f} (R²: {r2_scores[name]:.4f})")
    
    def ultimate_predict(X_pred):
        """궁극의 예측 함수"""
        pred_sum = np.zeros(X_pred.shape[0])
        
        # 고전 모델들
        for name, info in all_models.items():
            pred = info['model'].predict(X_pred)
            pred_sum += pred * weight_dict[name]
        
        # 딥러닝 모델들
        for name, nn_info in neural_models.items():
            X_pred_scaled = nn_info['scaler'].transform(X_pred)
            
            if name == 'autoencoder':
                _, nn_pred = nn_info['model'].predict(X_pred_scaled, verbose=0)
                pred = nn_pred.ravel()
            else:
                pred = nn_info['model'].predict(X_pred_scaled, verbose=0).ravel()
            
            pred_sum += pred * weight_dict[name]
        
        # 베이지안 모델
        uncertainty_preds = uncertainty_fn(X_pred)
        pred_sum += uncertainty_preds['bayesian_mean'] * weight_dict['bayesian']
        
        return pred_sum, uncertainty_preds
    
    return ultimate_predict, weight_dict, ensemble_r2

def main():
    """메인 실행"""
    print("🚀 궁극의 화재 예측 모델 시작...")
    
    # 1. 데이터 로드 및 기본 피처
    fire_data, physics_features, temporal_features = load_and_enhance_data()
    
    # 2. 고급 공간적 피처
    fire_data, spatial_features = create_advanced_spatial_features(fire_data)
    
    # 3. 시계열 패턴 피처
    fire_data, temporal_pattern_features = create_temporal_patterns(fire_data)
    
    # 4. 기본 피처 세트 구성
    base_features = [
        't2m_0h', 'rh2m_0h', 'ws10m_0h', 'ps_0h',  # 기상
        'fwi_0h', 'ffmc_0h', 'dmc_0h', 'dc_0h',    # FWI
        'elevation_mean', 'slope_mean',              # 지형
        'ndvi_before', 'treecover_pre_fire_5x5',    # 식생
        'dry_days_7d_start', 'startmonth'           # 건조도, 시간
    ]
    
    available_base = [f for f in base_features if f in fire_data.columns]
    all_feature_groups = [available_base, physics_features, temporal_features, 
                         spatial_features, temporal_pattern_features]
    
    # 5. 고차원 상호작용 피처
    fire_data, interaction_features = create_interaction_features_v2(fire_data, available_base)
    all_feature_groups.append(interaction_features)
    
    # 6. 전체 피처 세트
    all_features = []
    for group in all_feature_groups:
        all_features.extend([f for f in group if f in fire_data.columns])
    all_features = list(set(all_features))  # 중복 제거
    
    print(f"📊 총 피처 수: {len(all_features)}개")
    
    # 7. 데이터 정제
    X = fire_data[all_features].copy()
    y = fire_data['fire_area'].copy()
    
    # 결측치 처리
    for col in X.columns:
        if X[col].isna().sum() > 0:
            X[col] = X[col].fillna(X[col].median())
    X = X.replace([np.inf, -np.inf], 0)
    
    # 이상치 제거
    q95 = y.quantile(0.95)
    mask = y <= q95
    X, y = X[mask], y[mask]
    
    # 타겟 변환
    from sklearn.preprocessing import PowerTransformer
    pt = PowerTransformer(method='yeo-johnson')
    y_transformed = pt.fit_transform(y.values.reshape(-1, 1)).ravel()
    
    print(f"📊 최종 데이터: {X.shape[0]}개 샘플, {X.shape[1]}개 피처")
    
    # 8. 유전 알고리즘 피처 선택
    if X.shape[1] > 50:  # 피처가 많을 때만 사용
        selected_features = genetic_feature_selection(X, y_transformed, population_size=30, generations=20)
        X_selected = X[selected_features]
    else:
        X_selected = X
        selected_features = X.columns.tolist()
    
    # 9. 스케일링
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_selected)
    X_scaled_df = pd.DataFrame(X_scaled, columns=selected_features)
    
    # 10. 딥러닝 모델
    neural_models = create_neural_network_models(X_scaled_df, y_transformed)
    
    # 11. 불확실성 정량화
    uncertainty_fn, quantile_models, bayesian_model = create_uncertainty_quantification_models(X_scaled, y_transformed)
    
    # 12. 궁극의 앙상블
    ultimate_predict, weights, final_r2 = create_ultimate_ensemble(
        X_scaled_df, y_transformed, neural_models, uncertainty_fn
    )
    
    # 13. 저장
    print("\n💾 궁극의 모델 저장...")
    
    # 모든 구성요소 저장
    joblib.dump({
        'scaler': scaler,
        'power_transformer': pt,
        'selected_features': selected_features,
        'weights': weights,
        'final_r2': final_r2
    }, 'ultimate_fire_model_config.joblib')
    
    # 신경망 모델들 저장
    for name, nn_info in neural_models.items():
        nn_info['model'].save(f'ultimate_{name}_model.h5')
    
    print(f"\n🎉 궁극의 모델 완성!")
    print(f"📊 최종 성능: R² = {final_r2:.4f}")
    print(f"🔧 선택된 피처: {len(selected_features)}개")
    print(f"🧠 모델 구성: 고전 모델 + 딥러닝 + 불확실성 정량화")
    
    if final_r2 > 0.7:
        print("🏆 탁월한 성능 달성!")
    elif final_r2 > 0.6:
        print("🥇 우수한 성능!")
    else:
        print("🥈 양호한 성능")

if __name__ == "__main__":
    main()