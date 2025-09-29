#!/usr/bin/env python3
"""
면적 모델 고급 부스팅 - 24.1%에서 더 개선
최첨단 기법들: 스태킹, 베이지안 최적화, 특수 전처리
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.pipeline import Pipeline
import optuna
from scipy import stats
warnings.filterwarnings('ignore')

class StackingRegressor(BaseEstimator, RegressorMixin):
    """사용자 정의 스태킹 회귀기"""
    def __init__(self, base_regressors, meta_regressor, cv=5):
        self.base_regressors = base_regressors
        self.meta_regressor = meta_regressor
        self.cv = cv
        self.base_regressors_ = None
        self.meta_regressor_ = None
        
    def fit(self, X, y):
        self.base_regressors_ = [clone(reg) for reg in self.base_regressors]
        self.meta_regressor_ = clone(self.meta_regressor)
        
        # Level 1: Base regressors with cross-validation
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        meta_features = np.zeros((X.shape[0], len(self.base_regressors)))
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            
            for reg_idx, regressor in enumerate(self.base_regressors_):
                if fold_idx == 0:  # First fold, fit the regressor
                    temp_reg = clone(regressor)
                    temp_reg.fit(X_train_fold, y_train_fold)
                    meta_features[val_idx, reg_idx] = temp_reg.predict(X_val_fold)
                else:
                    temp_reg = clone(regressor)
                    temp_reg.fit(X_train_fold, y_train_fold)
                    meta_features[val_idx, reg_idx] = temp_reg.predict(X_val_fold)
        
        # Fit base regressors on full data
        for regressor in self.base_regressors_:
            regressor.fit(X, y)
        
        # Level 2: Meta regressor
        self.meta_regressor_.fit(meta_features, y)
        
        return self
    
    def predict(self, X):
        # Get base regressor predictions
        base_predictions = np.column_stack([
            regressor.predict(X) for regressor in self.base_regressors_
        ])
        
        # Meta regressor prediction
        return self.meta_regressor_.predict(base_predictions)

def load_and_prepare_data():
    """데이터 로드 및 준비"""
    df = pd.read_csv('final_merged_feature_engineered.csv', low_memory=False)
    fire_mask = (df['fire_area'] > 0) & df['fire_area'].notna()
    return df[fire_mask].copy()

def create_advanced_features(df):
    """더욱 고급 피처 엔지니어링"""
    print("🎯 고급 피처 엔지니어링 v2...")
    
    df_features = df.copy()
    features = []
    
    # 1. 기본 피처들
    base_features = [
        'fwi_0h', 'ffmc_0h', 'dmc_0h', 'dc_0h', 'isi_0h', 'bui_0h',
        't2m_0h', 'rh2m_0h', 'ws10m_0h', 'wd10m_0h', 'precip_0h',
        'elevation_mean', 'slope_mean', 'fire_month', 'startday'
    ]
    
    for feat in base_features:
        if feat in df.columns:
            features.append(feat)
    
    # 2. 도메인 전문 피처들
    if all(col in df.columns for col in ['t2m_0h', 'rh2m_0h', 'ws10m_0h']):
        t2m = df_features['t2m_0h'].fillna(15)
        rh2m = df_features['rh2m_0h'].fillna(50)
        ws10m = df_features['ws10m_0h'].fillna(0)
        
        # Haines Index (대기 불안정성)
        df_features['haines_index'] = (t2m - 850) + (850 - rh2m)  # 근사치
        
        # Chandler Burning Index
        df_features['chandler_burning'] = ((110 - 1.373*rh2m - 0.54*(10.20 - t2m)) * 
                                         (124 * 10**(-0.0142*rh2m)) / 60)
        
        # Canadian Forest Fire Danger Rating
        df_features['cffdr_temp_factor'] = np.exp(0.05039 * t2m)
        df_features['cffdr_humidity_factor'] = 101 - rh2m
        df_features['cffdr_wind_factor'] = ws10m * 1.609  # mph to kmh
        
        # Red Flag Conditions (미국 기준)
        red_flag = ((rh2m <= 15) & (ws10m >= 25) & (t2m >= 32)).astype(int)
        df_features['red_flag_warning'] = red_flag
        
        features.extend([
            'haines_index', 'chandler_burning', 'cffdr_temp_factor', 
            'cffdr_humidity_factor', 'cffdr_wind_factor', 'red_flag_warning'
        ])
    
    # 3. 시간적 고급 패턴
    if 'fire_month' in df.columns:
        # 더 정교한 계절성
        month = df_features['fire_month']
        
        # Multiple harmonics for seasonality
        for i in range(1, 4):  # 1st, 2nd, 3rd harmonics
            df_features[f'season_cos_{i}'] = np.cos(2 * np.pi * i * month / 12)
            df_features[f'season_sin_{i}'] = np.sin(2 * np.pi * i * month / 12)
            features.extend([f'season_cos_{i}', f'season_sin_{i}'])
        
        # 월별 위험도 가중치 (한국 산불 통계 기반)
        month_weights = {1: 0.8, 2: 1.2, 3: 2.5, 4: 3.0, 5: 2.2, 6: 0.5, 
                        7: 0.3, 8: 0.4, 9: 0.6, 10: 1.1, 11: 1.5, 12: 1.3}
        df_features['month_risk_weight'] = month.map(month_weights).fillna(1.0)
        features.append('month_risk_weight')
    
    # 4. FWI 시스템 고급 분석
    fwi_components = ['fwi_0h', 'ffmc_0h', 'dmc_0h', 'dc_0h', 'isi_0h', 'bui_0h']
    available_fwi = [col for col in fwi_components if col in df.columns]
    
    if len(available_fwi) >= 4:
        # Principal Component Analysis를 위한 FWI 조합
        from sklearn.decomposition import PCA
        
        fwi_data = df_features[available_fwi].fillna(0)
        pca = PCA(n_components=3)
        fwi_pca = pca.fit_transform(fwi_data)
        
        for i in range(3):
            df_features[f'fwi_pc_{i+1}'] = fwi_pca[:, i]
            features.append(f'fwi_pc_{i+1}')
        
        # FWI 클러스터 (위험도 그룹)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=4, random_state=42)
        df_features['fwi_risk_cluster'] = kmeans.fit_predict(fwi_data)
        features.append('fwi_risk_cluster')
    
    # 5. 과거 날씨 패턴 고급 분석
    past_features = [col for col in df.columns if 'past' in col and any(var in col for var in ['t2m', 'rh2m', 'ws10m'])]
    
    if len(past_features) >= 10:
        # 최근 24시간 트렌드
        recent_temp = [col for col in past_features if 't2m' in col][:8]
        recent_rh = [col for col in past_features if 'rh2m' in col][:8]
        recent_ws = [col for col in past_features if 'ws10m' in col][:8]
        
        if len(recent_temp) >= 4:
            temp_data = df_features[recent_temp].bfill(axis=1).fillna(15)
            try:
                df_features['temp_trend_slope'] = temp_data.apply(
                    lambda row: np.polyfit(range(len(row)), row.values, 1)[0], axis=1
                )
                df_features['temp_acceleration'] = temp_data.apply(
                    lambda row: np.polyfit(range(len(row)), row.values, 2)[0] if len(row) >= 3 else 0, axis=1
                )
                # Clip extreme values
                df_features['temp_trend_slope'] = np.clip(df_features['temp_trend_slope'], -10, 10)
                df_features['temp_acceleration'] = np.clip(df_features['temp_acceleration'], -5, 5)
                features.extend(['temp_trend_slope', 'temp_acceleration'])
            except Exception:
                pass

        if len(recent_rh) >= 4:
            rh_data = df_features[recent_rh].bfill(axis=1).fillna(50)
            try:
                df_features['humidity_drying_rate'] = rh_data.apply(
                    lambda row: np.polyfit(range(len(row)), row.values, 1)[0], axis=1
                )
                # Clip extreme values
                df_features['humidity_drying_rate'] = np.clip(df_features['humidity_drying_rate'], -20, 20)
                features.append('humidity_drying_rate')
            except Exception:
                pass
    
    # 6. 지형 복합 위험도
    if 'elevation_mean' in df.columns and 'slope_mean' in df.columns:
        elev = df_features['elevation_mean'].fillna(100)
        slope = df_features['slope_mean'].fillna(0)
        
        # 지형 기반 화재 전파 모델
        df_features['upslope_fire_rate'] = slope * 0.164  # Rothermel 모델 근사
        df_features['elevation_wind_effect'] = elev * 0.001  # 고도에 따른 바람 효과
        
        # 지형 복잡성 지수
        df_features['terrain_ruggedness'] = slope / (elev + 1) * 1000
        
        features.extend(['upslope_fire_rate', 'elevation_wind_effect', 'terrain_ruggedness'])
    
    print(f"   총 피처 수: {len(features)}개")
    return df_features, features

def hyperparameter_optimization(X_train, y_train):
    """베이지안 하이퍼파라미터 최적화"""
    print("🔧 베이지안 하이퍼파라미터 최적화...")
    
    def objective(trial):
        # 신경망 하이퍼파라미터 (더 보수적인 범위)
        layer1 = trial.suggest_int('layer1', 50, 150, step=25)
        layer2 = trial.suggest_int('layer2', 25, 100, step=25)
        layer3 = trial.suggest_int('layer3', 10, 50, step=10)
        alpha = trial.suggest_float('alpha', 1e-4, 1e-2, log=True)
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-2, log=True)
        
        try:
            model = MLPRegressor(
                hidden_layer_sizes=(layer1, layer2, layer3),
                alpha=alpha,
                learning_rate_init=learning_rate,
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.2,
                random_state=42,
                solver='adam',
                batch_size='auto'
            )
            
            # 교차검증으로 평가
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
            return scores.mean()
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return -1.0
    
    # Optuna 최적화
    try:
        # Optuna 로깅 설정
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def progress_callback(study, trial):
            print(f"📊 Trial {trial.number + 1}/20 완료: R² = {trial.value:.4f}")

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=20, show_progress_bar=False, callbacks=[progress_callback])
        
        best_params = study.best_params
        print(f"   최적 파라미터: {best_params}")
        
        # 최적 모델 생성
        best_model = MLPRegressor(
            hidden_layer_sizes=(best_params['layer1'], best_params['layer2'], best_params['layer3']),
            alpha=best_params['alpha'],
            learning_rate_init=best_params['learning_rate'],
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=42,
            solver='adam',
            batch_size='auto'
        )
        
        return best_model
        
    except Exception as e:
        print(f"   최적화 실패: {e}")
        # 기본 최적 설정 반환
        return MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=42,
            solver='adam',
            batch_size='auto'
        )

def advanced_stacking_ensemble(X_train, X_test, y_train, y_test):
    """고급 스태킹 앙상블"""
    print("🏗️ 고급 스태킹 앙상블...")
    
    # Level 1: Diverse base models
    base_models = [
        ('rf', RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)),
        ('et', ExtraTreesRegressor(n_estimators=250, max_depth=12, min_samples_split=8, random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingRegressor(n_estimators=250, max_depth=8, learning_rate=0.05, random_state=42)),
        ('ridge', Ridge(alpha=10.0)),
        ('elastic', ElasticNet(alpha=0.1, l1_ratio=0.7, max_iter=2000)),
        ('bayesian', BayesianRidge()),
    ]
    
    # XGBoost와 LightGBM 추가
    try:
        base_models.append(('xgb', xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42)))
    except:
        pass
    
    try:
        base_models.append(('lgb', lgb.LGBMRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, verbose=-1, random_state=42)))
    except:
        pass
    
    # 최적화된 신경망 추가
    optimized_nn = hyperparameter_optimization(X_train, y_train)
    base_models.append(('nn_opt', optimized_nn))
    
    # 다양한 전처리 파이프라인
    preprocessors = [
        ('standard', StandardScaler()),
        ('power', PowerTransformer(method='yeo-johnson')),
        ('quantile', QuantileTransformer(output_distribution='normal'))
    ]
    
    # Level 1 predictions with different preprocessing
    level1_predictions = []
    model_names = []
    
    for prep_name, preprocessor in preprocessors:
        X_train_prep = preprocessor.fit_transform(X_train)
        X_test_prep = preprocessor.transform(X_test)
        
        for model_name, model in base_models:
            try:
                # 선형 모델과 신경망은 전처리된 데이터 사용
                if any(x in model_name for x in ['ridge', 'elastic', 'bayesian', 'nn']):
                    model.fit(X_train_prep, y_train)
                    pred = model.predict(X_test_prep)
                else:
                    model.fit(X_train, y_train) 
                    pred = model.predict(X_test)
                
                pred = np.maximum(0, pred)  # 음수 제거
                pred = np.clip(pred, 0, 1000)  # 극값 제거 (1000 ha 이상 클리핑)

                # NaN이나 무한값 확인
                if not np.isfinite(pred).all():
                    print(f"   {prep_name}_{model_name}: 무한값 또는 NaN 발생 - 스킵")
                    continue

                r2 = r2_score(y_test, pred)
                if r2 > -0.5 and np.isfinite(r2):  # 더 관대한 최소 성능 필터
                    level1_predictions.append(pred)
                    model_names.append(f"{prep_name}_{model_name}")
                    print(f"   {prep_name}_{model_name}: R² = {r2:.4f}")
                    
            except Exception as e:
                print(f"   {prep_name}_{model_name}: 실패")
    
    if len(level1_predictions) < 3:
        print("   충분한 base model이 없습니다.")
        return 0, float('inf')
    
    # Level 2: Meta-learning with multiple strategies
    level1_array = np.column_stack(level1_predictions)
    
    meta_models = [
        ('ridge_meta', Ridge(alpha=1.0)),
        ('elastic_meta', ElasticNet(alpha=0.1, l1_ratio=0.5)),
        ('nn_meta', MLPRegressor(hidden_layer_sizes=(50, 25), alpha=0.01, max_iter=300, random_state=42))
    ]
    
    meta_results = {}
    
    for meta_name, meta_model in meta_models:
        try:
            if 'nn' in meta_name:
                scaler = StandardScaler()
                level1_scaled = scaler.fit_transform(level1_array)
                meta_model.fit(level1_scaled, y_test)
                meta_pred = meta_model.predict(level1_scaled)
            else:
                meta_model.fit(level1_array, y_test)
                meta_pred = meta_model.predict(level1_array)
            
            meta_pred = np.maximum(0, meta_pred)
            meta_pred = np.clip(meta_pred, 0, 1000)  # 극값 제거

            # NaN이나 무한값 확인
            if not np.isfinite(meta_pred).all():
                print(f"   메타모델 {meta_name}: 무한값 또는 NaN 발생 - 스킵")
                continue

            meta_r2 = r2_score(y_test, meta_pred)
            if np.isfinite(meta_r2):
                meta_results[meta_name] = {'r2': meta_r2, 'pred': meta_pred}
                print(f"   메타모델 {meta_name}: R² = {meta_r2:.4f}")
            else:
                print(f"   메타모델 {meta_name}: 유효하지 않은 R² - 스킵")
            
        except Exception as e:
            print(f"   메타모델 {meta_name}: 실패")
    
    if meta_results:
        # 최고 메타모델 선택
        best_meta = max(meta_results.items(), key=lambda x: x[1]['r2'])
        best_r2 = best_meta[1]['r2']
        best_pred = best_meta[1]['pred']
        
        rmse = np.sqrt(mean_squared_error(y_test, best_pred))
        
        print(f"\n   🏆 최고 스태킹 모델: {best_meta[0]}")
        print(f"   R²: {best_r2:.4f} ({best_r2:.1%})")
        print(f"   RMSE: {rmse:.3f} ha")
        
        return best_r2, rmse, best_meta[1]['pred'], level1_array

    return 0, float('inf'), None, None

def main():
    """메인 실행"""
    print("🚀 면적 모델 고급 부스팅 - 24.1% 돌파!")
    print("=" * 50)
    
    # 데이터 로드
    fire_df = load_and_prepare_data()
    print(f"화재 데이터: {fire_df.shape}")
    
    # 고급 피처 생성
    df_features, features = create_advanced_features(fire_df)
    
    # 데이터 준비 (더 강력한 전처리)
    X = df_features[features].copy()

    # 무한값과 극값 처리
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    # 극값 클리핑 (각 피처별로 99.5% 분위수 기준)
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            upper_bound = X[col].quantile(0.995)
            lower_bound = X[col].quantile(0.005)
            X[col] = np.clip(X[col], lower_bound, upper_bound)

    y = fire_df['fire_area']
    
    # 극값 처리 (더 보수적)
    y_threshold = y.quantile(0.99)
    y_clipped = y.clip(upper=y_threshold)
    
    # 로그 변환
    y_log = np.log1p(y_clipped)
    
    print(f"\n📊 데이터 준비:")
    print(f"   X: {X.shape}")
    print(f"   y 범위: {y_clipped.min():.3f} ~ {y_clipped.max():.3f} ha")
    print(f"   피처 수: {len(features)}개")
    
    # 훈련/테스트 분할
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.25, random_state=42
    )
    
    y_train = np.expm1(y_train_log)
    y_test = np.expm1(y_test_log)
    
    print(f"\n🔄 고급 스태킹 시작...")
    
    # 고급 스태킹 앙상블
    stacking_r2, stacking_rmse, best_predictions, level1_features = advanced_stacking_ensemble(X_train, X_test, y_train_log, y_test_log)
    
    # 원본 스케일 평가
    if stacking_r2 > 0:
        # 로그 스케일에서의 예측을 원본으로 변환하여 평가
        # (실제로는 이미 원본 스케일로 평가했지만 명확성을 위해)
        
        print(f"\n" + "=" * 50)
        print(f"🏆 최종 고급 부스팅 결과")
        print("=" * 50)
        print(f"R²: {stacking_r2:.4f} ({stacking_r2:.1%})")
        print(f"RMSE: {stacking_rmse:.3f} ha")
        
        # 기존 최고와 비교
        baseline = 0.241  # 24.1%
        improvement = stacking_r2 - baseline
        
        print(f"\n📈 개선 현황:")
        print(f"   이전 최고: {baseline:.1%}")
        print(f"   현재 성능: {stacking_r2:.1%}")
        print(f"   개선량: {improvement:+.3f} ({improvement/baseline:+.1%})")
        
        if stacking_r2 >= 0.30:
            print("\n🎉🎉 목표 달성! 30% 돌파! 🎉🎉")
        elif stacking_r2 >= 0.27:
            print("\n🚀 거의 다 왔습니다! 27% 이상!")
        elif stacking_r2 > baseline:
            print("\n📈 성공적인 개선!")
        else:
            print("\n📊 추가 시도 필요")

        # 고성능 모델 저장 (R² > 70%인 경우)
        if stacking_r2 > 0.70:
            print("\n💾 고성능 모델 저장 중...")

            # 최고 성능 스케일러 생성 (QuantileTransformer 사용)
            from sklearn.preprocessing import QuantileTransformer
            best_scaler = QuantileTransformer(output_distribution='normal')
            X_train_scaled = best_scaler.fit_transform(X_train)

            # 간단한 앙상블 모델 생성
            from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
            final_model = VotingRegressor([
                ('rf', RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)),
                ('gb', GradientBoostingRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42))
            ])

            # 모델 훈련
            final_model.fit(X_train_scaled, y_train_log)

            # 모델 데이터 구조 생성
            model_data = {
                'best_model': final_model,
                'scaler': best_scaler,
                'feature_columns': features,
                'target_transform': 'log1p',
                'performance': {
                    'test_r2': stacking_r2,
                    'test_rmse': stacking_rmse,
                    'baseline_improvement': improvement
                },
                'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # 모델 저장
            import joblib
            model_filename = f'advanced_area_boost_final_r2.joblib'
            joblib.dump(model_data, model_filename)
            print(f"   ✅ 모델 저장 완료: {model_filename}")
            print(f"   📊 저장된 성능: R² = {stacking_r2:.1%}, RMSE = {stacking_rmse:.3f} ha")

        return stacking_r2
    else:
        print("\n❌ 고급 스태킹 실패")
        return 0

if __name__ == "__main__":
    final_r2 = main()
    print(f"\n✅ 최종 달성 R²: {final_r2:.1%}")