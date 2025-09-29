import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import json
import joblib
from datetime import datetime

# 고급 모델들
import xgboost as xgb
import lightgbm as lgb

# 최적화 및 전처리
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 데이터 증강
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import PolynomialFeatures

# 베이지안 최적화 (optuna 대신 사용 가능)
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available. Using GridSearch for hyperparameter tuning.")

class AdvancedModelDeveloper:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def load_data(self):
        """데이터 로딩"""
        print("데이터 로딩...")
        self.df = pd.read_csv(self.data_path)
        
        # 기본 전처리
        self.df = self.df.fillna(self.df.median(numeric_only=True))
        
        print(f"데이터 크기: {self.df.shape}")
        return self.df
    
    def create_advanced_features(self, df):
        """고급 피처 엔지니어링"""
        print("고급 피처 엔지니어링 시작...")
        
        enhanced_df = df.copy()
        
        # 1. 물리 기반 복합 지수들
        print("  - 물리 기반 복합 지수 생성")
        
        # 화재 위험 지수 (Fire Weather Index 개선)
        if all(col in df.columns for col in ['t2m_0h', 'rh2m_0h', 'ws10m_0h']):
            # Haines Index (대기 불안정성)
            enhanced_df['haines_index'] = (df['t2m_0h'] - 10) + (100 - df['rh2m_0h']) / 10
            
            # Enhanced Fire Danger Index
            enhanced_df['enhanced_fdi'] = (df['t2m_0h'] / 10) * (100 - df['rh2m_0h']) * df['ws10m_0h'] / 10
        
        # 지형 복합 지수
        if all(col in df.columns for col in ['slope_mean', 'aspect_south_ratio', 'elevation_std']):
            # 지형 화재 확산 지수
            enhanced_df['terrain_fire_spread'] = (
                df['slope_mean'] * df['aspect_south_ratio'] * 
                (1 + df['elevation_std'] / 100)
            )
            
            # 지형 복잡도 지수
            enhanced_df['terrain_complexity'] = (
                df['slope_std'] * df['aspect_std'] * df['elevation_std']
            )
        
        # 연료 위험 지수
        if all(col in df.columns for col in ['ndvi_before', 'treecover_pre_fire_5x5', 'dry_days_30d_start']):
            # 연료 부하량 지수
            enhanced_df['fuel_load_index'] = (
                df['treecover_pre_fire_5x5'] * df['ndvi_before'] * 100
            )
            
            # 연료 건조 지수
            enhanced_df['fuel_moisture_deficit'] = (
                df['dry_days_30d_start'] * (1 - df['ndvi_before']) * 100
            )
        
        # 2. 시간적 특성 피처
        print("  - 시간적 특성 피처 생성")
        
        if 'fire_month' in df.columns:
            # 계절별 순환 특성
            enhanced_df['month_sin'] = np.sin(2 * np.pi * df['fire_month'] / 12)
            enhanced_df['month_cos'] = np.cos(2 * np.pi * df['fire_month'] / 12)
            
            # 화재 위험 계절 가중치
            fire_risk_months = {3: 1.2, 4: 1.5, 5: 1.3, 9: 1.3, 10: 1.5, 11: 1.2}
            enhanced_df['seasonal_fire_risk'] = df['fire_month'].map(fire_risk_months).fillna(1.0)
        
        # 3. 상호작용 피처들
        print("  - 상호작용 피처 생성")
        
        # 기상 상호작용
        weather_interactions = [
            ('t2m_0h', 'rh2m_0h', 'heat_humidity_interaction'),
            ('ws10m_0h', 'rh2m_0h', 'wind_dryness_interaction'),
            ('t2m_0h', 'ws10m_0h', 'heat_wind_interaction')
        ]
        
        for col1, col2, new_name in weather_interactions:
            if col1 in df.columns and col2 in df.columns:
                enhanced_df[new_name] = df[col1] * df[col2]
        
        # 지형-기상 상호작용
        if all(col in df.columns for col in ['slope_mean', 'ws10m_0h']):
            enhanced_df['slope_wind_effect'] = df['slope_mean'] * df['ws10m_0h']
        
        # 4. 통계적 파생 피처
        print("  - 통계적 파생 피처 생성")
        
        # 과거 데이터 통계 (과거 피처들이 있다면)
        past_features = [col for col in df.columns if '_past' in col]
        if past_features:
            # 온도 관련 과거 피처들
            temp_past = [col for col in past_features if 't2m' in col]
            if len(temp_past) > 3:
                temp_past_df = df[temp_past]
                enhanced_df['temp_past_mean'] = temp_past_df.mean(axis=1)
                enhanced_df['temp_past_std'] = temp_past_df.std(axis=1)
                enhanced_df['temp_past_trend'] = temp_past_df.iloc[:, -1] - temp_past_df.iloc[:, 0]
            
            # 습도 관련 과거 피처들
            rh_past = [col for col in past_features if 'rh2m' in col]
            if len(rh_past) > 3:
                rh_past_df = df[rh_past]
                enhanced_df['rh_past_mean'] = rh_past_df.mean(axis=1)
                enhanced_df['rh_past_std'] = rh_past_df.std(axis=1)
                enhanced_df['rh_past_min'] = rh_past_df.min(axis=1)
        
        # 5. 도메인 지식 기반 피처
        print("  - 도메인 지식 기반 피처 생성")
        
        # McArthur Forest Fire Danger Index 근사
        if all(col in df.columns for col in ['t2m_0h', 'rh2m_0h', 'ws10m_0h']):
            enhanced_df['mcarthur_ffdi'] = (
                2 * np.exp(-0.45 + 0.987 * np.log(np.maximum(df['dry_days_30d_start'], 1)) - 
                           0.0345 * df['rh2m_0h'] + 0.0338 * df['t2m_0h'] + 
                           0.0234 * df['ws10m_0h'])
            )
        
        # Keetch-Byram Drought Index 근사
        if 'dry_days_30d_start' in df.columns and 't2m_0h' in df.columns:
            enhanced_df['kbdi_approx'] = (
                df['dry_days_30d_start'] * (df['t2m_0h'] / 30) * 100
            )
        
        print(f"  피처 생성 완료: {len(enhanced_df.columns) - len(df.columns)}개 피처 추가")
        return enhanced_df
    
    def optimize_hyperparameters_bayesian(self, X, y, model_type='xgboost'):
        """베이지안 최적화를 통한 하이퍼파라미터 튜닝"""
        if not OPTUNA_AVAILABLE:
            return self.optimize_hyperparameters_grid(X, y, model_type)
        
        print(f"베이지안 최적화 시작 - {model_type}")
        
        def objective(trial):
            if model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'random_state': 42
                }
                model = xgb.XGBRegressor(**params)
                
            elif model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'random_state': 42,
                    'verbose': -1
                }
                model = lgb.LGBMRegressor(**params)
            
            # 교차 검증
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
            return cv_scores.mean()
        
        study = optuna.create_study(direction='maximize', study_name=f'{model_type}_optimization')
        study.optimize(objective, n_trials=50)
        
        print(f"최적 점수: {study.best_value:.4f}")
        print(f"최적 파라미터: {study.best_params}")
        
        return study.best_params
    
    def optimize_hyperparameters_grid(self, X, y, model_type='xgboost'):
        """Grid Search 기반 하이퍼파라미터 튜닝"""
        print(f"Grid Search 최적화 시작 - {model_type}")
        
        from sklearn.model_selection import GridSearchCV
        
        if model_type == 'xgboost':
            model = xgb.XGBRegressor(random_state=42)
            param_grid = {
                'n_estimators': [100, 200, 500],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        elif model_type == 'lightgbm':
            model = lgb.LGBMRegressor(random_state=42, verbose=-1)
            param_grid = {
                'n_estimators': [100, 200, 500],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1
        )
        grid_search.fit(X, y)
        
        print(f"최적 점수: {grid_search.best_score_:.4f}")
        print(f"최적 파라미터: {grid_search.best_params_}")
        
        return grid_search.best_params_
    
    def build_ensemble_model(self, X, y):
        """앙상블 모델 구축"""
        print("앙상블 모델 구축 중...")
        
        # 기본 모델들
        models = [
            ('rf', RandomForestRegressor(n_estimators=200, random_state=42)),
            ('xgb', xgb.XGBRegressor(n_estimators=200, random_state=42)),
            ('lgb', lgb.LGBMRegressor(n_estimators=200, random_state=42, verbose=-1)),
            ('elastic', ElasticNet(alpha=0.1, random_state=42))
        ]
        
        # Voting Regressor
        voting_regressor = VotingRegressor(estimators=models)
        
        # Stacking Regressor
        stacking_regressor = StackingRegressor(
            estimators=models[:-1],  # ElasticNet 제외
            final_estimator=Ridge(alpha=1.0),
            cv=5
        )
        
        # 모델 평가
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        ensemble_results = {}
        
        for name, model in [('Voting', voting_regressor), ('Stacking', stacking_regressor)]:
            print(f"  {name} 앙상블 훈련 중...")
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            ensemble_results[name] = {
                'model': model,
                'r2': r2,
                'rmse': rmse
            }
            
            print(f"    {name} R²: {r2:.4f}, RMSE: {rmse:.4f}")
        
        return ensemble_results
    
    def augment_data_smote(self, X, y):
        """SMOTE를 이용한 데이터 증강 (회귀용 변형)"""
        print("데이터 증강 (SMOTE 변형) 수행 중...")
        
        # 회귀 문제를 위해 타겟을 범주화
        y_binned = pd.cut(y, bins=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        
        # SMOTE 적용
        smote = SMOTE(random_state=42)
        X_resampled, y_binned_resampled = smote.fit_resample(X, y_binned)
        
        # 원래 타겟 값 복원 (각 빈의 평균값 사용)
        bin_means = y.groupby(y_binned).mean()
        y_resampled = y_binned_resampled.map(bin_means)
        
        print(f"데이터 증강 완료: {X.shape[0]} → {X_resampled.shape[0]} 샘플")
        
        return X_resampled, y_resampled
    
    def comprehensive_model_development(self):
        """종합적인 모델 개발 파이프라인"""
        print("="*60)
        print("🚀 고도화된 화재 예측 모델 개발 시작")
        print("="*60)
        
        # 1. 데이터 로딩
        df = self.load_data()
        
        # 2. 고급 피처 엔지니어링
        enhanced_df = self.create_advanced_features(df)
        
        # 3. 타겟과 피처 분리
        target_col = 'fire_area'
        X = enhanced_df.drop(columns=[target_col])
        y = enhanced_df[target_col]
        
        # 로그 변환
        y_log = np.log1p(y)
        
        # 4. 수치형 피처만 선택 및 전처리
        X_numeric = X.select_dtypes(include=[np.number])
        X_numeric = X_numeric.fillna(X_numeric.median())
        
        # 5. 피처 선택 (상위 100개)
        selector = SelectKBest(score_func=f_regression, k=min(100, X_numeric.shape[1]))
        X_selected = selector.fit_transform(X_numeric, y_log)
        selected_features = X_numeric.columns[selector.get_support()]
        
        print(f"선택된 피처 수: {len(selected_features)}")
        
        # 6. 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_log, test_size=0.2, random_state=42
        )
        
        # 7. 스케일링
        scaler = QuantileTransformer(n_quantiles=min(1000, X_train.shape[0]))
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 8. 고급 모델들 훈련
        results = {}
        
        # XGBoost
        print("\n🌟 XGBoost 모델 개발")
        try:
            if OPTUNA_AVAILABLE:
                xgb_params = self.optimize_hyperparameters_bayesian(X_train_scaled, y_train, 'xgboost')
            else:
                xgb_params = self.optimize_hyperparameters_grid(X_train_scaled, y_train, 'xgboost')
            
            xgb_model = xgb.XGBRegressor(**xgb_params)
            xgb_model.fit(X_train_scaled, y_train)
            
            y_pred_xgb = xgb_model.predict(X_test_scaled)
            r2_xgb = r2_score(y_test, y_pred_xgb)
            rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
            
            results['XGBoost'] = {
                'model': xgb_model,
                'r2': r2_xgb,
                'rmse': rmse_xgb,
                'params': xgb_params
            }
            
            print(f"XGBoost R²: {r2_xgb:.4f}, RMSE: {rmse_xgb:.4f}")
            
        except Exception as e:
            print(f"XGBoost 훈련 실패: {e}")
        
        # LightGBM
        print("\n🌟 LightGBM 모델 개발")
        try:
            if OPTUNA_AVAILABLE:
                lgb_params = self.optimize_hyperparameters_bayesian(X_train_scaled, y_train, 'lightgbm')
            else:
                lgb_params = self.optimize_hyperparameters_grid(X_train_scaled, y_train, 'lightgbm')
            
            lgb_model = lgb.LGBMRegressor(**lgb_params)
            lgb_model.fit(X_train_scaled, y_train)
            
            y_pred_lgb = lgb_model.predict(X_test_scaled)
            r2_lgb = r2_score(y_test, y_pred_lgb)
            rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
            
            results['LightGBM'] = {
                'model': lgb_model,
                'r2': r2_lgb,
                'rmse': rmse_lgb,
                'params': lgb_params
            }
            
            print(f"LightGBM R²: {r2_lgb:.4f}, RMSE: {rmse_lgb:.4f}")
            
        except Exception as e:
            print(f"LightGBM 훈련 실패: {e}")
        
        # 9. 앙상블 모델
        print("\n🌟 앙상블 모델 개발")
        try:
            ensemble_results = self.build_ensemble_model(X_train_scaled, y_train)
            results.update(ensemble_results)
        except Exception as e:
            print(f"앙상블 모델 구축 실패: {e}")
        
        # 10. 데이터 증강 후 재훈련
        print("\n🌟 데이터 증강 후 모델 재훈련")
        try:
            X_aug, y_aug = self.augment_data_smote(X_train_scaled, y_train)
            
            # 증강된 데이터로 최고 성능 모델 재훈련
            best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
            best_model_type = results[best_model_name]['model']
            
            if 'XGB' in best_model_name:
                aug_model = xgb.XGBRegressor(**results[best_model_name]['params'])
            elif 'LightGBM' in best_model_name:
                aug_model = lgb.LGBMRegressor(**results[best_model_name]['params'])
            else:
                aug_model = RandomForestRegressor(n_estimators=200, random_state=42)
            
            aug_model.fit(X_aug, y_aug)
            
            y_pred_aug = aug_model.predict(X_test_scaled)
            r2_aug = r2_score(y_test, y_pred_aug)
            rmse_aug = np.sqrt(mean_squared_error(y_test, y_pred_aug))
            
            results['Augmented_' + best_model_name] = {
                'model': aug_model,
                'r2': r2_aug,
                'rmse': rmse_aug
            }
            
            print(f"증강 후 {best_model_name} R²: {r2_aug:.4f}, RMSE: {rmse_aug:.4f}")
            
        except Exception as e:
            print(f"데이터 증강 실패: {e}")
        
        # 11. 결과 저장
        self.save_advanced_models(results, scaler, selected_features)
        
        return results
    
    def save_advanced_models(self, results, scaler, selected_features):
        """고도화된 모델들 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = '/Users/mmymacymac/Developer/Projects/WildFire_projects/wildfire/wildfire/ML'
        
        # 최고 성능 모델 찾기
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        best_model = results[best_model_name]
        
        # 최고 모델 저장
        model_path = f'{base_path}/best_advanced_model_{timestamp}.joblib'
        joblib.dump(best_model['model'], model_path)
        
        scaler_path = f'{base_path}/advanced_scaler_{timestamp}.joblib'
        joblib.dump(scaler, scaler_path)
        
        features_path = f'{base_path}/advanced_features_{timestamp}.json'
        with open(features_path, 'w') as f:
            json.dump(selected_features.tolist(), f, indent=2)
        
        # 결과 요약 저장
        summary = {
            'best_model': best_model_name,
            'best_r2': best_model['r2'],
            'best_rmse': best_model['rmse'],
            'all_results': {k: {'r2': v['r2'], 'rmse': v['rmse']} for k, v in results.items()},
            'feature_count': len(selected_features),
            'timestamp': timestamp
        }
        
        summary_path = f'{base_path}/advanced_model_summary_{timestamp}.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n🏆 최고 성능 모델: {best_model_name}")
        print(f"📊 성능: R² = {best_model['r2']:.4f}, RMSE = {best_model['rmse']:.4f}")
        print(f"💾 모델 저장 완료:")
        print(f"  - 모델: {model_path}")
        print(f"  - 스케일러: {scaler_path}")
        print(f"  - 피처: {features_path}")
        print(f"  - 요약: {summary_path}")

def main():
    print("🚀 화재 예측 모델 고도화 개발")
    
    data_path = '/Users/mmymacymac/Developer/Projects/WildFire_projects/wildfire/wildfire/ML/clean_training_dataset.csv'
    
    developer = AdvancedModelDeveloper(data_path)
    results = developer.comprehensive_model_development()
    
    print("\n" + "="*60)
    print("📊 최종 모델 성능 비교")
    print("="*60)
    
    for model_name, metrics in results.items():
        print(f"{model_name:20s}: R² = {metrics['r2']:6.4f}, RMSE = {metrics['rmse']:6.2f}")

if __name__ == "__main__":
    main()