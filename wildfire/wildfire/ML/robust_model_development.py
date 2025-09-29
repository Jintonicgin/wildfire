import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import json
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
import xgboost as xgb

class RobustModelDeveloper:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.results = {}
        
    def load_and_clean_data(self):
        """데이터 로딩 및 강력한 전처리"""
        print("📊 데이터 로딩 및 강력한 전처리...")
        self.df = pd.read_csv(self.data_path)
        
        print(f"원본 데이터 크기: {self.df.shape}")
        
        # 타겟 변수 확인
        target_col = 'fire_area'
        if target_col not in self.df.columns:
            raise ValueError(f"타겟 변수 '{target_col}'를 찾을 수 없습니다.")
        
        # 타겟 변수의 분포 확인
        print(f"타겟 변수 분포:")
        print(f"  - Min: {self.df[target_col].min():.4f}")
        print(f"  - Max: {self.df[target_col].max():.4f}")
        print(f"  - Mean: {self.df[target_col].mean():.4f}")
        print(f"  - Median: {self.df[target_col].median():.4f}")
        print(f"  - 0값 비율: {(self.df[target_col] == 0).sum() / len(self.df):.3f}")
        
        # 1. 수치형 컬럼만 선택
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"수치형 컬럼 수: {len(numeric_cols)}")
        
        # 2. 타겟 변수와 피처 분리
        X_cols = [col for col in numeric_cols if col != target_col]
        X = self.df[X_cols].copy()
        y = self.df[target_col].copy()
        
        # 3. 결측치 처리 전략
        print("  결측치 처리...")
        
        # 결측치 비율 계산
        missing_ratios = X.isnull().sum() / len(X)
        
        # 80% 이상 결측치가 있는 컬럼 제거
        high_missing_cols = missing_ratios[missing_ratios > 0.8].index.tolist()
        if high_missing_cols:
            print(f"    80% 이상 결측치 컬럼 제거: {len(high_missing_cols)}개")
            X = X.drop(columns=high_missing_cols)
        
        # 나머지 결측치를 중앙값으로 대체
        X = X.fillna(X.median())
        
        # 4. 무한값 및 이상치 처리
        print("  무한값 및 이상치 처리...")
        
        # 무한값을 NaN으로 변환 후 중앙값으로 대체
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # 극단적 이상치 처리 (IQR 방법)
        for col in X.columns:
            Q1 = X[col].quantile(0.01)
            Q3 = X[col].quantile(0.99)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            X[col] = X[col].clip(lower_bound, upper_bound)
        
        # 5. 분산이 너무 낮은 컬럼 제거
        print("  저분산 컬럼 제거...")
        variance_threshold = 1e-6
        low_var_cols = X.columns[X.var() < variance_threshold].tolist()
        if low_var_cols:
            print(f"    저분산 컬럼 제거: {len(low_var_cols)}개")
            X = X.drop(columns=low_var_cols)
        
        # 6. 최종 데이터 검증
        print("  최종 데이터 검증...")
        assert not X.isnull().any().any(), "여전히 NaN 값이 존재합니다!"
        assert not np.isinf(X.values).any(), "여전히 무한값이 존재합니다!"
        
        print(f"  최종 피처 수: {X.shape[1]}")
        print(f"  최종 샘플 수: {X.shape[0]}")
        
        return X, y
    
    def smart_feature_engineering(self, X, y):
        """간단하지만 효과적인 피처 엔지니어링"""
        print("🔧 스마트 피처 엔지니어링...")
        
        enhanced_X = X.copy()
        initial_feature_count = enhanced_X.shape[1]
        
        # 1. 핵심 기상 지수 생성
        if all(col in X.columns for col in ['t2m_0h', 'rh2m_0h', 'ws10m_0h']):
            print("  - 기상 복합 지수 생성")
            # 화재 위험도 지수 (온도 높고, 습도 낮고, 바람 강할 때 위험)
            enhanced_X['fire_weather_index'] = (
                X['t2m_0h'] * (100 - X['rh2m_0h']) * X['ws10m_0h'] / 10000
            )
            # 건조도 지수
            enhanced_X['dryness_index'] = X['t2m_0h'] / (X['rh2m_0h'] + 1)
            # 바람 효과
            enhanced_X['wind_drying_effect'] = X['ws10m_0h'] * (100 - X['rh2m_0h']) / 100
        
        # 2. FWI 시스템 기반 복합 지수
        fwi_cols = ['ffmc_0h', 'dmc_0h', 'dc_0h', 'isi_0h', 'bui_0h', 'fwi_0h']
        available_fwi = [col for col in fwi_cols if col in X.columns]
        if len(available_fwi) >= 3:
            print("  - FWI 복합 지수 생성")
            if 'fwi_0h' in available_fwi and 'bui_0h' in available_fwi:
                enhanced_X['fwi_bui_product'] = X['fwi_0h'] * X['bui_0h']
            if 'ffmc_0h' in available_fwi and 'isi_0h' in available_fwi:
                enhanced_X['fire_spread_potential'] = X['ffmc_0h'] * X['isi_0h'] / 100
        
        # 3. 지형 효과
        terrain_cols = ['slope_mean', 'elevation_mean', 'aspect_south_ratio']
        if all(col in X.columns for col in terrain_cols):
            print("  - 지형 복합 지수 생성")
            enhanced_X['terrain_fire_risk'] = (
                X['slope_mean'] * X['aspect_south_ratio'] * 
                np.log1p(X['elevation_mean'] / 1000)
            )
        
        # 4. 식생 지수
        if all(col in X.columns for col in ['ndvi_before', 'treecover_pre_fire_5x5']):
            print("  - 식생 복합 지수 생성")
            enhanced_X['vegetation_fuel_load'] = X['ndvi_before'] * X['treecover_pre_fire_5x5']
            
            if 'ndvi_stress' in X.columns:
                enhanced_X['vegetation_stress_load'] = X['ndvi_stress'] * X['treecover_pre_fire_5x5']
        
        # 5. 시간적 특성
        if 'fire_month' in X.columns:
            print("  - 시간적 특성 생성")
            # 고위험 계절 (봄, 가을)
            high_risk_months = [3, 4, 5, 9, 10, 11]
            enhanced_X['high_risk_season'] = X['fire_month'].isin(high_risk_months).astype(int)
            
            # 순환 인코딩
            enhanced_X['month_sin'] = np.sin(2 * np.pi * X['fire_month'] / 12)
            enhanced_X['month_cos'] = np.cos(2 * np.pi * X['fire_month'] / 12)
        
        # 6. 과거 기상 요약 (최근 데이터만)
        temp_past_cols = [col for col in X.columns if 't2m_' in col and '_past' in col][:5]
        if len(temp_past_cols) >= 3:
            print("  - 과거 온도 요약")
            temp_df = X[temp_past_cols]
            enhanced_X['temp_past_mean'] = temp_df.mean(axis=1)
            enhanced_X['temp_past_std'] = temp_df.std(axis=1)
            enhanced_X['temp_trend'] = temp_df.iloc[:, -1] - temp_df.iloc[:, 0]
        
        humidity_past_cols = [col for col in X.columns if 'rh2m_' in col and '_past' in col][:5]
        if len(humidity_past_cols) >= 3:
            print("  - 과거 습도 요약")
            humidity_df = X[humidity_past_cols]
            enhanced_X['humidity_past_min'] = humidity_df.min(axis=1)
            enhanced_X['humidity_past_mean'] = humidity_df.mean(axis=1)
        
        # 7. 최종 데이터 검증
        enhanced_X = enhanced_X.replace([np.inf, -np.inf], np.nan)
        enhanced_X = enhanced_X.fillna(enhanced_X.median())
        
        print(f"  피처 엔지니어링 완료: {enhanced_X.shape[1] - initial_feature_count}개 피처 추가")
        return enhanced_X
    
    def intelligent_feature_selection(self, X, y, max_features=50):
        """신뢰할 수 있는 피처 선택"""
        print(f"🎯 지능적 피처 선택 (최대 {max_features}개)")
        
        # 1. 타겟과의 상관관계 기반 초기 필터링
        print("  1단계: 상관관계 기반 필터링")
        correlations = []
        
        for col in X.columns:
            try:
                corr = np.abs(np.corrcoef(X[col], y)[0, 1])
                if not np.isnan(corr):
                    correlations.append((col, corr))
            except:
                continue
        
        # 상관관계 순으로 정렬
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 피처 선택 (최대 피처 수의 2배)
        top_features = [feat for feat, corr in correlations[:max_features * 2]]
        X_filtered = X[top_features]
        
        print(f"    상관관계 기반 선택: {len(top_features)}개")
        
        # 2. 통계적 피처 선택
        print("  2단계: 통계적 피처 선택")
        try:
            # f_regression 사용 (mutual_info_regression보다 안정적)
            selector = SelectKBest(score_func=f_regression, k=min(max_features, len(top_features)))
            X_selected = selector.fit_transform(X_filtered, y)
            selected_features = X_filtered.columns[selector.get_support()]
            
            print(f"    최종 선택된 피처: {len(selected_features)}개")
            return X_selected, selected_features
            
        except Exception as e:
            print(f"    통계적 선택 실패, 상관관계 기반 결과 사용: {e}")
            return X_filtered.values, X_filtered.columns
    
    def robust_preprocessing(self, X_train, X_test, y_train):
        """강력한 전처리 파이프라인"""
        print("⚙️ 강력한 전처리...")
        
        # 0. 추가 데이터 정리
        print("  추가 데이터 정리...")
        # 무한값 처리
        X_train = np.where(np.isinf(X_train), np.nan, X_train)
        X_test = np.where(np.isinf(X_test), np.nan, X_test)
        
        # NaN을 중앙값으로 처리
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        
        # 극값 클리핑 (99.9% 범위로)
        for i in range(X_train.shape[1]):
            percentile_low = np.percentile(X_train[:, i], 0.1)
            percentile_high = np.percentile(X_train[:, i], 99.9)
            X_train[:, i] = np.clip(X_train[:, i], percentile_low, percentile_high)
            X_test[:, i] = np.clip(X_test[:, i], percentile_low, percentile_high)
        
        # 1. 스케일러 선택 (RobustScaler가 이상치에 강함)
        print("  스케일러: RobustScaler 적용")
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 스케일링 후 추가 검증
        X_train_scaled = np.where(np.isinf(X_train_scaled), 0, X_train_scaled)
        X_test_scaled = np.where(np.isinf(X_test_scaled), 0, X_test_scaled)
        X_train_scaled = np.where(np.isnan(X_train_scaled), 0, X_train_scaled)
        X_test_scaled = np.where(np.isnan(X_test_scaled), 0, X_test_scaled)
        
        # 2. 타겟 변환 (log1p - 간단하고 효과적)
        print("  타겟 변환: log1p 적용")
        y_train_transformed = np.log1p(y_train)
        
        # 타겟 변환 후 검증
        y_train_transformed = np.where(np.isinf(y_train_transformed), 0, y_train_transformed)
        y_train_transformed = np.where(np.isnan(y_train_transformed), 0, y_train_transformed)
        
        print(f"  전처리 완료 - 훈련 데이터: {X_train_scaled.shape}, 테스트 데이터: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train_transformed, scaler
    
    def build_robust_models(self, X_train, X_test, y_train, y_test):
        """강력하고 신뢰할 수 있는 모델 구축"""
        print("🏗️ 강력한 모델 구축...")
        
        models = {}
        
        # 타겟 역변환 함수
        def inverse_transform(y_pred):
            return np.expm1(y_pred)
        
        # 성능 평가 함수
        def evaluate_model(y_true, y_pred, model_name):
            y_pred_orig = inverse_transform(y_pred)
            y_true_orig = inverse_transform(y_true)
            
            r2 = r2_score(y_true_orig, y_pred_orig)
            rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
            mae = mean_absolute_error(y_true_orig, y_pred_orig)
            
            return r2, rmse, mae, y_pred_orig
        
        # 1. Random Forest (강력하고 안정적)
        print("  1. Random Forest")
        try:
            rf_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            y_pred_rf = rf_model.predict(X_test)
            
            r2_rf, rmse_rf, mae_rf, y_pred_rf_orig = evaluate_model(y_test, y_pred_rf, 'RandomForest')
            models['RandomForest'] = {
                'model': rf_model, 'r2': r2_rf, 'rmse': rmse_rf, 'mae': mae_rf, 
                'predictions': y_pred_rf_orig
            }
            print(f"    RandomForest - R²: {r2_rf:.4f}, RMSE: {rmse_rf:.4f}")
            
        except Exception as e:
            print(f"    RandomForest 실패: {e}")
        
        # 2. Gradient Boosting
        print("  2. Gradient Boosting")
        try:
            gb_model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
            gb_model.fit(X_train, y_train)
            y_pred_gb = gb_model.predict(X_test)
            
            r2_gb, rmse_gb, mae_gb, y_pred_gb_orig = evaluate_model(y_test, y_pred_gb, 'GradientBoosting')
            models['GradientBoosting'] = {
                'model': gb_model, 'r2': r2_gb, 'rmse': rmse_gb, 'mae': mae_gb,
                'predictions': y_pred_gb_orig
            }
            print(f"    GradientBoosting - R²: {r2_gb:.4f}, RMSE: {rmse_gb:.4f}")
            
        except Exception as e:
            print(f"    GradientBoosting 실패: {e}")
        
        # 3. XGBoost
        print("  3. XGBoost")
        try:
            xgb_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='rmse'
            )
            xgb_model.fit(X_train, y_train)
            y_pred_xgb = xgb_model.predict(X_test)
            
            r2_xgb, rmse_xgb, mae_xgb, y_pred_xgb_orig = evaluate_model(y_test, y_pred_xgb, 'XGBoost')
            models['XGBoost'] = {
                'model': xgb_model, 'r2': r2_xgb, 'rmse': rmse_xgb, 'mae': mae_xgb,
                'predictions': y_pred_xgb_orig
            }
            print(f"    XGBoost - R²: {r2_xgb:.4f}, RMSE: {rmse_xgb:.4f}")
            
        except Exception as e:
            print(f"    XGBoost 실패: {e}")
        
        # 4. Ridge Regression (안정적인 기본 모델)
        print("  4. Ridge Regression")
        try:
            ridge_model = Ridge(alpha=1.0, random_state=42)
            ridge_model.fit(X_train, y_train)
            y_pred_ridge = ridge_model.predict(X_test)
            
            r2_ridge, rmse_ridge, mae_ridge, y_pred_ridge_orig = evaluate_model(y_test, y_pred_ridge, 'Ridge')
            models['Ridge'] = {
                'model': ridge_model, 'r2': r2_ridge, 'rmse': rmse_ridge, 'mae': mae_ridge,
                'predictions': y_pred_ridge_orig
            }
            print(f"    Ridge - R²: {r2_ridge:.4f}, RMSE: {rmse_ridge:.4f}")
            
        except Exception as e:
            print(f"    Ridge 실패: {e}")
        
        # 5. ElasticNet
        print("  5. ElasticNet")
        try:
            elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
            elastic_model.fit(X_train, y_train)
            y_pred_elastic = elastic_model.predict(X_test)
            
            r2_elastic, rmse_elastic, mae_elastic, y_pred_elastic_orig = evaluate_model(y_test, y_pred_elastic, 'ElasticNet')
            models['ElasticNet'] = {
                'model': elastic_model, 'r2': r2_elastic, 'rmse': rmse_elastic, 'mae': mae_elastic,
                'predictions': y_pred_elastic_orig
            }
            print(f"    ElasticNet - R²: {r2_elastic:.4f}, RMSE: {rmse_elastic:.4f}")
            
        except Exception as e:
            print(f"    ElasticNet 실패: {e}")
        
        # 6. 앙상블 (평균)
        if len(models) >= 2:
            print("  6. 앙상블 (평균)")
            try:
                predictions = [model_info['predictions'] for model_info in models.values()]
                ensemble_pred = np.mean(predictions, axis=0)
                
                y_test_orig = inverse_transform(y_test)
                r2_ensemble = r2_score(y_test_orig, ensemble_pred)
                rmse_ensemble = np.sqrt(mean_squared_error(y_test_orig, ensemble_pred))
                mae_ensemble = mean_absolute_error(y_test_orig, ensemble_pred)
                
                models['Ensemble'] = {
                    'model': 'ensemble', 'r2': r2_ensemble, 'rmse': rmse_ensemble, 'mae': mae_ensemble,
                    'predictions': ensemble_pred
                }
                print(f"    Ensemble - R²: {r2_ensemble:.4f}, RMSE: {rmse_ensemble:.4f}")
                
            except Exception as e:
                print(f"    Ensemble 실패: {e}")
        
        return models
    
    def comprehensive_model_development(self):
        """종합적인 강력한 모델 개발"""
        print("=" * 60)
        print("🚀 강력한 화재 예측 모델 개발")
        print("=" * 60)
        
        # 1. 데이터 로딩 및 강력한 전처리
        X, y = self.load_and_clean_data()
        
        # 2. 스마트 피처 엔지니어링
        X_enhanced = self.smart_feature_engineering(X, y)
        
        # 3. 지능적 피처 선택
        X_selected, selected_features = self.intelligent_feature_selection(X_enhanced, y, max_features=50)
        
        # 4. 시계열 고려 데이터 분할
        print("📊 시계열 고려 데이터 분할...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.25, shuffle=False, random_state=42
        )
        
        # 5. 강력한 전처리
        X_train_proc, X_test_proc, y_train_proc, scaler = self.robust_preprocessing(
            X_train, X_test, y_train
        )
        
        # 6. 강력한 모델 구축
        models = self.build_robust_models(X_train_proc, X_test_proc, y_train_proc, y_test)
        
        # 7. 결과 저장
        self.save_robust_models(models, scaler, selected_features)
        
        return models
    
    def save_robust_models(self, models, scaler, selected_features):
        """강력한 모델 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = '/Users/mmymacymac/Developer/Projects/WildFire_projects/wildfire/wildfire/ML'
        
        if not models:
            print("❌ 성공한 모델이 없습니다.")
            return
        
        # 최고 성능 모델 찾기
        best_model_name = max(models.keys(), key=lambda x: models[x]['r2'])
        best_model_info = models[best_model_name]
        
        # 모델 저장
        if best_model_info['model'] != 'ensemble':
            model_path = f'{base_path}/robust_best_model_{timestamp}.joblib'
            joblib.dump(best_model_info['model'], model_path)
        else:
            model_path = "ensemble_model"
        
        # 스케일러 저장
        scaler_path = f'{base_path}/robust_scaler_{timestamp}.joblib'
        joblib.dump(scaler, scaler_path)
        
        # 피처 목록 저장
        features_path = f'{base_path}/robust_features_{timestamp}.json'
        with open(features_path, 'w') as f:
            json.dump(selected_features.tolist(), f, indent=2)
        
        # 결과 요약 저장
        summary = {
            'best_model': best_model_name,
            'best_r2': float(best_model_info['r2']),
            'best_rmse': float(best_model_info['rmse']),
            'best_mae': float(best_model_info['mae']),
            'all_results': {
                k: {
                    'r2': float(v['r2']), 
                    'rmse': float(v['rmse']), 
                    'mae': float(v['mae'])
                } for k, v in models.items()
            },
            'feature_count': len(selected_features),
            'improvements_applied': [
                'Robust data cleaning and outlier handling',
                'Smart feature engineering',
                'Correlation-based feature selection',
                'RobustScaler for preprocessing',
                'Log1p target transformation',
                'Time series aware data split',
                'Multiple model ensemble'
            ],
            'timestamp': timestamp
        }
        
        summary_path = f'{base_path}/robust_model_summary_{timestamp}.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "=" * 60)
        print("📊 최종 강력한 모델 성능 비교")
        print("=" * 60)
        
        for model_name, metrics in models.items():
            print(f"{model_name:20s}: R² = {metrics['r2']:6.4f}, RMSE = {metrics['rmse']:8.2f}, MAE = {metrics['mae']:8.2f}")
        
        print(f"\n🏆 최고 성능 모델: {best_model_name}")
        print(f"📈 최종 성능:")
        print(f"  - R² = {best_model_info['r2']:.4f}")
        print(f"  - RMSE = {best_model_info['rmse']:.4f}")
        print(f"  - MAE = {best_model_info['mae']:.4f}")
        
        print(f"\n💾 저장된 파일:")
        print(f"  - 모델: {model_path}")
        print(f"  - 스케일러: {scaler_path}")
        print(f"  - 피처 목록: {features_path}")
        print(f"  - 결과 요약: {summary_path}")

def main():
    print("🔥 강력한 화재 예측 모델 개발")
    
    data_path = '/Users/mmymacymac/Developer/Projects/WildFire_projects/wildfire/wildfire/ML/clean_training_dataset.csv'
    
    developer = RobustModelDeveloper(data_path)
    results = developer.comprehensive_model_development()
    
    return results

if __name__ == "__main__":
    main()