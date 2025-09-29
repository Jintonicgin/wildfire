#!/usr/bin/env python3
"""
통합 고급 면적 모델 - 3가지 모델의 장점 결합
======================================================
1. advanced_area_boost.py: 스태킹, 베이지안 최적화, 고급 부스팅
2. sophisticated_area_model.py: 화재 물리학 기반 피처 엔지니어링, 다단계 예측
3. train_clean_models.py: 클린 데이터셋, 데이터 누수 제거

결합된 특징:
- 화재 물리학 기반 피처 엔지니어링
- 고급 스태킹 및 앙상블
- 베이지안 최적화
- 클린 데이터 전처리
- 다단계 예측 (발생 확률 → 면적 크기)
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.base import BaseEstimator, RegressorMixin, clone
import xgboost as xgb
import lightgbm as lgb
import optuna
from scipy import stats
from scipy.special import boxcox1p
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

warnings.filterwarnings('ignore')

class CombinedAdvancedAreaModel:
    """통합 고급 면적 모델"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = {}
        self.best_params = {}

    def load_and_clean_data(self):
        """클린 데이터 로드 및 고급 전처리"""
        print("🔥 클린 데이터 로딩 및 전처리...")

        # 클린 데이터 로드
        try:
            df = pd.read_csv('clean_training_dataset.csv')
            print("   ✅ 클린 데이터셋 사용")
        except:
            df = pd.read_csv('final_merged_feature_engineered.csv')
            print("   ⚠️ 원본 데이터셋 사용 - 클린 전처리 적용")
            df = self._apply_clean_preprocessing(df)

        # 화재 데이터 필터링
        fire_mask = (df['fire_area'] > 0) & df['fire_area'].notna()
        fire_df = df[fire_mask].copy()

        print(f"   📊 화재 데이터: {fire_df.shape}")
        print(f"   📊 면적 분포: 평균 {fire_df['fire_area'].mean():.2f} ha, 중앙값 {fire_df['fire_area'].median():.2f} ha")

        return fire_df

    def _apply_clean_preprocessing(self, df):
        """클린 데이터 전처리 (train_clean_models.py 기법)"""
        print("   🧹 클린 전처리 적용 중...")

        # 결측치 처리
        if 'prectotcorr_0h' in df.columns:
            df['prectotcorr_0h'].fillna(0, inplace=True)

        # 기상 데이터 결측치 처리
        weather_cols = [col for col in df.columns if any(weather in col for weather in
                       ['t2m_', 'rh2m_', 'ws2m_', 'wd2m_', 'ws10m_', 'wd10m_', 'ps_', 'allsky_sfc_sw_dwn_'])]

        for col in weather_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mean(), inplace=True)

        # 지형/식생 데이터 결측치 처리
        terrain_cols = ['elevation_mean', 'slope_mean', 'aspect_mode', 'ndvi_before']
        for col in terrain_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)

        return df

    def create_physics_based_features(self, df):
        """화재 물리학 기반 피처 엔지니어링 (sophisticated_area_model.py 기법)"""
        print("   🔬 화재 물리학 기반 피처 생성...")

        df_enhanced = df.copy()

        # 1. 화재 위험도 지수
        if all(col in df.columns for col in ['fwi_0h', 'ffmc_0h', 'dmc_0h', 'dc_0h']):
            df_enhanced['fire_danger_index'] = (
                df['fwi_0h'] * 0.4 +
                df['ffmc_0h'] * 0.3 +
                df['dmc_0h'] * 0.2 +
                df['dc_0h'] * 0.1
            )

        # 2. 확산 잠재력 지수 (바람 + 경사)
        if all(col in df.columns for col in ['ws10m_0h', 'slope_mean']):
            df_enhanced['spread_potential'] = df['ws10m_0h'] * np.sqrt(df['slope_mean'])

        # 3. 건조 지수 (습도 + 건조일수)
        if all(col in df.columns for col in ['rh2m_0h', 'dry_days_30d_start']):
            df_enhanced['dryness_index'] = (100 - df['rh2m_0h']) * np.log1p(df['dry_days_30d_start'])

        # 4. 지형적 화재 위험도 (고도 + 경사 + 향)
        if all(col in df.columns for col in ['elevation_mean', 'slope_mean', 'aspect_mode']):
            # 남향 경사면이 더 위험
            south_aspect = np.abs(df['aspect_mode'] - 180) / 180
            df_enhanced['terrain_fire_risk'] = (
                df['elevation_mean'] / 1000 *
                df['slope_mean'] / 45 *
                (1 - south_aspect)
            )

        # 5. 식생 화재 위험도
        if 'ndvi_before' in df.columns:
            # NDVI가 중간값일 때 가장 위험 (너무 낮으면 연료 부족, 너무 높으면 수분 많음)
            optimal_ndvi = 0.3
            df_enhanced['vegetation_fire_risk'] = 1 - np.abs(df['ndvi_before'] - optimal_ndvi) / optimal_ndvi

        # 6. 복합 위험도 지수
        risk_columns = ['fire_danger_index', 'spread_potential', 'dryness_index',
                       'terrain_fire_risk', 'vegetation_fire_risk']
        available_risk_cols = [col for col in risk_columns if col in df_enhanced.columns]

        if available_risk_cols:
            df_enhanced['composite_risk_index'] = df_enhanced[available_risk_cols].mean(axis=1)

        # 7. 화재 크기 카테고리 (다단계 예측용)
        if 'fire_area' in df.columns:
            q25, q75, q95 = df['fire_area'].quantile([0.25, 0.75, 0.95])
            df_enhanced['fire_size_category'] = pd.cut(
                df['fire_area'],
                bins=[-np.inf, q25, q75, q95, np.inf],
                labels=['small', 'medium', 'large', 'extreme'],
                include_lowest=True
            )

        print(f"   ✅ {len([col for col in df_enhanced.columns if col not in df.columns])}개 물리학 기반 피처 생성")
        return df_enhanced

    def select_best_features(self, X, y, max_features=35):
        """고급 피처 선택"""
        print(f"   🎯 최적 피처 선택 (목표: {max_features}개)...")

        from sklearn.feature_selection import SelectKBest, f_regression, RFE
        from sklearn.ensemble import RandomForestRegressor

        # 1. 통계적 피처 선택
        print(f"      📊 통계적 피처 선택 중...")
        selector_stats = SelectKBest(score_func=f_regression, k=min(max_features*2, X.shape[1]))
        X_stats = selector_stats.fit_transform(X, y)
        selected_features_stats = X.columns[selector_stats.get_support()].tolist()
        print(f"      ✅ 통계적 방법: {len(selected_features_stats)}개 선택")

        # 2. 모델 기반 피처 선택 (빠른 방법)
        print(f"      🤖 모델 기반 피처 선택 중...")
        rf = RandomForestRegressor(n_estimators=50, random_state=42)  # 더 빠르게
        rf.fit(X, y)

        # RFE 대신 중요도 기반 선택 (훨씬 빠름)
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        selected_features_rf = importance_df.head(max_features)['feature'].tolist()
        print(f"      ✅ 모델 기반: {len(selected_features_rf)}개 선택")

        # 3. 두 방법의 교집합 + 중요도 기반 추가
        print(f"      🔄 피처 결합 중...")
        common_features = list(set(selected_features_stats) & set(selected_features_rf))
        print(f"      📝 공통 피처: {len(common_features)}개")

        if len(common_features) < max_features:
            # 부족한 만큼 중요도가 높은 피처 추가
            print(f"      ➕ 추가 피처 선택 중...")
            rf.fit(X, y)
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)

            additional_features = []
            for feature in importance_df['feature']:
                if feature not in common_features and len(common_features + additional_features) < max_features:
                    additional_features.append(feature)

            final_features = common_features + additional_features
            print(f"      ✅ 추가 피처: {len(additional_features)}개")
        else:
            final_features = common_features[:max_features]

        print(f"   ✅ 최종 선택된 피처: {len(final_features)}개")
        return final_features

    def create_stacking_regressor(self, trial=None):
        """고급 스태킹 회귀기 생성 (advanced_area_boost.py 기법)"""

        # Base 모델들
        base_models = [
            RandomForestRegressor(
                n_estimators=trial.suggest_int('rf_n_estimators', 100, 500) if trial else 300,
                max_depth=trial.suggest_int('rf_max_depth', 10, 30) if trial else 20,
                random_state=42
            ),
            GradientBoostingRegressor(
                n_estimators=trial.suggest_int('gb_n_estimators', 100, 300) if trial else 200,
                learning_rate=trial.suggest_float('gb_learning_rate', 0.05, 0.2) if trial else 0.1,
                random_state=42
            ),
            xgb.XGBRegressor(
                n_estimators=trial.suggest_int('xgb_n_estimators', 100, 300) if trial else 200,
                learning_rate=trial.suggest_float('xgb_learning_rate', 0.05, 0.2) if trial else 0.1,
                random_state=42
            ),
            ExtraTreesRegressor(
                n_estimators=trial.suggest_int('et_n_estimators', 100, 300) if trial else 200,
                random_state=42
            ),
            Ridge(
                alpha=trial.suggest_float('ridge_alpha', 0.1, 10.0) if trial else 1.0
            )
        ]

        # Meta 모델
        meta_model = ElasticNet(
            alpha=trial.suggest_float('meta_alpha', 0.1, 2.0) if trial else 0.5,
            l1_ratio=trial.suggest_float('meta_l1_ratio', 0.1, 0.9) if trial else 0.5
        )

        return VotingRegressor(estimators=[
            ('rf', base_models[0]),
            ('gb', base_models[1]),
            ('xgb', base_models[2]),
            ('et', base_models[3]),
            ('ridge', base_models[4])
        ])

    def optimize_with_optuna(self, X_train, y_train, X_val, y_val, n_trials=50):
        """베이지안 최적화 (advanced_area_boost.py 기법)"""
        print("   🎯 베이지안 최적화 실행...")

        def objective(trial):
            print(f"      Trial {trial.number + 1}/{n_trials} 실행 중...", end=" ")

            model = self.create_stacking_regressor(trial)

            # 스케일러 최적화
            scaler_type = trial.suggest_categorical('scaler', ['standard', 'robust', 'quantile'])
            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'robust':
                scaler = RobustScaler()
            else:
                scaler = QuantileTransformer()

            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # 간단한 로그 변환만 사용 (안정적)
            y_train_transformed = np.log1p(y_train)
            y_val_transformed = np.log1p(y_val)

            # 모델 훈련
            print("모델 훈련 중...", end=" ")
            model.fit(X_train_scaled, y_train_transformed)
            y_pred_transformed = model.predict(X_val_scaled)

            # 역변환
            y_pred = np.expm1(y_pred_transformed)

            # 음수값 제거
            y_pred = np.maximum(y_pred, 0.001)

            # R² 스코어 계산
            r2 = r2_score(y_val, y_pred)
            print(f"R² = {r2:.4f}")
            return r2

        # Optuna 로그 레벨 조정
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())

        # 진행상황 콜백 추가
        def progress_callback(study, trial):
            print(f"   📊 Trial {trial.number + 1}/{n_trials} 완료: R² = {trial.value:.4f} (최고: {study.best_value:.4f})")

        study.optimize(objective, n_trials=n_trials, callbacks=[progress_callback])

        self.best_params = study.best_params
        print(f"   ✅ 최적화 완료! 최고 R²: {study.best_value:.4f}")
        print(f"   🏆 최적 파라미터: {study.best_params}")
        return study.best_params

    def train_combined_model(self, data_path=None):
        """통합 모델 훈련"""
        print("🚀 통합 고급 면적 모델 훈련 시작")
        print("="*60)

        # 1. 데이터 로드 및 전처리
        fire_df = self.load_and_clean_data()

        # 2. 화재 물리학 기반 피처 생성
        fire_df_enhanced = self.create_physics_based_features(fire_df)

        # 3. 피처 및 타겟 준비
        target_col = 'fire_area'
        exclude_cols = [target_col, 'fire_size_category'] + [col for col in fire_df_enhanced.columns
                       if col.startswith('Unnamed') or 'index' in col.lower()]

        feature_cols = [col for col in fire_df_enhanced.columns if col not in exclude_cols]
        X = fire_df_enhanced[feature_cols].select_dtypes(include=[np.number]).fillna(0)
        y = fire_df_enhanced[target_col]

        # 무한대 및 극값 처리
        print("   🧹 무한대 및 극값 처리...")
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())

        # 극값 클리핑 (99.9% 분위수 기준)
        for col in X.columns:
            q99 = X[col].quantile(0.999)
            q01 = X[col].quantile(0.001)
            X[col] = X[col].clip(lower=q01, upper=q99)

        print(f"   📊 초기 피처: {X.shape[1]}개")

        # 4. 피처 선택
        selected_features = self.select_best_features(X, y, max_features=35)
        X_selected = X[selected_features]

        # 5. 데이터 분할
        X_train, X_temp, y_train, y_temp = train_test_split(X_selected, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        print(f"   📊 훈련 데이터: {X_train.shape}")
        print(f"   📊 검증 데이터: {X_val.shape}")
        print(f"   📊 테스트 데이터: {X_test.shape}")

        # 6. 베이지안 최적화
        best_params = self.optimize_with_optuna(X_train, y_train, X_val, y_val, n_trials=30)

        # 7. 최적 모델 훈련
        print("   🎯 최적 모델 훈련...")
        print(f"      🔧 모델 생성 중...")
        best_model = self.create_stacking_regressor()

        # 최적 스케일러 적용
        scaler_type = best_params.get('scaler', 'robust')
        print(f"      ⚖️ 스케일러 적용: {scaler_type}")
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            scaler = QuantileTransformer()

        print(f"      📏 데이터 스케일링 중...")
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # 타겟 변환 (로그만 사용)
        print(f"      🔄 타겟 변환: 로그 변환")
        y_train_transformed = np.log1p(y_train)

        # 모델 훈련
        print(f"      🤖 앙상블 모델 훈련 중...")
        best_model.fit(X_train_scaled, y_train_transformed)
        print(f"      ✅ 모델 훈련 완료!")

        # 8. 성능 평가
        y_pred_train_transformed = best_model.predict(X_train_scaled)
        y_pred_val_transformed = best_model.predict(X_val_scaled)
        y_pred_test_transformed = best_model.predict(X_test_scaled)

        # 역변환
        if target_transform == 'log':
            y_pred_train = np.expm1(y_pred_train_transformed)
            y_pred_val = np.expm1(y_pred_val_transformed)
            y_pred_test = np.expm1(y_pred_test_transformed)
        elif target_transform == 'sqrt':
            y_pred_train = y_pred_train_transformed ** 2
            y_pred_val = y_pred_val_transformed ** 2
            y_pred_test = y_pred_test_transformed ** 2
        else:
            y_pred_train = y_pred_train_transformed
            y_pred_val = y_pred_val_transformed
            y_pred_test = y_pred_test_transformed

        # 성능 계산
        train_r2 = r2_score(y_train, y_pred_train)
        val_r2 = r2_score(y_val, y_pred_val)
        test_r2 = r2_score(y_test, y_pred_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

        print(f"\n📊 최종 성능:")
        print(f"   훈련 R²: {train_r2:.4f} | RMSE: {train_rmse:.3f}")
        print(f"   검증 R²: {val_r2:.4f} | RMSE: {val_rmse:.3f}")
        print(f"   테스트 R²: {test_r2:.4f} | RMSE: {test_rmse:.3f}")

        # 9. 모델 저장
        model_data = {
            'best_model': best_model,
            'scaler': scaler,
            'target_transform': target_transform,
            'feature_columns': selected_features,
            'best_params': best_params,
            'performance': {
                'train_r2': train_r2,
                'val_r2': val_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'test_rmse': test_rmse
            },
            'training_date': datetime.now().isoformat()
        }

        joblib.dump(model_data, 'combined_advanced_area_model_final.joblib')

        # 피처 정보 저장
        with open('combined_area_features.json', 'w') as f:
            json.dump(selected_features, f, indent=2)

        print(f"\n💾 모델 저장 완료:")
        print(f"   - combined_advanced_area_model_final.joblib")
        print(f"   - combined_area_features.json")
        print(f"   - 최종 성능: {test_r2:.1%} R²")

        return model_data

def main():
    """메인 실행 함수"""
    trainer = CombinedAdvancedAreaModel()
    model_data = trainer.train_combined_model()

    print("\n🎉 통합 고급 면적 모델 훈련 완료!")
    print(f"🏆 최종 성능: {model_data['performance']['test_r2']:.1%} R²")

if __name__ == "__main__":
    main()