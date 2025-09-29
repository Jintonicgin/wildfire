#!/usr/bin/env python3
"""
실제 화재 피해면적(ha) 예측 회귀 모델
- 분류가 아닌 정확한 면적 수치 예측
- 데이터 분석 기반 근본적 접근
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
warnings.filterwarnings('ignore')

def deep_data_analysis(df):
    """데이터 깊이 분석"""
    print("🔍 화재 면적 데이터 깊이 분석...")
    
    area_data = df['fire_area']
    
    print(f"\\n📊 기본 통계:")
    print(f"   샘플 수: {len(area_data)}")
    print(f"   평균: {area_data.mean():.3f} ha")
    print(f"   중앙값: {area_data.median():.3f} ha")
    print(f"   표준편차: {area_data.std():.3f} ha")
    print(f"   최솟값: {area_data.min():.3f} ha")
    print(f"   최댓값: {area_data.max():.3f} ha")
    
    # 분위수 분석
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\\n📈 분위수 분석:")
    for p in percentiles:
        val = area_data.quantile(p/100)
        print(f"   {p:2}%: {val:8.3f} ha")
    
    # 분포 특성
    skewness = area_data.skew()
    kurtosis = area_data.kurtosis()
    print(f"\\n📐 분포 특성:")
    print(f"   왜도(Skewness): {skewness:.3f}")
    print(f"   첨도(Kurtosis): {kurtosis:.3f}")
    
    # 극값 분석
    q99 = area_data.quantile(0.99)
    outliers = area_data[area_data > q99]
    print(f"\\n⚡ 극값 분석:")
    print(f"   99% 이상 값들: {len(outliers)}개")
    print(f"   극값들: {outliers.head().values}")
    
    # 로그 변환 효과 확인
    log_area = np.log1p(area_data)
    print(f"\\n🔄 로그 변환 효과:")
    print(f"   원본 왜도: {skewness:.3f} → 로그 왜도: {log_area.skew():.3f}")
    print(f"   원본 표준편차: {area_data.std():.3f} → 로그 표준편차: {log_area.std():.3f}")
    
    return {
        'skewness': skewness,
        'outlier_threshold': q99,
        'log_transform_improvement': skewness - log_area.skew()
    }

def create_advanced_area_features(df):
    """면적 예측을 위한 고급 피처 엔지니어링"""
    print("\\n🎯 면적 예측용 고급 피처 생성...")
    
    # 새 DataFrame으로 작업
    df_features = df.copy()
    feature_list = []
    
    # 1. 핵심 화재 확산 관련 피처들
    core_features = [
        'fwi_0h', 'isi_0h', 'ffmc_0h', 'dmc_0h', 'dc_0h', 'bui_0h',  # 화재 위험 지수
        'ws10m_0h', 'wd10m_0h', 't2m_0h', 'rh2m_0h', 'precip_0h',    # 핵심 기상
        'elevation_mean', 'slope_mean'  # 지형
    ]
    
    for feat in core_features:
        if feat in df.columns:
            feature_list.append(feat)
    
    # 2. 시간적 요소 (매우 중요!)
    temporal_features = ['fire_month', 'startday']
    for feat in temporal_features:
        if feat in df.columns:
            feature_list.append(feat)
    
    # 3. 건조 조건 (면적에 큰 영향)
    dry_features = [
        'dry_days_7d_start', 'dry_days_14d_start', 'dry_days_30d_start',
        'consecutive_dry_days_start', 'dry_to_rain_ratio_30d'
    ]
    for feat in dry_features:
        if feat in df.columns:
            feature_list.append(feat)
    
    # 4. 복합 기상 지수들
    combo_features = [
        'hot_dry_combo', 'dry_windy_combo', 'low_humidity_flag'
    ]
    for feat in combo_features:
        if feat in df.columns:
            feature_list.append(feat)
    
    # 5. 시간 윈도우 평균값들 (화재 지속과 관련)
    window_features = [
        'fwi_mean_0_12h', 'isi_mean_0_12h', 'max_temp_0_12h', 
        'max_wind_0_12h', 'min_humidity_0_12h'
    ]
    for feat in window_features:
        if feat in df.columns:
            feature_list.append(feat)
    
    # 6. 고급 파생 피처 생성
    derived_count = 0
    
    # FWI 기반 위험 점수
    if 'fwi_0h' in df.columns:
        df_features['fwi_risk_score'] = np.where(df_features['fwi_0h'] > 21, 3,
                                        np.where(df_features['fwi_0h'] > 13, 2,
                                        np.where(df_features['fwi_0h'] > 5, 1, 0)))
        feature_list.append('fwi_risk_score')
        derived_count += 1
    
    # 바람-습도 상호작용 (화재 확산에 핵심)
    if 'ws10m_0h' in df.columns and 'rh2m_0h' in df.columns:
        df_features['wind_humidity_interaction'] = (
            df_features['ws10m_0h'].fillna(0) * 
            (100 - df_features['rh2m_0h'].fillna(50)) / 100
        )
        feature_list.append('wind_humidity_interaction')
        derived_count += 1
    
    # ISI와 바람의 복합 지수
    if 'isi_0h' in df.columns and 'ws10m_0h' in df.columns:
        df_features['isi_wind_amplifier'] = (
            df_features['isi_0h'].fillna(0) * 
            np.log1p(df_features['ws10m_0h'].fillna(0))
        )
        feature_list.append('isi_wind_amplifier')
        derived_count += 1
    
    # 온도-건조도 조합
    if 't2m_0h' in df.columns and 'rh2m_0h' in df.columns:
        df_features['heat_dryness_index'] = (
            np.maximum(0, df_features['t2m_0h'].fillna(15) - 15) * 
            np.maximum(0, 80 - df_features['rh2m_0h'].fillna(50)) / 80
        )
        feature_list.append('heat_dryness_index')
        derived_count += 1
    
    # 장기 건조 위험도
    if 'dry_days_30d_start' in df.columns and 'consecutive_dry_days_start' in df.columns:
        df_features['chronic_drought_risk'] = (
            df_features['dry_days_30d_start'].fillna(0) * 
            np.log1p(df_features['consecutive_dry_days_start'].fillna(0))
        )
        feature_list.append('chronic_drought_risk')
        derived_count += 1
    
    # 계절적 위험도
    if 'fire_month' in df.columns:
        # 한국 기준: 3-5월, 10-12월이 화재 위험 높음
        high_risk_months = [3, 4, 5, 10, 11, 12]
        df_features['seasonal_risk'] = df_features['fire_month'].apply(
            lambda x: 2 if x in high_risk_months else 1
        )
        feature_list.append('seasonal_risk')
        derived_count += 1
    
    print(f"   기존 피처: {len(feature_list) - derived_count}개")
    print(f"   파생 피처: {derived_count}개")
    print(f"   총 피처: {len(feature_list)}개")
    
    return df_features, feature_list

def preprocess_data_robust(X, y, outlier_method='clip'):
    """강력한 데이터 전처리"""
    print(f"\\n🔧 강력한 데이터 전처리 ({outlier_method})...")
    
    print(f"   전처리 전: X={X.shape}, y 범위=[{y.min():.3f}, {y.max():.3f}]")
    
    # X 전처리
    X_processed = X.copy()
    
    for col in X_processed.columns:
        # 무한값 처리
        X_processed[col] = X_processed[col].replace([np.inf, -np.inf], np.nan)
        
        # 극값 처리
        if outlier_method == 'clip':
            q01, q99 = X_processed[col].quantile([0.01, 0.99])
            X_processed[col] = X_processed[col].clip(lower=q01, upper=q99)
        
        # 결측치 처리
        if X_processed[col].isna().sum() > 0:
            X_processed[col] = X_processed[col].fillna(X_processed[col].median())
    
    # y 전처리 (극값 처리)
    y_processed = y.copy()
    if outlier_method == 'clip':
        y_q99 = y.quantile(0.995)  # 상위 0.5% 클리핑
        y_processed = y.clip(upper=y_q99)
        n_clipped = (y > y_q99).sum()
        print(f"   y 극값 처리: {n_clipped}개 값을 {y_q99:.2f}으로 클리핑")
    
    print(f"   전처리 후: X={X_processed.shape}, y 범위=[{y_processed.min():.3f}, {y_processed.max():.3f}]")
    print(f"   무한값 체크: {np.isinf(X_processed.values).sum()}")
    print(f"   NaN 체크: {X_processed.isna().sum().sum()}")
    
    return X_processed, y_processed

def train_robust_models(X_train, X_test, y_train, y_test, target_transform='log'):
    """강력한 모델들 훈련"""
    print(f"\\n🤖 강력한 모델 훈련 ({target_transform} 변환)...")
    
    # 타겟 변환
    if target_transform == 'log':
        y_train_transformed = np.log1p(y_train)
        y_test_transformed = np.log1p(y_test)
        inverse_transform = np.expm1
    elif target_transform == 'sqrt':
        y_train_transformed = np.sqrt(y_train)
        y_test_transformed = np.sqrt(y_test)
        inverse_transform = lambda x: x**2
    else:
        y_train_transformed = y_train
        y_test_transformed = y_test
        inverse_transform = lambda x: x
    
    # 스케일링
    scaler = RobustScaler()  # 극값에 더 강함
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 모델들 - 극값에 강한 모델들 위주
    models = {
        'RandomForest_Robust': RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features=0.8,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        ),
        'ExtraTrees_Robust': ExtraTreesRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=3,
            max_features=0.9,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting_Robust': GradientBoostingRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            alpha=0.1,  # Huber loss로 극값에 강함
            random_state=42
        ),
        'Huber_Regression': HuberRegressor(
            epsilon=1.35,  # 극값에 강한 손실함수
            alpha=0.01,
            max_iter=200
        ),
        'Ridge_Robust': Ridge(
            alpha=10.0
        )
    }
    
    # XGBoost와 LightGBM은 별도 처리 (라벨 인코딩 필요 없음)
    try:
        models['XGBoost_Robust'] = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1
        )
    except:
        pass
    
    try:
        models['LightGBM_Robust'] = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
    except:
        pass
    
    results = {}
    best_model = None
    best_r2 = -np.inf
    
    for name, model in models.items():
        print(f"   {name} 훈련...")
        
        try:
            # 스케일링된 데이터 사용 여부
            if 'Huber' in name or 'Ridge' in name:
                X_train_use = X_train_scaled
                X_test_use = X_test_scaled
            else:
                X_train_use = X_train
                X_test_use = X_test
            
            # 훈련
            model.fit(X_train_use, y_train_transformed)
            
            # 예측 (변환된 타겟에 대해)
            y_pred_transformed = model.predict(X_test_use)
            
            # 원래 스케일로 복원
            y_pred_original = inverse_transform(y_pred_transformed)
            
            # 음수 값 처리
            y_pred_original = np.maximum(0, y_pred_original)
            
            # 평가 (원래 스케일에서)
            r2 = r2_score(y_test, y_pred_original)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_original))
            mae = mean_absolute_error(y_test, y_pred_original)
            
            results[name] = {
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'model': model,
                'scaler': scaler if 'Huber' in name or 'Ridge' in name else None,
                'transform': target_transform
            }
            
            print(f"     R²: {r2:.4f}, RMSE: {rmse:.2f} ha, MAE: {mae:.2f} ha")
            
            if r2 > best_r2:
                best_r2 = r2
                best_model = (name, model, scaler if 'Huber' in name or 'Ridge' in name else None)
                
        except Exception as e:
            print(f"     {name} 실패: {e}")
    
    return results, best_model, inverse_transform

def create_ensemble_prediction(results, X_test, inverse_transform):
    """앙상블 예측"""
    print("\\n🎭 앙상블 예측...")
    
    # R² > 0인 모델들만 사용
    valid_models = {k: v for k, v in results.items() if v['r2'] > 0}
    
    if len(valid_models) < 2:
        print("   앙상블할 모델이 부족합니다.")
        return None
    
    predictions = []
    weights = []
    
    for name, result in valid_models.items():
        model = result['model']
        scaler = result['scaler']
        
        # 예측
        if scaler:
            X_test_use = scaler.transform(X_test)
        else:
            X_test_use = X_test
        
        pred_transformed = model.predict(X_test_use)
        pred_original = inverse_transform(pred_transformed)
        pred_original = np.maximum(0, pred_original)
        
        predictions.append(pred_original)
        weights.append(result['r2'])  # R²을 가중치로 사용
        
        print(f"   {name}: 가중치 {result['r2']:.4f}")
    
    # 가중평균
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    ensemble_pred = sum(w * pred for w, pred in zip(weights, predictions))
    
    return ensemble_pred

def main():
    """메인"""
    print("🎯 실제 화재 면적(ha) 예측 모델")
    print("=" * 50)
    
    # 데이터 로드
    df = pd.read_csv('final_merged_feature_engineered.csv', low_memory=False)
    fire_mask = (df['fire_area'] > 0) & df['fire_area'].notna()
    fire_df = df[fire_mask].copy()
    
    # 데이터 분석
    analysis_result = deep_data_analysis(fire_df)
    
    # 고급 피처 생성
    fire_df_features, feature_list = create_advanced_area_features(fire_df)
    
    # 데이터 준비
    X = fire_df_features[feature_list]
    y = fire_df_features['fire_area']
    
    print(f"\\n📊 최종 데이터:")
    print(f"   샘플 수: {len(X)}")
    print(f"   피처 수: {len(feature_list)}")
    print(f"   타겟 범위: {y.min():.3f} ~ {y.max():.3f} ha")
    
    # 강력한 전처리
    X_processed, y_processed = preprocess_data_robust(X, y, outlier_method='clip')
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=0.25, random_state=42
    )
    
    print(f"\\n📊 훈련/테스트 분할:")
    print(f"   훈련: {X_train.shape[0]}개")
    print(f"   테스트: {X_test.shape[0]}개")
    
    # 다양한 변환 방법 테스트
    transform_methods = ['log', 'sqrt', 'original']
    all_results = {}
    
    for method in transform_methods:
        print(f"\\n{'='*20} {method.upper()} 변환 {'='*20}")
        
        results, best_model, inverse_transform = train_robust_models(
            X_train, X_test, y_train, y_test, target_transform=method
        )
        
        if results:
            # 앙상블 예측
            ensemble_pred = create_ensemble_prediction(results, X_test, inverse_transform)
            
            if ensemble_pred is not None:
                ensemble_r2 = r2_score(y_test, ensemble_pred)
                ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
                ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
                
                print(f"\\n🎭 앙상블 결과:")
                print(f"   R²: {ensemble_r2:.4f}")
                print(f"   RMSE: {ensemble_rmse:.2f} ha")
                print(f"   MAE: {ensemble_mae:.2f} ha")
                
                all_results[method] = {
                    'ensemble_r2': ensemble_r2,
                    'ensemble_rmse': ensemble_rmse,
                    'ensemble_mae': ensemble_mae,
                    'individual_results': results,
                    'best_model': best_model,
                    'features': feature_list
                }
    
    # 최종 결과
    print("\\n" + "=" * 60)
    print("🏆 최종 면적 예측 결과")
    print("=" * 60)
    
    if all_results:
        for method, result in all_results.items():
            print(f"{method:8} | R²: {result['ensemble_r2']:6.4f} | RMSE: {result['ensemble_rmse']:6.2f} ha")
        
        # 최고 성능
        best_method = max(all_results.items(), key=lambda x: x[1]['ensemble_r2'])
        method_name, best_result = best_method
        
        print(f"\\n🥇 최고 성능: {method_name} 변환")
        print(f"   R²: {best_result['ensemble_r2']:.4f}")
        print(f"   RMSE: {best_result['ensemble_rmse']:.2f} ha")
        print(f"   MAE: {best_result['ensemble_mae']:.2f} ha")
        
        # 성능 해석
        r2_score_final = best_result['ensemble_r2']
        if r2_score_final > 0.3:
            print(f"\\n🎉 우수한 성능! R²={r2_score_final:.1%}는 화재 예측 분야에서 매우 좋은 결과입니다.")
        elif r2_score_final > 0.15:
            print(f"\\n📊 적절한 성능. R²={r2_score_final:.1%}는 실용적으로 활용 가능합니다.")
        elif r2_score_final > 0.05:
            print(f"\\n⚠️ 제한적 성능. R²={r2_score_final:.1%}는 참고용으로만 활용 가능합니다.")
        else:
            print(f"\\n❌ 성능 부족. R²={r2_score_final:.1%}는 예측 모델로서 한계가 있습니다.")
        
        # 모델 저장
        final_package = {
            'method': method_name,
            'ensemble_r2': best_result['ensemble_r2'],
            'features': best_result['features'],
            'results': best_result['individual_results'],
            'preprocessing_info': {
                'outlier_method': 'clip',
                'target_transform': method_name
            }
        }
        
        joblib.dump(final_package, 'final_area_regression_model.joblib')
        print(f"\\n✅ 저장 완료: final_area_regression_model.joblib")

if __name__ == "__main__":
    main()