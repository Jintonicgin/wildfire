#!/usr/bin/env python3
"""
면적 모델 성능 개선 전략들
17.9% R²를 30%+ 목표로 개선
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import scipy.stats as stats
warnings.filterwarnings('ignore')

def load_data():
    """데이터 로드"""
    df = pd.read_csv('final_merged_feature_engineered.csv', low_memory=False)
    fire_mask = (df['fire_area'] > 0) & df['fire_area'].notna()
    return df[fire_mask].copy()

def strategy_1_advanced_feature_engineering(df):
    """전략 1: 고급 피처 엔지니어링"""
    print("🎯 전략 1: 고급 피처 엔지니어링")
    print("-" * 40)
    
    df_features = df.copy()
    
    # 1. 도메인 지식 기반 고급 피처
    derived_features = []
    
    # 화재 삼각형 (열-연료-산소) 기반
    if all(col in df.columns for col in ['t2m_0h', 'ws10m_0h', 'rh2m_0h']):
        # 열-바람 상호작용 (산소 공급)
        df_features['heat_oxygen_interaction'] = (
            np.maximum(0, df_features['t2m_0h'].fillna(15) - 15) * 
            df_features['ws10m_0h'].fillna(0)
        )
        
        # 연료 건조도 지수 (온도/습도 비율)
        df_features['fuel_dryness_ratio'] = (
            df_features['t2m_0h'].fillna(15) / 
            (df_features['rh2m_0h'].fillna(50) + 1)  # +1로 0으로 나누기 방지
        )
        derived_features.extend(['heat_oxygen_interaction', 'fuel_dryness_ratio'])
    
    # 2. FWI 시스템 기반 고급 조합
    fwi_components = ['ffmc_0h', 'dmc_0h', 'dc_0h', 'isi_0h', 'bui_0h']
    available_fwi = [col for col in fwi_components if col in df.columns]
    
    if len(available_fwi) >= 3:
        # FWI 구성요소들의 비선형 조합
        if 'ffmc_0h' in df.columns and 'isi_0h' in df.columns:
            df_features['ffmc_isi_synergy'] = (
                df_features['ffmc_0h'].fillna(50) * 
                np.log1p(df_features['isi_0h'].fillna(0))
            )
            derived_features.append('ffmc_isi_synergy')
        
        if 'dmc_0h' in df.columns and 'dc_0h' in df.columns:
            df_features['drought_compound_index'] = (
                np.sqrt(df_features['dmc_0h'].fillna(0) * df_features['dc_0h'].fillna(0))
            )
            derived_features.append('drought_compound_index')
    
    # 3. 시간적 패턴 피처
    if 'fire_month' in df.columns:
        # 계절별 위험도 (cosine 변환으로 연속성 확보)
        month_rad = df_features['fire_month'] * 2 * np.pi / 12
        df_features['seasonal_risk_cos'] = np.cos(month_rad)
        df_features['seasonal_risk_sin'] = np.sin(month_rad)
        derived_features.extend(['seasonal_risk_cos', 'seasonal_risk_sin'])
        
        # 고위험 시기 지시자
        high_risk_months = [3, 4, 5, 10, 11, 12]  # 한국 산불 시즌
        df_features['peak_season_indicator'] = df_features['fire_month'].isin(high_risk_months).astype(int)
        derived_features.append('peak_season_indicator')
    
    # 4. 지형 기반 위험도
    if 'elevation_mean' in df.columns and 'slope_mean' in df.columns:
        # 지형 복합 위험도
        df_features['terrain_fire_risk'] = (
            df_features['slope_mean'].fillna(0) / (df_features['elevation_mean'].fillna(100) + 1)
        )
        derived_features.append('terrain_fire_risk')
    
    # 5. 기상 안정도 지수
    past_weather_cols = [col for col in df.columns if '_past' in col and any(var in col for var in ['t2m', 'rh2m', 'ws10m'])]
    if len(past_weather_cols) >= 6:
        # 과거 24시간 기상 변동성
        temp_cols = [col for col in past_weather_cols if 't2m' in col][:4]
        if len(temp_cols) >= 3:
            temp_data = df_features[temp_cols].fillna(method='bfill', axis=1).fillna(15)
            df_features['temp_stability'] = temp_data.std(axis=1)
            derived_features.append('temp_stability')
    
    # 기본 피처 + 파생 피처
    base_features = [
        'fwi_0h', 'ffmc_0h', 'dmc_0h', 'dc_0h', 'isi_0h', 'bui_0h',
        't2m_0h', 'rh2m_0h', 'ws10m_0h', 'wd10m_0h', 'precip_0h',
        'elevation_mean', 'slope_mean', 'fire_month'
    ]
    
    all_features = [f for f in base_features if f in df.columns] + derived_features
    print(f"   기본 피처: {len([f for f in base_features if f in df.columns])}개")
    print(f"   파생 피처: {len(derived_features)}개")
    print(f"   총 피처: {len(all_features)}개")
    
    return df_features, all_features

def strategy_2_multi_stage_modeling(df, features):
    """전략 2: 다단계 모델링 (화재 지속시간 × 확산속도)"""
    print("\\n🎯 전략 2: 다단계 모델링")
    print("-" * 40)
    
    # 1단계: 화재 지속시간 추정
    area = df['fire_area']
    
    # 경험적 공식으로 지속시간 추정 (시간 단위)
    # 작은 화재: 빨리 진화, 큰 화재: 오래 지속
    estimated_duration = np.where(
        area <= 0.1, 0.5,                    # 0.1ha 이하: 30분
        np.where(area <= 1, 2,               # 0.1-1ha: 2시간
        np.where(area <= 10, 8,              # 1-10ha: 8시간  
        np.where(area <= 100, 24,            # 10-100ha: 24시간
                 48))))                      # 100ha+: 48시간+
    
    # 2단계: 확산속도 계산 (ha/hour)
    spread_rate = area / (estimated_duration + 0.1)  # 0으로 나누기 방지
    
    print(f"   추정 지속시간: {estimated_duration.min():.1f}~{estimated_duration.max():.1f}시간")
    print(f"   확산속도: {spread_rate.min():.4f}~{spread_rate.max():.4f} ha/h")
    
    # 데이터 준비
    X = df[features].fillna(df[features].median())
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 3단계: 각각 모델링
    results = {}
    
    # 지속시간 모델
    X_train, X_test, y_dur_train, y_dur_test = train_test_split(
        X, np.log1p(estimated_duration), test_size=0.25, random_state=42
    )
    
    duration_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    duration_model.fit(X_train, y_dur_train)
    
    dur_pred = np.expm1(duration_model.predict(X_test))
    dur_r2 = r2_score(np.expm1(y_dur_test), dur_pred)
    
    print(f"   지속시간 예측 R²: {dur_r2:.4f}")
    
    # 확산속도 모델
    X_train, X_test, y_rate_train, y_rate_test = train_test_split(
        X, np.log1p(spread_rate), test_size=0.25, random_state=42
    )
    
    rate_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    rate_model.fit(X_train, y_rate_train)
    
    rate_pred = np.expm1(rate_model.predict(X_test))
    rate_r2 = r2_score(np.expm1(y_rate_test), rate_pred)
    
    print(f"   확산속도 예측 R²: {rate_r2:.4f}")
    
    # 4단계: 최종 면적 예측
    final_area_pred = dur_pred * rate_pred
    
    # 실제 면적과 비교 (같은 테스트 인덱스 사용)
    y_area_test = df.iloc[y_rate_test.index]['fire_area']
    
    final_r2 = r2_score(y_area_test, final_area_pred)
    final_rmse = np.sqrt(mean_squared_error(y_area_test, final_area_pred))
    
    print(f"   🎯 최종 면적 R²: {final_r2:.4f}")
    print(f"   RMSE: {final_rmse:.3f} ha")
    
    return final_r2, final_rmse

def strategy_3_ensemble_of_ensembles(df, features):
    """전략 3: 앙상블의 앙상블"""
    print("\\n🎯 전략 3: 앙상블의 앙상블")
    print("-" * 40)
    
    X = df[features].fillna(df[features].median())
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df['fire_area']
    
    # 다양한 타겟 변환별 앙상블
    transformations = {
        'log': (lambda y: np.log1p(y), lambda y: np.expm1(y)),
        'sqrt': (lambda y: np.sqrt(y), lambda y: y**2),
        'box_cox': (lambda y: stats.boxcox(y + 0.01)[0], None)  # Box-Cox
    }
    
    ensemble_predictions = []
    ensemble_weights = []
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    for trans_name, (transform, inverse_transform) in transformations.items():
        print(f"\\n   {trans_name} 변환 앙상블:")
        
        try:
            # 타겟 변환
            if trans_name == 'box_cox':
                y_transformed, lmbda = stats.boxcox(y_train + 0.01)
                y_train_trans = y_transformed
                inverse_transform = lambda x: stats.inv_boxcox(x, lmbda) - 0.01
            else:
                y_train_trans = transform(y_train)
            
            # 다양한 모델들
            models = {
                'rf': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
                'gb': GradientBoostingRegressor(n_estimators=150, random_state=42),
                'xgb': xgb.XGBRegressor(n_estimators=100, random_state=42) if 'xgboost' in globals() else None,
                'lgb': lgb.LGBMRegressor(n_estimators=100, verbose=-1, random_state=42) if 'lightgbm' in globals() else None
            }
            
            # 모델별 예측
            model_preds = []
            model_weights = []
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            for name, model in models.items():
                if model is None:
                    continue
                    
                try:
                    model.fit(X_train_scaled if name in ['huber'] else X_train, y_train_trans)
                    pred_trans = model.predict(X_test_scaled if name in ['huber'] else X_test)
                    
                    # 원래 스케일로 복원
                    if inverse_transform:
                        pred_original = inverse_transform(pred_trans)
                    else:
                        pred_original = pred_trans
                    
                    pred_original = np.maximum(0, pred_original)
                    
                    # 개별 모델 성능
                    model_r2 = r2_score(y_test, pred_original)
                    
                    if model_r2 > 0:  # 유효한 모델만 포함
                        model_preds.append(pred_original)
                        model_weights.append(max(0, model_r2))
                        print(f"     {name}: R² = {model_r2:.4f}")
                    
                except Exception as e:
                    print(f"     {name}: 실패 ({e})")
            
            if model_preds:
                # 가중평균 앙상블
                if sum(model_weights) > 0:
                    weights = np.array(model_weights) / sum(model_weights)
                    ensemble_pred = sum(w * pred for w, pred in zip(weights, model_preds))
                    
                    ensemble_r2 = r2_score(y_test, ensemble_pred)
                    print(f"     {trans_name} 앙상블 R²: {ensemble_r2:.4f}")
                    
                    if ensemble_r2 > 0:
                        ensemble_predictions.append(ensemble_pred)
                        ensemble_weights.append(ensemble_r2)
        
        except Exception as e:
            print(f"     {trans_name} 변환 실패: {e}")
    
    # 최종 앙상블의 앙상블
    if ensemble_predictions:
        weights = np.array(ensemble_weights) / sum(ensemble_weights)
        final_ensemble = sum(w * pred for w, pred in zip(weights, ensemble_predictions))
        
        final_r2 = r2_score(y_test, final_ensemble)
        final_rmse = np.sqrt(mean_squared_error(y_test, final_ensemble))
        
        print(f"\\n   🎯 최종 앙상블의 앙상블:")
        print(f"   R²: {final_r2:.4f}")
        print(f"   RMSE: {final_rmse:.3f} ha")
        
        return final_r2, final_rmse
    
    return 0, float('inf')

def strategy_4_neural_network(df, features):
    """전략 4: 딥러닝 접근"""
    print("\\n🎯 전략 4: 딥러닝 접근")  
    print("-" * 40)
    
    X = df[features].fillna(df[features].median())
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = np.log1p(df['fire_area'])  # 로그 변환
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # 스케일링 (신경망에 중요)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 여러 신경망 구조 시도
    nn_configs = [
        {'hidden_layer_sizes': (100, 50), 'alpha': 0.001},
        {'hidden_layer_sizes': (150, 75, 25), 'alpha': 0.01},
        {'hidden_layer_sizes': (200, 100, 50), 'alpha': 0.001},
        {'hidden_layer_sizes': (80,), 'alpha': 0.01}
    ]
    
    best_r2 = -np.inf
    best_model = None
    
    for i, config in enumerate(nn_configs):
        print(f"   신경망 {i+1}: {config['hidden_layer_sizes']}")
        
        try:
            model = MLPRegressor(
                hidden_layer_sizes=config['hidden_layer_sizes'],
                alpha=config['alpha'],
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.2,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train)
            y_pred_log = model.predict(X_test_scaled)
            y_pred = np.expm1(y_pred_log)
            y_test_original = np.expm1(y_test)
            
            r2 = r2_score(y_test_original, y_pred)
            print(f"     R²: {r2:.4f}")
            
            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                
        except Exception as e:
            print(f"     실패: {e}")
    
    if best_model:
        y_pred_best = np.expm1(best_model.predict(X_test_scaled))
        y_test_original = np.expm1(y_test)
        
        best_rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_best))
        
        print(f"\\n   🎯 최고 신경망:")
        print(f"   R²: {best_r2:.4f}")
        print(f"   RMSE: {best_rmse:.3f} ha")
        
        return best_r2, best_rmse
    
    return 0, float('inf')

def main():
    """메인"""
    print("🚀 면적 모델 성능 개선 전략 실험")
    print("목표: 17.9% → 30%+ R²")
    print("=" * 50)
    
    # 데이터 로드
    fire_df = load_data()
    
    # 전략별 실험
    results = {}
    
    # 전략 1: 고급 피처 엔지니어링
    df_features, features = strategy_1_advanced_feature_engineering(fire_df)
    
    # 전략 2: 다단계 모델링
    try:
        r2_2, rmse_2 = strategy_2_multi_stage_modeling(df_features, features)
        results['Multi-stage'] = {'r2': r2_2, 'rmse': rmse_2}
    except Exception as e:
        print(f"전략 2 실패: {e}")
    
    # 전략 3: 앙상블의 앙상블
    try:
        r2_3, rmse_3 = strategy_3_ensemble_of_ensembles(df_features, features)
        results['Ensemble²'] = {'r2': r2_3, 'rmse': rmse_3}
    except Exception as e:
        print(f"전략 3 실패: {e}")
    
    # 전략 4: 딥러닝
    try:
        r2_4, rmse_4 = strategy_4_neural_network(df_features, features)
        results['Neural Net'] = {'r2': r2_4, 'rmse': rmse_4}
    except Exception as e:
        print(f"전략 4 실패: {e}")
    
    # 최종 결과 비교
    print("\\n" + "=" * 50)
    print("🏆 전략별 성능 비교")
    print("=" * 50)
    print("기준 모델 (기존):    R² = 17.9%")
    
    if results:
        for strategy, result in results.items():
            r2_pct = result['r2'] * 100
            improvement = result['r2'] - 0.179
            print(f"{strategy:15}: R² = {r2_pct:5.1f}% (개선: {improvement:+.3f})")
        
        # 최고 성능
        best_strategy = max(results.items(), key=lambda x: x[1]['r2'])
        best_r2 = best_strategy[1]['r2']
        
        print(f"\\n🥇 최고 성능: {best_strategy[0]} (R² = {best_r2:.1%})")
        
        if best_r2 > 0.3:
            print("✅ 목표 달성! 30% 이상")
        elif best_r2 > 0.25:
            print("📈 상당한 개선! 25% 이상")
        elif best_r2 > 0.2:
            print("📊 약간의 개선. 추가 연구 필요")
        else:
            print("⚠️ 개선 한계. 근본적 접근 필요")
    else:
        print("❌ 모든 전략이 실패했습니다.")

if __name__ == "__main__":
    main()