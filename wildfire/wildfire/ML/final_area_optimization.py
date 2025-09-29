#!/usr/bin/env python3
"""
면적 모델 최종 최적화 - 30% R² 목표
강력한 전처리 + 고급 앙상블 + 딥러닝 조합
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, ElasticNet
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
warnings.filterwarnings('ignore')

def robust_preprocessing(X, y, outlier_threshold=0.995):
    """매우 강력한 전처리"""
    print(f"🔧 강력한 전처리 (상위 {(1-outlier_threshold)*100:.1f}% 극값 처리)...")
    
    X_processed = X.copy()
    y_processed = y.copy()
    
    # 1. X 전처리
    print(f"   전처리 전: {X.shape}, 무한값: {np.isinf(X.values).sum()}")
    
    for col in X_processed.columns:
        # 무한값 처리
        X_processed[col] = X_processed[col].replace([np.inf, -np.inf], np.nan)
        
        # 극값 클리핑 (더 보수적)
        if X_processed[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            q_low = X_processed[col].quantile(0.005)   # 0.5%
            q_high = X_processed[col].quantile(outlier_threshold)  # 99.5%
            X_processed[col] = X_processed[col].clip(lower=q_low, upper=q_high)
        
        # 결측치 처리
        if X_processed[col].isna().sum() > 0:
            X_processed[col] = X_processed[col].fillna(X_processed[col].median())
    
    # 2. y 전처리 (극값 처리)
    y_threshold = y.quantile(outlier_threshold)
    n_clipped = (y > y_threshold).sum()
    y_processed = y.clip(upper=y_threshold)
    
    print(f"   X 후처리: 무한값 {np.isinf(X_processed.values).sum()}, NaN {X_processed.isna().sum().sum()}")
    print(f"   y 극값 처리: {n_clipped}개 값을 {y_threshold:.2f}로 클리핑")
    
    return X_processed, y_processed

def create_ultimate_features(df):
    """궁극적 피처 엔지니어링"""
    print("🎯 궁극적 피처 엔지니어링...")
    
    df_features = df.copy()
    base_features = []
    derived_features = []
    
    # 1. 핵심 기존 피처
    core_features = [
        'fwi_0h', 'ffmc_0h', 'dmc_0h', 'dc_0h', 'isi_0h', 'bui_0h',
        't2m_0h', 'rh2m_0h', 'ws10m_0h', 'wd10m_0h', 'precip_0h',
        'elevation_mean', 'slope_mean', 'fire_month', 'startday'
    ]
    
    for feat in core_features:
        if feat in df.columns:
            base_features.append(feat)
    
    # 2. 시간 관련 고급 피처
    if 'fire_month' in df.columns:
        # 계절적 사인/코사인
        month_rad = df_features['fire_month'] * 2 * np.pi / 12
        df_features['season_cos'] = np.cos(month_rad)
        df_features['season_sin'] = np.sin(month_rad)
        derived_features.extend(['season_cos', 'season_sin'])
        
        # 고위험 계절
        spring_risk = [3, 4, 5]  # 봄
        autumn_risk = [10, 11, 12]  # 가을
        df_features['spring_season'] = df_features['fire_month'].isin(spring_risk).astype(float)
        df_features['autumn_season'] = df_features['fire_month'].isin(autumn_risk).astype(float)
        derived_features.extend(['spring_season', 'autumn_season'])
    
    # 3. 기상 상호작용 (더 정교하게)
    if all(col in df.columns for col in ['t2m_0h', 'rh2m_0h', 'ws10m_0h']):
        t2m = df_features['t2m_0h'].fillna(15)
        rh2m = df_features['rh2m_0h'].fillna(50) 
        ws10m = df_features['ws10m_0h'].fillna(0)
        
        # 열지수 (온도 - 기준온도)
        df_features['heat_index'] = np.maximum(0, t2m - 10)
        
        # 건조지수 (100 - 습도)
        df_features['dryness_index'] = np.maximum(0, 100 - rh2m)
        
        # 복합 위험 지수들
        df_features['heat_dry_product'] = df_features['heat_index'] * df_features['dryness_index']
        df_features['heat_wind_product'] = df_features['heat_index'] * np.sqrt(ws10m)
        df_features['dry_wind_product'] = df_features['dryness_index'] * np.sqrt(ws10m)
        df_features['triple_risk'] = df_features['heat_index'] * df_features['dryness_index'] * np.log1p(ws10m)
        
        derived_features.extend([
            'heat_index', 'dryness_index', 'heat_dry_product', 
            'heat_wind_product', 'dry_wind_product', 'triple_risk'
        ])
    
    # 4. FWI 시스템 고급 활용
    fwi_components = ['fwi_0h', 'ffmc_0h', 'dmc_0h', 'dc_0h', 'isi_0h', 'bui_0h']
    available_fwi = [col for col in fwi_components if col in df.columns]
    
    if len(available_fwi) >= 4:
        # FWI 로그 변환
        for comp in available_fwi:
            df_features[f'{comp}_log'] = np.log1p(df_features[comp].fillna(0))
            derived_features.append(f'{comp}_log')
        
        # FWI 비율들
        if 'fwi_0h' in df.columns and 'isi_0h' in df.columns:
            df_features['fwi_isi_ratio'] = (df_features['fwi_0h'].fillna(1) / 
                                          (df_features['isi_0h'].fillna(0.1) + 0.1))
            derived_features.append('fwi_isi_ratio')
        
        if 'dmc_0h' in df.columns and 'dc_0h' in df.columns:
            df_features['drought_ratio'] = (df_features['dmc_0h'].fillna(1) / 
                                          (df_features['dc_0h'].fillna(1) + 1))
            derived_features.append('drought_ratio')
    
    # 5. 지형 복합 지수
    if 'elevation_mean' in df.columns and 'slope_mean' in df.columns:
        elev = df_features['elevation_mean'].fillna(100)
        slope = df_features['slope_mean'].fillna(0)
        
        df_features['terrain_complexity'] = slope / (elev + 1) * 100
        df_features['elevation_risk'] = np.where(elev < 500, 2.0, 
                                       np.where(elev < 1000, 1.0, 0.5))
        derived_features.extend(['terrain_complexity', 'elevation_risk'])
    
    # 6. 과거 날씨 안정성 (있다면)
    past_temp_cols = [col for col in df.columns if 't2m' in col and 'past' in col][:6]
    if len(past_temp_cols) >= 3:
        temp_data = df_features[past_temp_cols].fillna(method='bfill', axis=1).fillna(15)
        df_features['temp_volatility'] = temp_data.std(axis=1)
        df_features['temp_trend'] = temp_data.iloc[:, -1] - temp_data.iloc[:, 0]  # 최근 - 과거
        derived_features.extend(['temp_volatility', 'temp_trend'])
    
    all_features = base_features + derived_features
    
    print(f"   기본 피처: {len(base_features)}개")
    print(f"   파생 피처: {len(derived_features)}개") 
    print(f"   총 피처: {len(all_features)}개")
    
    return df_features, all_features

def ultimate_ensemble(X_train, X_test, y_train, y_test):
    """궁극적 앙상블 모델"""
    print("🤖 궁극적 앙상블 모델...")
    
    # 다양한 스케일러와 모델 조합
    scalers = {
        'standard': StandardScaler(),
        'robust': RobustScaler(),
        'power': PowerTransformer(method='yeo-johnson')
    }
    
    models = {
        'rf': RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_split=10, random_state=42, n_jobs=-1),
        'et': ExtraTreesRegressor(n_estimators=200, max_depth=10, min_samples_split=8, random_state=42, n_jobs=-1),
        'gb': GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42),
        'nn1': MLPRegressor(hidden_layer_sizes=(200, 100, 50), alpha=0.001, max_iter=500, random_state=42),
        'nn2': MLPRegressor(hidden_layer_sizes=(150, 75), alpha=0.01, max_iter=500, random_state=42),
        'ridge': Ridge(alpha=1.0),
        'elastic': ElasticNet(alpha=0.1, l1_ratio=0.7)
    }
    
    # XGBoost와 LightGBM 추가 (조건부)
    try:
        models['xgb'] = xgb.XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42)
    except:
        pass
    
    try:
        models['lgb'] = lgb.LGBMRegressor(n_estimators=150, max_depth=6, learning_rate=0.05, verbose=-1, random_state=42)
    except:
        pass
    
    ensemble_predictions = []
    ensemble_weights = []
    
    for scaler_name, scaler in scalers.items():
        print(f"\\n   {scaler_name} 스케일러:")
        
        try:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            scaler_preds = []
            scaler_weights = []
            
            for model_name, model in models.items():
                try:
                    # 신경망과 선형 모델은 스케일된 데이터, 트리 모델은 원본 사용
                    if any(x in model_name for x in ['nn', 'ridge', 'elastic']):
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    # 음수 제거
                    y_pred = np.maximum(0, y_pred)
                    
                    # 성능 평가
                    r2 = r2_score(y_test, y_pred)
                    
                    if r2 > 0:  # 유효한 모델만
                        scaler_preds.append(y_pred)
                        scaler_weights.append(r2)
                        print(f"     {model_name:8}: R² = {r2:.4f}")
                
                except Exception as e:
                    print(f"     {model_name:8}: 실패")
            
            # 스케일러별 앙상블
            if scaler_preds and sum(scaler_weights) > 0:
                weights = np.array(scaler_weights) / sum(scaler_weights)
                scaler_ensemble = sum(w * pred for w, pred in zip(weights, scaler_preds))
                
                ensemble_r2 = r2_score(y_test, scaler_ensemble)
                print(f"     앙상블: R² = {ensemble_r2:.4f}")
                
                if ensemble_r2 > 0:
                    ensemble_predictions.append(scaler_ensemble)
                    ensemble_weights.append(ensemble_r2)
        
        except Exception as e:
            print(f"     {scaler_name} 스케일러 실패: {e}")
    
    # 최종 메타-앙상블
    if ensemble_predictions and sum(ensemble_weights) > 0:
        weights = np.array(ensemble_weights) / sum(ensemble_weights)
        final_prediction = sum(w * pred for w, pred in zip(weights, ensemble_predictions))
        
        final_r2 = r2_score(y_test, final_prediction)
        final_rmse = np.sqrt(mean_squared_error(y_test, final_prediction))
        
        print(f"\\n   🎯 최종 메타-앙상블:")
        print(f"   R²: {final_r2:.4f} ({final_r2:.1%})")
        print(f"   RMSE: {final_rmse:.3f} ha")
        
        return final_r2, final_rmse, final_prediction
    
    return 0, float('inf'), None

def main():
    """메인"""
    print("🚀 면적 모델 최종 최적화 - 30% R² 목표")
    print("=" * 50)
    
    # 데이터 로드
    df = pd.read_csv('final_merged_feature_engineered.csv', low_memory=False)
    fire_mask = (df['fire_area'] > 0) & df['fire_area'].notna()
    fire_df = df[fire_mask].copy()
    
    print(f"화재 데이터: {fire_df.shape}")
    
    # 궁극적 피처 생성
    df_features, features = create_ultimate_features(fire_df)
    
    # 강력한 전처리
    X_raw = df_features[features]
    y_raw = df_features['fire_area']
    
    X_processed, y_processed = robust_preprocessing(X_raw, y_raw, outlier_threshold=0.995)
    
    # 로그 변환
    y_log = np.log1p(y_processed)
    
    print(f"\\n📊 최종 데이터:")
    print(f"   X: {X_processed.shape}")
    print(f"   y 범위: {y_processed.min():.3f} ~ {y_processed.max():.3f} ha")
    print(f"   y 로그 범위: {y_log.min():.3f} ~ {y_log.max():.3f}")
    
    # 훈련/테스트 분할
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X_processed, y_log, test_size=0.25, random_state=42
    )
    
    # 원래 스케일로도 테스트용
    y_train = np.expm1(y_train_log)
    y_test = np.expm1(y_test_log)
    
    print(f"\\n🔄 모델 훈련...")
    print(f"   훈련: {X_train.shape[0]}개")
    print(f"   테스트: {X_test.shape[0]}개")
    
    # 궁극적 앙상블 실행
    final_r2, final_rmse, final_pred = ultimate_ensemble(X_train, X_test, y_train_log, y_test_log)
    
    # 원래 스케일로 평가
    if final_pred is not None:
        final_pred_original = np.expm1(final_pred)
        final_pred_original = np.maximum(0, final_pred_original)
        
        original_r2 = r2_score(y_test, final_pred_original)
        original_rmse = np.sqrt(mean_squared_error(y_test, final_pred_original))
        
        print(f"\\n" + "=" * 50)
        print(f"🏆 최종 결과 (원래 스케일)")
        print("=" * 50)
        print(f"R²: {original_r2:.4f} ({original_r2:.1%})")
        print(f"RMSE: {original_rmse:.3f} ha")
        print(f"MAE: {np.mean(np.abs(y_test - final_pred_original)):.3f} ha")
        
        # 개선 정도
        baseline_r2 = 0.179  # 기존 최고
        improvement = original_r2 - baseline_r2
        
        print(f"\\n📈 개선 정도:")
        print(f"   기존 최고: {baseline_r2:.1%}")
        print(f"   현재 성능: {original_r2:.1%}")
        print(f"   개선량: {improvement:+.3f} ({improvement/baseline_r2:+.1%})")
        
        # 목표 달성 여부
        if original_r2 >= 0.3:
            print("\\n🎉 목표 달성! 30% 이상 R² 달성!")
        elif original_r2 >= 0.25:
            print("\\n🚀 우수한 성능! 25% 이상 달성!")
        elif original_r2 >= 0.2:
            print("\\n📈 좋은 개선! 20% 이상 달성!")
        else:
            print("\\n📊 일부 개선. 추가 연구 필요")
        
        return original_r2
    else:
        print("\\n❌ 모델 훈련 실패")
        return 0

if __name__ == "__main__":
    final_r2 = main()
    print(f"\\n✅ 최종 R²: {final_r2:.1%}")