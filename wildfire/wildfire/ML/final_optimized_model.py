#!/usr/bin/env python3
"""
최종 최적화 화재 예측 모델
- 집중적 하이퍼파라미터 튜닝
- 더 정교한 피처 선택
- 앙상블 최적화
- 현실적 성능 목표
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import randint, uniform
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """데이터 로드 및 준비"""
    print("📁 최종 데이터 준비 중...")
    
    df = pd.read_csv('final_merged_feature_engineered.csv')
    fire_data = df[(df['fire_area'] > 0) & (df['fire_area'].notna())].copy()
    
    # 가장 중요한 피처들만 엄선 (분석 결과 기반)
    critical_features = [
        # 시간 정보 (필수)
        'startmonth', 'startday', 'startyear',
        
        # 현재 기상 (필수)
        't2m_0h', 'rh2m_0h', 'ws10m_0h', 'ws2m_0h', 'ps_0h',
        
        # 단기 기상 (3시간 이내)
        't2m_3h', 'rh2m_3h', 'ws10m_3h', 'ws2m_3h', 'ps_3h',
        
        # FWI 시스템 (화재 위험 지수)
        'fwi_0h', 'ffmc_0h', 'dmc_0h', 'dc_0h', 'isi_0h', 'bui_0h',
        
        # 지형 (고정값)
        'elevation_mean', 'slope_mean', 'aspect_mode',
        
        # 식생 (화재 전)
        'ndvi_before', 'treecover_pre_fire_5x5',
        
        # 건조도 (과거 기록)
        'dry_days_7d_start', 'dry_days_14d_start', 'dry_days_30d_start'
    ]
    
    # 실제 존재하는 피처만 선택
    available_features = [f for f in critical_features if f in fire_data.columns]
    
    print(f"✅ 핵심 피처: {len(available_features)}개")
    return fire_data, available_features

def create_optimized_features(df, base_features):
    """최적화된 피처 생성"""
    print("🔬 최적화 피처 생성...")
    
    df_opt = df.copy()
    new_features = []
    
    # 1. 가장 중요한 비선형 조합만
    if 't2m_0h' in base_features and 'rh2m_0h' in base_features:
        # 증기압 부족 (Vapor Pressure Deficit) - 화재에 매우 중요
        df_opt['vpd'] = df_opt['t2m_0h'] * (1 - df_opt['rh2m_0h'] / 100)
        new_features.append('vpd')
    
    # 2. 풍속 위험 지수
    if 'ws10m_0h' in base_features:
        # 강풍 위험 플래그
        wind_threshold = df_opt['ws10m_0h'].quantile(0.75)  # 상위 25%
        df_opt['high_wind_risk'] = (df_opt['ws10m_0h'] > wind_threshold).astype(int)
        new_features.append('high_wind_risk')
    
    # 3. 계절-날씨 상호작용
    if 'startmonth' in base_features and 't2m_0h' in base_features:
        # 여름 고온 위험
        df_opt['summer_heat_risk'] = ((df_opt['startmonth'].isin([6, 7, 8])) & 
                                     (df_opt['t2m_0h'] > df_opt['t2m_0h'].quantile(0.7))).astype(int)
        new_features.append('summer_heat_risk')
        
        # 봄가을 건조 위험  
        df_opt['spring_autumn_dry'] = ((df_opt['startmonth'].isin([3, 4, 5, 9, 10, 11])) & 
                                      (df_opt['rh2m_0h'] < df_opt['rh2m_0h'].quantile(0.3))).astype(int)
        new_features.append('spring_autumn_dry')
    
    # 4. FWI 강화 지수
    fwi_cols = [col for col in base_features if col in ['fwi_0h', 'ffmc_0h', 'dmc_0h', 'dc_0h', 'isi_0h']]
    if len(fwi_cols) >= 3:
        # FWI 위험 임계값 초과 개수
        df_opt['fwi_risk_count'] = 0
        for col in fwi_cols:
            if col in df_opt.columns:
                threshold = df_opt[col].quantile(0.6)  # 상위 40%
                df_opt['fwi_risk_count'] += (df_opt[col] > threshold).astype(int)
        new_features.append('fwi_risk_count')
    
    # 5. 극한 건조 위험
    if 'dry_days_7d_start' in base_features:
        df_opt['extreme_dry_risk'] = (df_opt['dry_days_7d_start'] >= 5).astype(int)
        new_features.append('extreme_dry_risk')
    
    print(f"✅ 최적화 피처: {len(new_features)}개")
    return df_opt, new_features

def advanced_feature_selection(X, y, features, top_k=25):
    """고급 피처 선택"""
    print(f"🔍 고급 피처 선택 (상위 {top_k}개)...")
    
    # 1. 상호정보량 기반 선택
    mi_selector = SelectKBest(mutual_info_regression, k=top_k)
    X_mi = mi_selector.fit_transform(X, y)
    mi_features = np.array(features)[mi_selector.get_support()]
    
    # 2. F-score 기반 선택  
    f_selector = SelectKBest(f_regression, k=top_k)
    X_f = f_selector.fit_transform(X, y)
    f_features = np.array(features)[f_selector.get_support()]
    
    # 3. RandomForest 중요도 기반
    rf_selector = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_selector.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_selector.feature_importances_
    }).sort_values('importance', ascending=False)
    
    rf_features = feature_importance.head(top_k)['feature'].values
    
    # 4. 교집합 - 모든 방법에서 선택된 피처
    mi_set = set(mi_features)
    f_set = set(f_features)
    rf_set = set(rf_features)
    
    # 최소 2개 방법에서 선택된 피처
    consensus_features = []
    for feature in features:
        vote_count = sum([feature in mi_set, feature in f_set, feature in rf_set])
        if vote_count >= 2:
            consensus_features.append(feature)
    
    # 부족하면 RF 중요도 순으로 추가
    if len(consensus_features) < top_k:
        for feature in rf_features:
            if feature not in consensus_features:
                consensus_features.append(feature)
                if len(consensus_features) >= top_k:
                    break
    
    final_features = consensus_features[:top_k]
    
    print(f"✅ 최종 선택: {len(final_features)}개 피처")
    print("상위 10개 피처:")
    for i, feature in enumerate(final_features[:10], 1):
        importance = feature_importance[feature_importance['feature'] == feature]['importance']
        imp_val = importance.iloc[0] if len(importance) > 0 else 0
        print(f"   {i:2d}. {feature:<20} ({imp_val:.4f})")
    
    return final_features

def optimize_hyperparameters(X, y):
    """집중적 하이퍼파라미터 최적화"""
    print("⚙️ 하이퍼파라미터 최적화 중...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models_to_optimize = {}
    
    # 1. XGBoost 최적화
    print("🚀 XGBoost 최적화...")
    xgb_param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0, 0.1, 0.5, 1.0]
    }
    
    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    xgb_search = RandomizedSearchCV(
        xgb_model, xgb_param_dist, n_iter=50, cv=3, 
        scoring='r2', random_state=42, n_jobs=-1
    )
    xgb_search.fit(X_train, y_train)
    models_to_optimize['xgboost'] = xgb_search.best_estimator_
    
    # 2. LightGBM 최적화
    print("💡 LightGBM 최적화...")
    lgb_param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0, 0.1, 0.5, 1.0],
        'num_leaves': [15, 31, 63, 127]
    }
    
    lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
    lgb_search = RandomizedSearchCV(
        lgb_model, lgb_param_dist, n_iter=50, cv=3,
        scoring='r2', random_state=42, n_jobs=-1  
    )
    lgb_search.fit(X_train, y_train)
    models_to_optimize['lightgbm'] = lgb_search.best_estimator_
    
    # 3. RandomForest 최적화
    print("🌲 RandomForest 최적화...")
    rf_param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [5, 8, 10, 12, 15, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 0.5, 0.7, 1.0],
        'bootstrap': [True, False]
    }
    
    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_search = RandomizedSearchCV(
        rf_model, rf_param_dist, n_iter=50, cv=3,
        scoring='r2', random_state=42, n_jobs=-1
    )
    rf_search.fit(X_train, y_train)
    models_to_optimize['randomforest'] = rf_search.best_estimator_
    
    # 4. ExtraTrees 추가
    print("🌳 ExtraTrees 최적화...")
    et_param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [5, 8, 10, 12, 15, None],
        'min_samples_split': [2, 5, 10, 15],  
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 0.5, 0.7, 1.0],
        'bootstrap': [False, True]
    }
    
    et_model = ExtraTreesRegressor(random_state=42, n_jobs=-1)
    et_search = RandomizedSearchCV(
        et_model, et_param_dist, n_iter=50, cv=3,
        scoring='r2', random_state=42, n_jobs=-1
    )
    et_search.fit(X_train, y_train)
    models_to_optimize['extratrees'] = et_search.best_estimator_
    
    # 성능 비교
    results = {}
    for name, model in models_to_optimize.items():
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results[name] = {'r2': r2, 'rmse': rmse, 'model': model}
        print(f"   {name:12s}: R² = {r2:6.4f}, RMSE = {rmse:6.4f}")
    
    return results

def create_optimal_ensemble(results, X, y):
    """최적 앙상블 생성"""
    print("🤝 최적 앙상블 생성 중...")
    
    # 성능 기반 가중치 계산
    r2_scores = {name: res['r2'] for name, res in results.items()}
    
    # 음수 R²는 0으로 처리
    positive_r2 = {name: max(0, r2) for name, r2 in r2_scores.items()}
    
    if sum(positive_r2.values()) > 0:
        # 정규화된 가중치
        total_r2 = sum(positive_r2.values())
        weights = {name: r2/total_r2 for name, r2 in positive_r2.items()}
    else:
        # 모든 모델이 음수면 동일 가중치
        weights = {name: 1/len(results) for name in results.keys()}
    
    print("앙상블 가중치:")
    for name, weight in weights.items():
        print(f"   {name}: {weight:.3f}")
    
    # 앙상블 예측 함수
    def ensemble_predict(X_pred):
        predictions = []
        for name, res in results.items():
            pred = res['model'].predict(X_pred) * weights[name]
            predictions.append(pred)
        return np.sum(predictions, axis=0)
    
    # 앙상블 성능 평가
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    ensemble_pred = ensemble_predict(X_test)
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    
    print(f"🏆 앙상블 성능: R² = {ensemble_r2:.4f}, RMSE = {ensemble_rmse:.4f}")
    
    return ensemble_predict, weights

def main():
    """메인 실행"""
    print("🚀 최종 최적화 화재 예측 모델 시작...")
    
    # 1. 데이터 준비
    fire_data, base_features = load_and_prepare_data()
    
    # 2. 최적화 피처 생성
    enhanced_data, new_features = create_optimized_features(fire_data, base_features)
    all_features = base_features + new_features
    
    # 3. 데이터 정제
    X = enhanced_data[all_features].copy()
    y = enhanced_data['fire_area'].copy()
    
    # 무한값/결측치 처리
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if X[col].isna().sum() > 0:
            X[col] = X[col].fillna(X[col].median())
    
    # 이상치 제거 (85%ile로 더 완화)
    area_85th = y.quantile(0.85)
    mask = y <= area_85th
    X = X[mask]
    y = y[mask]
    
    # 로그 변환
    y_log = np.log1p(y)
    
    print(f"📊 최종 데이터: {X.shape[0]}개 샘플, {X.shape[1]}개 피처")
    
    # 4. 피처 선택
    selected_features = advanced_feature_selection(X, y_log, all_features, top_k=20)
    X_selected = X[selected_features]
    
    # 5. 스케일링
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # 6. 하이퍼파라미터 최적화
    optimization_results = optimize_hyperparameters(X_scaled, y_log)
    
    # 7. 최적 앙상블 생성
    ensemble_fn, weights = create_optimal_ensemble(optimization_results, X_scaled, y_log)
    
    # 8. 최종 교차검증
    print("\n📊 최종 교차검증...")
    best_model_name = max(optimization_results.keys(), key=lambda k: optimization_results[k]['r2'])
    best_model = optimization_results[best_model_name]['model']
    
    cv_scores = cross_val_score(best_model, X_scaled, y_log, cv=5, scoring='r2')
    print(f"🏆 최고 성능 모델 ({best_model_name}): {optimization_results[best_model_name]['r2']:.4f}")
    print(f"📊 교차검증 R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # 9. 저장
    joblib.dump(best_model, 'final_optimized_model.joblib')
    joblib.dump(scaler, 'final_scaler.joblib')
    
    with open('final_features.json', 'w') as f:
        json.dump(selected_features, f, indent=2)
    
    with open('ensemble_weights.json', 'w') as f:
        json.dump(weights, f, indent=2)
    
    print("\n🎉 최종 최적화 완료!")
    print(f"📊 단일 모델 최고 성능: {optimization_results[best_model_name]['r2']:.4f}")
    print(f"🤝 앙상블 성능: 별도 평가 필요")
    print(f"🔧 최종 피처 수: {len(selected_features)}개")
    
    print("\n💡 성능 해석:")
    if cv_scores.mean() > 0.3:
        print("   🟢 우수한 성능 (R² > 0.3)")
    elif cv_scores.mean() > 0.15:
        print("   🟡 보통 성능 (0.15 < R² < 0.3)")
    else:
        print("   🔴 낮은 성능 (R² < 0.15)")
        print("   → 화재 면적 예측은 본질적으로 매우 어려운 문제입니다")
        print("   → 현재 결과도 무작위보다는 나은 예측력을 제공합니다")

if __name__ == "__main__":
    main()