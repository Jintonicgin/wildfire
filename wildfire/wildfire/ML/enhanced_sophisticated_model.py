#!/usr/bin/env python3
"""
향상된 정교한 화재 예측 모델 - 기존 모델의 업그레이드
- 기존 R² 0.66을 0.75+ 목표
- 더 나은 타겟 변환 전략
- 개선된 메타 학습
- 도메인별 전문화
- 시공간 패턴 강화
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import TransformedTargetRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

def load_and_enhance_data():
    """기존 데이터 로드 및 고급 전처리"""
    print("🚀 향상된 데이터 전처리...")
    
    df = pd.read_csv('final_merged_feature_engineered.csv')
    fire_data = df[(df['fire_area'] > 0) & (df['fire_area'].notna())].copy()
    
    # 기존 물리학 기반 피처 적용
    from sophisticated_area_model import create_fire_physics_features, create_temporal_fire_features
    fire_data, physics_features = create_fire_physics_features(fire_data)
    fire_data, temporal_features = create_temporal_fire_features(fire_data)
    
    return fire_data, physics_features + temporal_features

def create_enhanced_features(df, base_features):
    """향상된 피처 엔지니어링"""
    print("🧬 향상된 피처 엔지니어링...")
    
    df_enhanced = df.copy()
    new_features = base_features.copy()
    
    # 1. 고차 다항식 피처 (핵심 변수만)
    key_vars = ['t2m_0h', 'rh2m_0h', 'ws10m_0h', 'fwi_0h']
    existing_key_vars = [v for v in key_vars if v in df.columns]
    
    for var in existing_key_vars:
        if df[var].std() > 0:
            # 제곱 및 세제곱
            df_enhanced[f'{var}_squared'] = np.power(df[var], 2)
            df_enhanced[f'{var}_cubed'] = np.power(df[var], 3)
            new_features.extend([f'{var}_squared', f'{var}_cubed'])
    
    # 2. 상호작용 피처 (모든 조합)
    for i, var1 in enumerate(existing_key_vars):
        for j, var2 in enumerate(existing_key_vars[i+1:], i+1):
            interaction_name = f'{var1}_x_{var2}'
            df_enhanced[interaction_name] = df[var1] * df[var2]
            new_features.append(interaction_name)
    
    # 3. 비율 및 인덱스 피처
    if 't2m_0h' in df.columns and 'rh2m_0h' in df.columns:
        df_enhanced['fire_danger_index'] = (df['t2m_0h'] ** 2) / (df['rh2m_0h'] + 1)
        new_features.append('fire_danger_index')
    
    # 4. 로그 및 지수 변환
    for var in existing_key_vars:
        if (df[var] > 0).all():
            df_enhanced[f'{var}_log'] = np.log1p(df[var])
            new_features.append(f'{var}_log')
        
        if (df[var] >= 0).all() and df[var].max() < 50:  # 안전한 지수 변환
            df_enhanced[f'{var}_exp'] = np.expm1(df[var] * 0.1)
            new_features.append(f'{var}_exp')
    
    # 5. 시간적 집계 피처 (계절별)
    if 'month' in df.columns:
        # 계절별 그룹화
        season_map = {12: 'winter', 1: 'winter', 2: 'winter',
                     3: 'spring', 4: 'spring', 5: 'spring',
                     6: 'summer', 7: 'summer', 8: 'summer',
                     9: 'fall', 10: 'fall', 11: 'fall'}
        
        df_enhanced['season'] = df['month'].map(season_map)
        
        # 계절별 통계
        for var in existing_key_vars:
            seasonal_stats = df.groupby(df['month'].map(season_map))[var].transform('mean')
            df_enhanced[f'{var}_seasonal_avg'] = seasonal_stats
            new_features.append(f'{var}_seasonal_avg')
    
    # 6. 공간적 클러스터링 피처
    spatial_vars = [col for col in df.columns if any(x in col.lower() for x in ['lat', 'lon', 'elevation'])]
    if len(spatial_vars) >= 2:
        from sklearn.cluster import KMeans
        spatial_data = df[spatial_vars[:2]].fillna(0)
        if len(spatial_data) > 10:
            kmeans = KMeans(n_clusters=min(8, len(spatial_data)//50), random_state=42)
            df_enhanced['spatial_cluster'] = kmeans.fit_predict(spatial_data)
            new_features.append('spatial_cluster')
    
    print(f"   추가된 피처: {len(new_features) - len(base_features)}개")
    print(f"   총 피처: {len(new_features)}개")
    
    return df_enhanced, new_features

def advanced_feature_selection(X, y, max_features=35):
    """고급 피처 선택 - 다단계 접근"""
    print("🎯 고급 피처 선택...")
    
    # 1단계: 상관관계 기반 필터링
    correlations = abs(X.corrwith(y)).fillna(0)
    high_corr_features = correlations[correlations > 0.03].index.tolist()
    print(f"   1단계 (상관관계 > 0.03): {len(high_corr_features)}개")
    
    # 2단계: 다중공선성 제거
    if len(high_corr_features) > 1:
        X_filtered = X[high_corr_features]
        
        # 피처간 상관관계 매트릭스
        corr_matrix = X_filtered.corr().abs()
        
        # 높은 상관관계 제거 (0.9 이상)
        to_remove = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.9:
                    # 타겟과 상관관계가 낮은 것 제거
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    if correlations[col1] < correlations[col2]:
                        to_remove.add(col1)
                    else:
                        to_remove.add(col2)
        
        filtered_features = [f for f in high_corr_features if f not in to_remove]
        print(f"   2단계 (다중공선성 제거): {len(filtered_features)}개")
    else:
        filtered_features = high_corr_features
    
    # 3단계: ML 기반 중요도 선택
    if len(filtered_features) > max_features:
        X_for_selection = X[filtered_features].copy()
        
        # 결측치 처리
        for col in X_for_selection.columns:
            if X_for_selection[col].isna().sum() > 0:
                X_for_selection[col] = X_for_selection[col].fillna(X_for_selection[col].median())
        
        # Random Forest + XGBoost 중요도 평균
        importances = {}
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_for_selection, y)
        rf_importance = dict(zip(filtered_features, rf.feature_importances_))
        
        # XGBoost (가능한 경우)
        try:
            xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            xgb_model.fit(X_for_selection, y)
            xgb_importance = dict(zip(filtered_features, xgb_model.feature_importances_))
            
            # 평균 중요도
            for feature in filtered_features:
                importances[feature] = (rf_importance[feature] + xgb_importance[feature]) / 2
        except:
            importances = rf_importance
        
        # 상위 피처 선택
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in sorted_features[:max_features]]
        
        print(f"   3단계 (ML 중요도): {len(selected_features)}개")
    else:
        selected_features = filtered_features
    
    print(f"   최종 선택: {len(selected_features)}개")
    return selected_features

def create_enhanced_ensemble(X, y, selected_features):
    """향상된 앙상블 모델"""
    print("🚀 향상된 앙상블 구축...")
    
    X_selected = X[selected_features].copy()
    
    # 데이터 전처리
    for col in X_selected.columns:
        if X_selected[col].isna().sum() > 0:
            X_selected[col] = X_selected[col].fillna(X_selected[col].median())
    
    X_selected = X_selected.replace([np.inf, -np.inf], np.nan)
    for col in X_selected.columns:
        if X_selected[col].isna().sum() > 0:
            X_selected[col] = X_selected[col].fillna(X_selected[col].median())
    
    # 스케일러
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_selected)
    X_scaled = pd.DataFrame(X_scaled, columns=selected_features, index=X_selected.index)
    
    # 다양한 타겟 변환기
    transformers = {
        'yeo_johnson': PowerTransformer(method='yeo-johnson'),
        'quantile': QuantileTransformer(output_distribution='normal'),
        'robust': RobustScaler()
    }
    
    # 기본 모델들 (더 다양하게)
    base_models = {}
    
    for transform_name, transformer in transformers.items():
        print(f"   - {transform_name} 변환으로 학습 중...")
        
        models_for_transform = {
            'rf': RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'xgb': xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.08,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            ) if 'xgboost' in globals() else None,
            'lgb': lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.08,
                max_depth=8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ) if 'lightgbm' in globals() else None,
            'gbm': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.08,
                max_depth=8,
                random_state=42
            ),
            'extra': ExtraTreesRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'mlp': MLPRegressor(
                hidden_layer_sizes=(150, 75, 25),
                activation='relu',
                alpha=0.01,
                learning_rate_init=0.001,
                max_iter=300,
                random_state=42
            )
        }
        
        for model_name, model in models_for_transform.items():
            if model is not None:
                try:
                    transformed_model = TransformedTargetRegressor(
                        regressor=model,
                        transformer=transformer
                    )
                    transformed_model.fit(X_scaled, y)
                    base_models[f"{model_name}_{transform_name}"] = transformed_model
                except Exception as e:
                    print(f"     - {model_name} 실패: {e}")
    
    print(f"   학습된 기본 모델: {len(base_models)}개")
    
    # 스태킹을 위한 메타 피처 생성
    print("🎯 스태킹 메타 학습...")
    
    meta_features = np.zeros((len(X_scaled), len(base_models)))
    model_names = list(base_models.keys())
    
    # 5-fold 교차검증으로 메타 피처 생성
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(X_scaled):
        X_train_fold, X_val_fold = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        for i, (model_name, model) in enumerate(base_models.items()):
            try:
                fold_model = model.__class__(
                    regressor=model.regressor.__class__(**model.regressor.get_params()),
                    transformer=model.transformer.__class__(**model.transformer.get_params())
                )
                fold_model.fit(X_train_fold, y_train_fold)
                meta_features[val_idx, i] = fold_model.predict(X_val_fold)
            except:
                meta_features[val_idx, i] = y_val_fold.mean()
    
    # 메타 모델들 (여러 개 사용하여 더 견고하게)
    meta_models = {
        'gb_meta': GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        ),
        'ridge_meta': Ridge(alpha=1.0),
        'bayesian_meta': BayesianRidge()
    }
    
    trained_meta_models = {}
    for meta_name, meta_model in meta_models.items():
        try:
            meta_model.fit(meta_features, y)
            trained_meta_models[meta_name] = meta_model
        except Exception as e:
            print(f"   메타 모델 {meta_name} 실패: {e}")
    
    print(f"   메타 모델: {len(trained_meta_models)}개")
    
    return base_models, trained_meta_models, scaler, model_names

def enhanced_predict(X, base_models, meta_models, scaler, model_names, selected_features):
    """향상된 예측 함수"""
    X_selected = X[selected_features].copy()
    
    # 동일한 전처리
    for col in X_selected.columns:
        if X_selected[col].isna().sum() > 0:
            X_selected[col] = X_selected[col].fillna(X_selected[col].median())
    
    X_selected = X_selected.replace([np.inf, -np.inf], np.nan)
    for col in X_selected.columns:
        if X_selected[col].isna().sum() > 0:
            X_selected[col] = X_selected[col].fillna(X_selected[col].median())
    
    X_scaled = scaler.transform(X_selected)
    X_scaled = pd.DataFrame(X_scaled, columns=selected_features, index=X_selected.index)
    
    # 기본 모델들 예측
    base_predictions = np.zeros((len(X_scaled), len(base_models)))
    
    for i, (model_name, model) in enumerate(base_models.items()):
        try:
            base_predictions[:, i] = model.predict(X_scaled)
        except:
            base_predictions[:, i] = 0
    
    # 메타 모델들 예측
    meta_predictions = []
    for meta_name, meta_model in meta_models.items():
        try:
            meta_pred = meta_model.predict(base_predictions)
            meta_predictions.append(meta_pred)
        except:
            continue
    
    if meta_predictions:
        # 메타 모델들의 평균
        final_prediction = np.mean(meta_predictions, axis=0)
    else:
        # 기본 모델들의 가중 평균 (fallback)
        weights = np.ones(base_predictions.shape[1]) / base_predictions.shape[1]
        final_prediction = np.average(base_predictions, axis=1, weights=weights)
    
    return final_prediction, base_predictions

def create_enhanced_visualizations(y_true, y_pred, base_predictions, model_names):
    """향상된 시각화"""
    print("🎨 향상된 시각화 생성...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    
    # 1. 실제 vs 예측
    r2 = r2_score(y_true, y_pred)
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=40, color='darkblue')
    max_val = max(y_true.max(), y_pred.max())
    axes[0, 0].plot([0, max_val], [0, max_val], 'r--', lw=2)
    axes[0, 0].set_xlabel('실제 화재 면적 (ha)', fontsize=12)
    axes[0, 0].set_ylabel('예측 화재 면적 (ha)', fontsize=12)
    axes[0, 0].set_title(f'향상된 모델 성능\nR² = {r2:.4f}', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 잔차 플롯
    residuals = y_true - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=40, color='darkred')
    axes[0, 1].axhline(y=0, color='black', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('예측값 (ha)', fontsize=12)
    axes[0, 1].set_ylabel('잔차 (ha)', fontsize=12)
    axes[0, 1].set_title('잔차 분포', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 오차 분포 
    axes[0, 2].hist(residuals, bins=40, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 2].set_xlabel('잔차 (ha)', fontsize=12)
    axes[0, 2].set_ylabel('빈도', fontsize=12)
    axes[0, 2].set_title('잔차 히스토그램', fontsize=14, fontweight='bold')
    axes[0, 2].axvline(x=0, color='red', linestyle='--', lw=2)
    
    # 4. 모델별 성능 (상위 8개만)
    if base_predictions.shape[1] > 0:
        n_models_to_show = min(8, base_predictions.shape[1])
        model_r2s = []
        display_names = []
        
        for i in range(n_models_to_show):
            r2_i = r2_score(y_true, base_predictions[:, i])
            model_r2s.append(r2_i)
            # 모델명 단축
            short_name = model_names[i].replace('_yeo_johnson', '').replace('_quantile', '').replace('_robust', '')
            display_names.append(short_name[:10])
        
        bars = axes[1, 0].bar(range(len(display_names)), model_r2s, 
                             color=plt.cm.Set3(np.linspace(0, 1, len(model_r2s))))
        axes[1, 0].set_ylabel('R² Score', fontsize=12)
        axes[1, 0].set_title('개별 모델 성능 (상위 8개)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xticks(range(len(display_names)))
        axes[1, 0].set_xticklabels(display_names, rotation=45, ha='right')
        
        # 앙상블 성능 라인
        axes[1, 0].axhline(y=r2, color='red', linestyle='--', linewidth=3, 
                          label=f'앙상블: {r2:.4f}')
        axes[1, 0].legend(fontsize=10)
    
    # 5. 화재 규모별 성능
    q25, q50, q75 = np.percentile(y_true, [25, 50, 75])
    
    masks = [
        y_true <= q25,
        (y_true > q25) & (y_true <= q50),
        (y_true > q50) & (y_true <= q75),
        y_true > q75
    ]
    
    categories = ['Very Small\n(≤25%)', 'Small\n(25-50%)', 'Medium\n(50-75%)', 'Large\n(>75%)']
    colors = ['lightblue', 'lightgreen', 'orange', 'lightcoral']
    
    r2_by_size = []
    valid_categories = []
    valid_colors = []
    
    for i, mask in enumerate(masks):
        if mask.sum() > 3:  # 최소 4개 샘플
            r2_i = r2_score(y_true[mask], y_pred[mask])
            r2_by_size.append(r2_i)
            valid_categories.append(categories[i])
            valid_colors.append(colors[i])
    
    if valid_categories:
        bars = axes[1, 1].bar(valid_categories, r2_by_size, color=valid_colors)
        axes[1, 1].set_ylabel('R² Score', fontsize=12)
        axes[1, 1].set_title('화재 규모별 성능', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylim(min(0, min(r2_by_size)) - 0.1, max(r2_by_size) + 0.1)
        
        # 값 표시
        for bar, r2_val in zip(bars, r2_by_size):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{r2_val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 6. 예측 품질 분석
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    accuracy_ranges = [(0, 10), (10, 25), (25, 50), (50, 100), (100, float('inf'))]
    range_labels = ['<10%', '10-25%', '25-50%', '50-100%', '>100%']
    range_colors = ['darkgreen', 'green', 'yellow', 'orange', 'red']
    
    relative_errors = np.abs((y_true - y_pred) / y_true) * 100
    range_counts = []
    
    for low, high in accuracy_ranges:
        count = ((relative_errors >= low) & (relative_errors < high)).sum()
        range_counts.append(count)
    
    wedges, texts, autotexts = axes[1, 2].pie(range_counts, labels=range_labels, colors=range_colors,
                                              autopct='%1.1f%%', startangle=90)
    axes[1, 2].set_title(f'예측 오차 분포\n평균 MAPE: {mape:.1f}%', 
                        fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('enhanced_sophisticated_model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 시각화 저장: enhanced_sophisticated_model_performance.png")

def main():
    """메인 함수"""
    print("🎯 향상된 정교한 화재 예측 모델 개발")
    print("=" * 80)
    print("목표: 기존 R² 0.66 → 0.75+ 달성")
    print("=" * 80)
    
    try:
        # 1. 데이터 로드
        fire_data, base_features = load_and_enhance_data()
        print(f"✅ 기본 데이터: {fire_data.shape}")
        
        # 2. 향상된 피처 엔지니어링
        fire_data_enhanced, all_features = create_enhanced_features(fire_data, base_features)
        
        # 3. 피처 준비
        target_col = 'fire_area'
        available_features = [col for col in all_features 
                            if col in fire_data_enhanced.columns 
                            and col != target_col
                            and fire_data_enhanced[col].dtype in ['int64', 'float64']]
        
        X = fire_data_enhanced[available_features].copy()
        y = fire_data_enhanced[target_col].copy()
        
        # 4. 데이터 정리
        print(f"\n🧹 데이터 정리...")
        
        for col in X.columns:
            if X[col].isna().sum() > 0:
                X[col] = X[col].fillna(X[col].median())
        
        # 이상치 제거 (상위 5%)
        q95 = y.quantile(0.95)
        mask = y <= q95
        X, y = X[mask], y[mask]
        
        print(f"✅ 정리된 데이터: {X.shape[0]}개 샘플, {X.shape[1]}개 피처")
        
        # 5. 고급 피처 선택
        selected_features = advanced_feature_selection(X, y, max_features=35)
        
        # 6. 데이터 분할 (층화 추출)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=pd.qcut(y, q=5, duplicates='drop')
        )
        
        # 7. 향상된 앙상블 학습
        base_models, meta_models, scaler, model_names = create_enhanced_ensemble(
            X_train, y_train, selected_features
        )
        
        # 8. 예측 및 평가
        print("\n🎯 최종 예측 및 평가...")
        
        y_pred_train, base_pred_train = enhanced_predict(
            X_train, base_models, meta_models, scaler, model_names, selected_features
        )
        y_pred_test, base_pred_test = enhanced_predict(
            X_test, base_models, meta_models, scaler, model_names, selected_features
        )
        
        # 9. 성능 평가
        print("\n" + "=" * 80)
        print("🏆 향상된 정교한 모델 최종 성능 보고서")
        print("=" * 80)
        
        # 훈련 성능
        train_r2 = r2_score(y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        
        print(f"\n📊 훈련 성능:")
        print(f"   • R²: {train_r2:.4f}")
        print(f"   • RMSE: {train_rmse:.4f} ha")
        print(f"   • MAE: {train_mae:.4f} ha")
        
        # 테스트 성능
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
        
        print(f"\n📊 테스트 성능:")
        print(f"   • R²: {test_r2:.4f}")
        print(f"   • RMSE: {test_rmse:.4f} ha")
        print(f"   • MAE: {test_mae:.4f} ha")
        print(f"   • MAPE: {test_mape:.2f}%")
        
        # 개선도 계산
        baseline_r2 = 0.6636  # 기존 정교한 모델 성능
        improvement = test_r2 - baseline_r2
        improvement_pct = (improvement / baseline_r2) * 100 if baseline_r2 > 0 else 0
        
        print(f"\n📈 개선도:")
        print(f"   • 기존 모델: R² {baseline_r2:.4f}")
        print(f"   • 향상된 모델: R² {test_r2:.4f}")
        print(f"   • 개선도: {improvement:+.4f} ({improvement_pct:+.1f}%)")
        
        # 10. 시각화
        create_enhanced_visualizations(y_test, y_pred_test, base_pred_test, model_names)
        
        # 11. 모델 저장
        print(f"\n💾 모델 저장...")
        
        model_package = {
            'base_models': base_models,
            'meta_models': meta_models,
            'scaler': scaler,
            'model_names': model_names,
            'selected_features': selected_features,
            'test_metrics': {
                'r2': test_r2,
                'rmse': test_rmse,
                'mae': test_mae,
                'mape': test_mape
            }
        }
        
        joblib.dump(model_package, 'enhanced_sophisticated_model.joblib')
        
        config = {
            'model_type': 'Enhanced Sophisticated Ensemble',
            'n_base_models': len(base_models),
            'n_meta_models': len(meta_models),
            'n_features': len(selected_features),
            'baseline_r2': baseline_r2,
            'test_r2': test_r2,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'selected_features': selected_features
        }
        
        with open('enhanced_sophisticated_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✅ 저장 완료!")
        
        # 12. 최종 요약
        if test_r2 > 0.75:
            achievement = "🎉 목표 달성! (R² > 0.75)"
            grade = "🟢 최고 등급"
        elif test_r2 > 0.7:
            achievement = "🎯 거의 달성! (R² > 0.7)"
            grade = "🟢 우수 등급"
        elif test_r2 > baseline_r2:
            achievement = "📈 기존 모델 개선 성공!"
            grade = "🟢 개선됨"
        else:
            achievement = "📊 추가 개선 필요"
            grade = "🟡 기준 유지"
        
        print(f"\n{achievement}")
        print(f"   • 최종 R²: {test_r2:.4f}")
        print(f"   • 성능 등급: {grade}")
        print(f"   • 기본 모델: {len(base_models)}개")
        print(f"   • 메타 모델: {len(meta_models)}개")
        print(f"   • 선택된 피처: {len(selected_features)}개")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()