#!/usr/bin/env python3
"""
점진적 개선 모델 - 기존 Sophisticated Model 기반
목표: R² 0.66 → 0.72-0.75 (안정적 개선)

개선 전략:
1. 타겟 변환 최적화
2. 하이퍼파라미터 세밀 조정
3. 피처 상호작용 추가 (보수적)
4. 앙상블 가중치 최적화
5. 검증 방법 강화
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

def analyze_current_model_limitations():
    """현재 모델의 한계점 분석"""
    print("🔍 현재 모델 한계점 분석...")
    
    # 기존 모델 로드 및 분석
    try:
        base_models = joblib.load('sophisticated_base_models.joblib')
        with open('sophisticated_config.json', 'r') as f:
            config = json.load(f)
        
        print(f"✅ 현재 모델 현황:")
        print(f"   - 성능: R² {config['final_score']:.4f}")
        print(f"   - 기본 모델: {len(base_models)}개")
        print(f"   - 피처 수: {config['feature_count']}개")
        print(f"   - 타겟 변환: {config['best_transform']}")
        
        print(f"\n🎯 개선 포인트:")
        print(f"   1. 타겟 변환: Yeo-Johnson이 최적인지 재검토")
        print(f"   2. 하이퍼파라미터: 기본값 사용 → 세밀 조정")
        print(f"   3. 피처 상호작용: 핵심 변수간 추가 상호작용")
        print(f"   4. 앙상블 가중치: 균등 → 성능 기반 가중치")
        print(f"   5. 검증 방식: 일반 분할 → 시간적/지역적 분할")
        
        return config
        
    except FileNotFoundError:
        print("❌ 기존 모델 파일을 찾을 수 없습니다.")
        return None

def load_and_prepare_data():
    """데이터 로드 및 기본 준비"""
    print("📊 데이터 로드...")
    
    df = pd.read_csv('final_merged_feature_engineered.csv')
    fire_data = df[(df['fire_area'] > 0) & (df['fire_area'].notna())].copy()
    
    # 기존 물리학 피처 적용
    from sophisticated_area_model import create_fire_physics_features, create_temporal_fire_features
    fire_data, physics_features = create_fire_physics_features(fire_data)
    fire_data, temporal_features = create_temporal_fire_features(fire_data)
    
    return fire_data, physics_features + temporal_features

def optimize_target_transformation(y):
    """타겟 변환 최적화"""
    print("🎯 타겟 변환 최적화...")
    
    transformations = {
        'yeo_johnson': PowerTransformer(method='yeo-johnson'),
        'box_cox': PowerTransformer(method='box-cox'),
        'quantile_uniform': QuantileTransformer(output_distribution='uniform'),
        'quantile_normal': QuantileTransformer(output_distribution='normal'),
        'log1p': None  # 별도 처리
    }
    
    best_transform = None
    best_score = -np.inf
    best_name = None
    
    for name, transformer in transformations.items():
        try:
            if name == 'log1p':
                y_transformed = np.log1p(y)
            elif name == 'box_cox' and (y <= 0).any():
                continue  # Box-Cox는 양수만 가능
            else:
                y_transformed = transformer.fit_transform(y.values.reshape(-1, 1)).ravel()
            
            # 정규성 검정 (Shapiro-Wilk는 샘플 크기가 클 때 부정확하므로 Anderson-Darling 사용)
            if len(y_transformed) > 5000:
                # 서브샘플링
                sample_idx = np.random.choice(len(y_transformed), 5000, replace=False)
                y_sample = y_transformed[sample_idx]
            else:
                y_sample = y_transformed
            
            # Anderson-Darling 정규성 검정
            ad_stat, critical_values, significance_level = stats.anderson(y_sample, dist='norm')
            # 낮은 통계값이 더 정규분포에 가까움
            normality_score = -ad_stat
            
            # 추가 지표: 왜도와 첨도
            skewness = abs(stats.skew(y_transformed))
            kurtosis = abs(stats.kurtosis(y_transformed))
            
            # 종합 점수 (정규성 + 왜도 + 첨도)
            composite_score = normality_score - skewness - kurtosis/4
            
            print(f"   {name:20s}: 점수 {composite_score:8.4f} (왜도: {skewness:.3f})")
            
            if composite_score > best_score:
                best_score = composite_score
                best_transform = transformer
                best_name = name
                
        except Exception as e:
            print(f"   {name:20s}: 실패 ({e})")
    
    print(f"✅ 최적 변환: {best_name} (점수: {best_score:.4f})")
    return best_transform, best_name

def create_enhanced_interactions(df, base_features):
    """보수적인 피처 상호작용 추가"""
    print("🔗 핵심 상호작용 피처 생성...")
    
    df_enhanced = df.copy()
    new_features = base_features.copy()
    
    # 검증된 화재 예측 상호작용만 추가
    key_interactions = [
        # 온도-습도-바람 (화재 삼각형)
        ('t2m_0h', 'rh2m_0h', 'temp_rh_interaction'),
        ('ws10m_0h', 'rh2m_0h', 'wind_rh_interaction'),  
        ('t2m_0h', 'ws10m_0h', 'temp_wind_interaction'),
        
        # FWI 관련 상호작용
        ('fwi_0h', 't2m_0h', 'fwi_temp_interaction'),
        ('fwi_0h', 'ws10m_0h', 'fwi_wind_interaction'),
        
        # 지형-기후 상호작용
        ('elevation_mean', 't2m_0h', 'elevation_temp_interaction'),
        ('slope_mean', 'ws10m_0h', 'slope_wind_interaction'),
    ]
    
    added_count = 0
    for var1, var2, interaction_name in key_interactions:
        if var1 in df.columns and var2 in df.columns:
            # 정규화 후 상호작용
            v1_norm = (df[var1] - df[var1].mean()) / (df[var1].std() + 1e-8)
            v2_norm = (df[var2] - df[var2].mean()) / (df[var2].std() + 1e-8)
            
            df_enhanced[interaction_name] = v1_norm * v2_norm
            new_features.append(interaction_name)
            added_count += 1
    
    # 비선형 변환 (핵심 변수만)
    nonlinear_vars = ['t2m_0h', 'rh2m_0h', 'ws10m_0h', 'fwi_0h']
    for var in nonlinear_vars:
        if var in df.columns:
            # 제곱근 변환 (강건함)
            df_enhanced[f'{var}_sqrt'] = np.sqrt(np.abs(df[var]))
            new_features.append(f'{var}_sqrt')
            added_count += 1
    
    print(f"   추가된 상호작용 피처: {added_count}개")
    return df_enhanced, new_features

def optimize_hyperparameters(X, y, model_type='rf'):
    """모델별 하이퍼파라미터 최적화"""
    print(f"⚙️ {model_type.upper()} 하이퍼파라미터 최적화...")
    
    if model_type == 'rf':
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [10, 15, 20],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [2, 4, 6],
            'max_features': ['sqrt', 'log2']
        }
    elif model_type == 'xgb':
        model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [200, 300],
            'learning_rate': [0.05, 0.08, 0.1],
            'max_depth': [6, 8, 10],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
    elif model_type == 'lgb':
        model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
        param_grid = {
            'n_estimators': [200, 300],
            'learning_rate': [0.05, 0.08, 0.1],
            'max_depth': [6, 8, 10],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
    else:
        return model  # 최적화 안함
    
    # 작은 그리드로 빠른 검색
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0
    )
    
    grid_search.fit(X, y)
    print(f"   최적 파라미터: {grid_search.best_params_}")
    print(f"   최적 점수: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def create_improved_ensemble(X_train, y_train, selected_features, target_transformer):
    """개선된 앙상블 생성"""
    print("🚀 개선된 앙상블 구축...")
    
    X_selected = X_train[selected_features].copy()
    
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
    
    # 최적화된 기본 모델들
    base_models = {}
    
    print("   개별 모델 최적화...")
    
    # 1. Random Forest (최적화)
    rf_optimized = optimize_hyperparameters(X_scaled, y_train, 'rf')
    rf_transformed = TransformedTargetRegressor(
        regressor=rf_optimized,
        transformer=target_transformer
    )
    rf_transformed.fit(X_scaled, y_train)
    base_models['rf_optimized'] = rf_transformed
    
    # 2. XGBoost (최적화)
    try:
        xgb_optimized = optimize_hyperparameters(X_scaled, y_train, 'xgb')
        xgb_transformed = TransformedTargetRegressor(
            regressor=xgb_optimized,
            transformer=target_transformer
        )
        xgb_transformed.fit(X_scaled, y_train)
        base_models['xgb_optimized'] = xgb_transformed
    except:
        print("   XGBoost 최적화 실패 - 기본값 사용")
    
    # 3. LightGBM (최적화)
    try:
        lgb_optimized = optimize_hyperparameters(X_scaled, y_train, 'lgb')
        lgb_transformed = TransformedTargetRegressor(
            regressor=lgb_optimized,
            transformer=target_transformer
        )
        lgb_transformed.fit(X_scaled, y_train)
        base_models['lgb_optimized'] = lgb_transformed
    except:
        print("   LightGBM 최적화 실패 - 스킵")
    
    # 4. Gradient Boosting (수동 최적화)
    gb_model = GradientBoostingRegressor(
        n_estimators=250,
        learning_rate=0.06,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        subsample=0.85,
        random_state=42
    )
    gb_transformed = TransformedTargetRegressor(
        regressor=gb_model,
        transformer=target_transformer
    )
    gb_transformed.fit(X_scaled, y_train)
    base_models['gb_tuned'] = gb_transformed
    
    # 5. Ridge (안정성용)
    ridge_model = Ridge(alpha=2.0)
    ridge_transformed = TransformedTargetRegressor(
        regressor=ridge_model,
        transformer=target_transformer
    )
    ridge_transformed.fit(X_scaled, y_train)
    base_models['ridge_stable'] = ridge_transformed
    
    print(f"✅ 기본 모델 {len(base_models)}개 학습 완료")
    
    # 개별 모델 성능 평가 (교차검증)
    print("📊 개별 모델 성능 평가...")
    model_scores = {}
    for name, model in base_models.items():
        try:
            scores = cross_val_score(model, X_scaled, y_train, cv=3, scoring='r2')
            model_scores[name] = scores.mean()
            print(f"   {name:15s}: R² = {scores.mean():.4f} (±{scores.std():.3f})")
        except:
            model_scores[name] = 0.0
    
    # 성능 기반 가중치 계산
    weights = optimize_ensemble_weights(X_scaled, y_train, base_models)
    
    return base_models, scaler, weights

def optimize_ensemble_weights(X, y, models):
    """앙상블 가중치 최적화"""
    print("⚖️ 앙상블 가중치 최적화...")
    
    # 교차검증으로 각 모델의 예측값 생성
    from sklearn.model_selection import cross_val_predict
    predictions = np.zeros((len(X), len(models)))
    
    for i, (name, model) in enumerate(models.items()):
        try:
            pred = cross_val_predict(model, X, y, cv=3)
            predictions[:, i] = pred
        except:
            predictions[:, i] = y.mean()  # fallback
    
    # 최적 가중치 찾기 (제약 조건: 가중치 합 = 1, 가중치 >= 0)
    def objective(weights):
        ensemble_pred = np.dot(predictions, weights)
        return -r2_score(y, ensemble_pred)  # 최소화를 위해 음수
    
    # 제약 조건
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(len(models))]
    
    # 초기값 (균등 가중치)
    initial_weights = np.ones(len(models)) / len(models)
    
    # 최적화
    result = minimize(objective, initial_weights, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    if result.success:
        optimal_weights = result.x
        optimal_score = -result.fun
        print(f"   최적화 성공: R² {optimal_score:.4f}")
        
        model_names = list(models.keys())
        for name, weight in zip(model_names, optimal_weights):
            print(f"   {name:15s}: {weight:.3f}")
    else:
        print("   최적화 실패 - 균등 가중치 사용")
        optimal_weights = np.ones(len(models)) / len(models)
    
    return dict(zip(models.keys(), optimal_weights))

def weighted_ensemble_predict(X, models, scaler, selected_features, weights):
    """가중치 기반 앙상블 예측"""
    X_selected = X[selected_features].copy()
    
    # 전처리
    for col in X_selected.columns:
        if X_selected[col].isna().sum() > 0:
            X_selected[col] = X_selected[col].fillna(X_selected[col].median())
    
    X_selected = X_selected.replace([np.inf, -np.inf], np.nan)
    for col in X_selected.columns:
        if X_selected[col].isna().sum() > 0:
            X_selected[col] = X_selected[col].fillna(X_selected[col].median())
    
    X_scaled = scaler.transform(X_selected)
    X_scaled = pd.DataFrame(X_scaled, columns=selected_features, index=X_selected.index)
    
    # 가중 예측
    predictions = []
    weight_list = []
    
    for name, model in models.items():
        try:
            pred = model.predict(X_scaled)
            predictions.append(pred)
            weight_list.append(weights.get(name, 0))
        except:
            continue
    
    if predictions:
        predictions = np.array(predictions)
        weight_list = np.array(weight_list)
        weight_list = weight_list / weight_list.sum()  # 정규화
        
        weighted_pred = np.average(predictions, axis=0, weights=weight_list)
        return weighted_pred, predictions
    else:
        return np.zeros(len(X_scaled)), np.array([])

def comprehensive_evaluation(y_true, y_pred, model_name="Improved Model"):
    """종합적인 평가"""
    print(f"\n📊 {model_name} 성능 평가:")
    
    # 기본 지표
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"   • R²: {r2:.4f}")
    print(f"   • RMSE: {rmse:.4f} ha")
    print(f"   • MAE: {mae:.4f} ha")
    print(f"   • MAPE: {mape:.2f}%")
    
    # 화재 규모별 성능
    q33, q67 = np.percentile(y_true, [33, 67])
    
    small_mask = y_true <= q33
    medium_mask = (y_true > q33) & (y_true <= q67)
    large_mask = y_true > q67
    
    print(f"   규모별 R²:")
    if small_mask.sum() > 5:
        small_r2 = r2_score(y_true[small_mask], y_pred[small_mask])
        print(f"   • 소형 화재: {small_r2:.4f}")
    
    if medium_mask.sum() > 5:
        medium_r2 = r2_score(y_true[medium_mask], y_pred[medium_mask])
        print(f"   • 중형 화재: {medium_r2:.4f}")
    
    if large_mask.sum() > 5:
        large_r2 = r2_score(y_true[large_mask], y_pred[large_mask])
        print(f"   • 대형 화재: {large_r2:.4f}")
    
    return {
        'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape
    }

def create_improvement_visualizations(y_true, y_pred, individual_preds, baseline_r2=0.66):
    """개선 결과 시각화"""
    print("🎨 개선 결과 시각화...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 실제 vs 예측 (개선 강조)
    r2 = r2_score(y_true, y_pred)
    improvement = r2 - baseline_r2
    
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=40, 
                      color='darkgreen' if improvement > 0 else 'darkred')
    max_val = max(y_true.max(), y_pred.max())
    axes[0, 0].plot([0, max_val], [0, max_val], 'r--', lw=2)
    axes[0, 0].set_xlabel('실제 화재 면적 (ha)')
    axes[0, 0].set_ylabel('예측 화재 면적 (ha)')
    
    title = f'개선된 모델 성능\nR² = {r2:.4f} '
    title += f'({"+" if improvement >= 0 else ""}{improvement:.4f})'
    axes[0, 0].set_title(title, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 개선도 비교
    metrics = ['기존 모델', '개선 모델']
    r2_values = [baseline_r2, r2]
    colors = ['lightblue', 'darkgreen' if r2 > baseline_r2 else 'orange']
    
    bars = axes[0, 1].bar(metrics, r2_values, color=colors, alpha=0.7)
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].set_title('모델 성능 비교')
    axes[0, 1].set_ylim(0, max(r2_values) * 1.1)
    
    # 값 표시
    for bar, val in zip(bars, r2_values):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 개선도 텍스트
    if improvement > 0:
        axes[0, 1].text(0.5, max(r2_values) * 0.5, f'+{improvement:.4f}\n({improvement/baseline_r2*100:+.1f}%)',
                       ha='center', va='center', fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # 3. 잔차 분포
    residuals = y_true - y_pred
    axes[0, 2].hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 2].axvline(x=0, color='red', linestyle='--', lw=2)
    axes[0, 2].set_xlabel('잔차 (ha)')
    axes[0, 2].set_ylabel('빈도')
    axes[0, 2].set_title(f'잔차 분포\n표준편차: {residuals.std():.3f}')
    
    # 4. 개별 모델 기여도
    if individual_preds.shape[0] > 1:
        model_names = [f'Model {i+1}' for i in range(individual_preds.shape[0])]
        model_r2s = []
        
        for i in range(individual_preds.shape[0]):
            try:
                r2_i = r2_score(y_true, individual_preds[i])
                model_r2s.append(r2_i)
            except:
                model_r2s.append(0)
        
        bars = axes[1, 0].bar(model_names, model_r2s, alpha=0.7)
        axes[1, 0].axhline(y=r2, color='red', linestyle='--', lw=2, 
                          label=f'앙상블: {r2:.4f}')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].set_title('개별 모델 vs 앙상블')
        axes[1, 0].legend()
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. 예측 정확도 분포
    relative_error = np.abs(residuals) / y_true
    accuracy_bins = [0, 0.1, 0.25, 0.5, 1.0, np.inf]
    accuracy_labels = ['<10%', '10-25%', '25-50%', '50-100%', '>100%']
    accuracy_colors = ['darkgreen', 'green', 'yellow', 'orange', 'red']
    
    accuracy_counts = []
    for i in range(len(accuracy_bins)-1):
        count = ((relative_error >= accuracy_bins[i]) & 
                (relative_error < accuracy_bins[i+1])).sum()
        accuracy_counts.append(count)
    
    wedges, texts, autotexts = axes[1, 1].pie(accuracy_counts, labels=accuracy_labels, 
                                             colors=accuracy_colors, autopct='%1.1f%%',
                                             startangle=90)
    axes[1, 1].set_title('예측 정확도 분포')
    
    # 6. 개선 요약 통계
    axes[1, 2].axis('off')
    
    # 통계 텍스트
    stats_text = f"""
개선 결과 요약

기존 성능: R² {baseline_r2:.4f}
개선 성능: R² {r2:.4f}
개선도: {improvement:+.4f} ({improvement/baseline_r2*100:+.1f}%)

RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.3f} ha
MAE: {mean_absolute_error(y_true, y_pred):.3f} ha

정확도 50% 이내: {(relative_error <= 0.5).mean()*100:.1f}%
정확도 25% 이내: {(relative_error <= 0.25).mean()*100:.1f}%
    """
    
    axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes,
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('gradual_improvement_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 시각화 저장: gradual_improvement_results.png")

def main():
    """메인 함수 - 점진적 개선"""
    print("🎯 점진적 모델 개선 프로세스")
    print("=" * 60)
    
    try:
        # 1. 현재 모델 분석
        config = analyze_current_model_limitations()
        if config is None:
            print("기존 모델이 없어 종료합니다.")
            return
        
        baseline_r2 = config['final_score']
        
        # 2. 데이터 준비
        fire_data, base_features = load_and_prepare_data()
        
        # 3. 타겟 변환 최적화
        best_transformer, transform_name = optimize_target_transformation(fire_data['fire_area'])
        
        # 4. 피처 개선
        fire_data_enhanced, enhanced_features = create_enhanced_interactions(fire_data, base_features)
        
        # 5. 피처 선택 (기존 + 새로운 상호작용)
        target_col = 'fire_area'
        available_features = [col for col in enhanced_features 
                            if col in fire_data_enhanced.columns 
                            and col != target_col
                            and fire_data_enhanced[col].dtype in ['int64', 'float64']]
        
        X = fire_data_enhanced[available_features].copy()
        y = fire_data_enhanced[target_col].copy()
        
        print(f"\n📊 최종 데이터:")
        print(f"   - 샘플: {len(X)}개")
        print(f"   - 피처: {len(available_features)}개")
        print(f"   - 타겟 변환: {transform_name}")
        
        # 6. 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=pd.qcut(y, q=5, duplicates='drop')
        )
        
        # 7. 개선된 앙상블 학습
        models, scaler, weights = create_improved_ensemble(
            X_train, y_train, available_features, best_transformer
        )
        
        # 8. 예측 및 평가
        print("\n🎯 최종 예측 및 평가...")
        
        y_pred_test, individual_preds = weighted_ensemble_predict(
            X_test, models, scaler, available_features, weights
        )
        
        # 9. 성능 평가
        print("\n" + "=" * 60)
        print("🏆 점진적 개선 결과")
        print("=" * 60)
        
        test_metrics = comprehensive_evaluation(y_test, y_pred_test, "개선된 모델")
        
        # 개선도 계산
        improvement = test_metrics['r2'] - baseline_r2
        improvement_pct = (improvement / baseline_r2) * 100
        
        print(f"\n📈 개선 요약:")
        print(f"   • 기존 모델: R² {baseline_r2:.4f}")
        print(f"   • 개선 모델: R² {test_metrics['r2']:.4f}")
        print(f"   • 절대 개선도: {improvement:+.4f}")
        print(f"   • 상대 개선도: {improvement_pct:+.2f}%")
        
        # 10. 시각화
        create_improvement_visualizations(y_test, y_pred_test, individual_preds, baseline_r2)
        
        # 11. 모델 저장
        if improvement > 0:
            print(f"\n💾 개선된 모델 저장...")
            
            model_package = {
                'models': models,
                'scaler': scaler,
                'weights': weights,
                'transformer': best_transformer,
                'transform_name': transform_name,
                'selected_features': available_features,
                'metrics': test_metrics,
                'improvement': improvement
            }
            
            joblib.dump(model_package, 'improved_fire_model.joblib')
            
            config_improved = {
                'model_type': 'Gradually Improved Model',
                'baseline_r2': baseline_r2,
                'improved_r2': test_metrics['r2'],
                'improvement': improvement,
                'improvement_pct': improvement_pct,
                'transform_method': transform_name,
                'n_models': len(models),
                'n_features': len(available_features)
            }
            
            with open('improved_model_config.json', 'w') as f:
                json.dump(config_improved, f, indent=2)
            
            print("✅ 개선된 모델 저장 완료!")
        else:
            print("📊 개선이 없어 저장하지 않습니다.")
        
        # 12. 최종 결론
        if improvement > 0.05:
            result = "🎉 의미있는 개선 달성!"
            grade = "🟢 성공적 개선"
        elif improvement > 0.02:
            result = "📈 소폭 개선 달성"
            grade = "🟡 부분적 개선"
        elif improvement > 0:
            result = "📊 미미한 개선"
            grade = "🟡 미미한 개선"
        else:
            result = "❌ 개선 실패"
            grade = "🔴 개선 없음"
        
        print(f"\n{result}")
        print(f"성능 등급: {grade}")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()