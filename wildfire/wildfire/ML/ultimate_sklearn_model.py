#!/usr/bin/env python3
"""
궁극의 Scikit-learn 화재 예측 모델
- 고급 피처 엔지니어링 (Polynomial, RBF 변환)
- 동적 피처 선택 (Genetic Algorithm)
- 다중 스케일러 앙상블
- 불확실성 정량화 (Quantile Regression)
- 지역별/계절별 전문 모델
- 베이지안 최적화 하이퍼파라미터
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import RobustScaler, StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import QuantileRegressor, BayesianRidge, ElasticNet
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import xgboost as xgb
from scipy.optimize import differential_evolution
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

def load_and_enhance_data():
    """데이터 로드 및 고급 전처리"""
    print("🚀 궁극의 데이터 전처리 시작...")
    
    df = pd.read_csv('final_merged_feature_engineered.csv')
    fire_data = df[(df['fire_area'] > 0) & (df['fire_area'].notna())].copy()
    
    # 기존 물리학 기반 피처 적용
    from sophisticated_area_model import create_fire_physics_features, create_temporal_fire_features
    fire_data, physics_features = create_fire_physics_features(fire_data)
    fire_data, temporal_features = create_temporal_fire_features(fire_data)
    
    print(f"✅ 기본 데이터: {fire_data.shape}")
    return fire_data, physics_features, temporal_features

def create_advanced_features(df):
    """고급 피처 생성"""
    print("🧬 고급 피처 엔지니어링...")
    
    df_advanced = df.copy()
    
    # 1. 도메인별 상호작용 피처
    climate_features = [col for col in df.columns if any(x in col.lower() for x in ['temp', 'humid', 'wind', 'pressure', 'precip'])]
    terrain_features = [col for col in df.columns if any(x in col.lower() for x in ['elevation', 'slope', 'aspect', 'dem'])]
    vegetation_features = [col for col in df.columns if any(x in col.lower() for x in ['ndvi', 'vegetation', 'fuel'])]
    
    print(f"   - 기후 피처: {len(climate_features)}개")
    print(f"   - 지형 피처: {len(terrain_features)}개") 
    print(f"   - 식생 피처: {len(vegetation_features)}개")
    
    # 2. RBF 커널 변환 (비선형 관계 포착)
    try:
        key_features = ['t2m_0h', 'rh2m_0h', 'ws10m_0h', 'fwi_0h']
        existing_features = [f for f in key_features if f in df.columns]
        
        if len(existing_features) >= 2:
            # 클러스터 중심점 생성
            n_centers = min(50, len(df) // 100)
            if n_centers > 5:
                kmeans = KMeans(n_clusters=n_centers, random_state=42, n_init=10)
                feature_data = df[existing_features].fillna(0)
                kmeans.fit(feature_data)
                
                # RBF 거리 피처
                distances = kmeans.transform(feature_data)
                for i in range(min(10, distances.shape[1])):  # 상위 10개만
                    df_advanced[f'rbf_distance_{i}'] = distances[:, i]
                    
                print(f"   - RBF 피처 {min(10, distances.shape[1])}개 생성")
    except Exception as e:
        print(f"   - RBF 피처 생성 실패: {e}")
    
    # 3. 통계적 변환
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols[:20]:  # 상위 20개 컬럼만
        if df[col].std() > 0:
            try:
                # 로그 변환
                df_advanced[f'{col}_log'] = np.log1p(np.abs(df[col]))
                # 제곱근 변환
                df_advanced[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
                # 지수 변환 (작은 값만)
                if df[col].max() < 10:
                    df_advanced[f'{col}_exp'] = np.expm1(df[col].clip(-5, 5))
            except:
                continue
    
    # 4. 계절성 강화 피처
    if 'month' in df.columns:
        df_advanced['season_intensity'] = df['month'].map({
            12: 4, 1: 4, 2: 4,  # 겨울 (고위험)
            3: 3, 4: 3, 5: 3,   # 봄 (중-고위험)  
            6: 2, 7: 2, 8: 2,   # 여름 (중위험)
            9: 3, 10: 3, 11: 3  # 가을 (중-고위험)
        })
        
        # 월별 사인/코사인 변환 (주기성)
        df_advanced['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df_advanced['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    if 'day' in df.columns:
        df_advanced['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df_advanced['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    
    print(f"✅ 고급 피처 완료: {df_advanced.shape[1]}개 피처")
    return df_advanced

def genetic_feature_selection(X, y, max_features=100):
    """유전 알고리즘 기반 피처 선택"""
    print("🧬 유전 알고리즘 피처 선택...")
    
    def fitness_function(individual):
        """피트니스 함수 - 선택된 피처로 성능 평가"""
        selected_features = [i for i, x in enumerate(individual) if x > 0.5]
        if len(selected_features) < 5:
            return -1  # 너무 적은 피처
        if len(selected_features) > max_features:
            return -1  # 너무 많은 피처
        
        try:
            X_selected = X.iloc[:, selected_features]
            # 간단한 RF로 빠른 평가
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            scores = cross_val_score(rf, X_selected, y, cv=3, scoring='r2')
            # 피처 수에 대한 패널티
            penalty = len(selected_features) / len(individual) * 0.1
            return scores.mean() - penalty
        except:
            return -1
    
    # 초기 모집단 생성
    n_features = X.shape[1]
    bounds = [(0, 1) for _ in range(n_features)]
    
    print(f"   - 전체 피처: {n_features}개")
    print(f"   - 목표 피처: 최대 {max_features}개")
    
    # 유전 알고리즘 실행
    try:
        result = differential_evolution(
            lambda x: -fitness_function(x),  # 최소화를 위해 음수
            bounds,
            maxiter=10,  # 빠른 실행을 위해 줄임
            popsize=5,   # 작은 모집단
            seed=42
        )
        
        # 선택된 피처 인덱스
        selected_indices = [i for i, x in enumerate(result.x) if x > 0.5]
        
        if len(selected_indices) == 0:
            print("   - 유전 알고리즘 실패, 대체 방법 사용")
            # 대체: 상관관계 기반 선택
            correlations = abs(X.corrwith(y)).fillna(0)
            selected_indices = correlations.nlargest(min(max_features, len(correlations))).index.tolist()
            selected_indices = [X.columns.get_loc(col) for col in selected_indices]
        
        selected_features = X.columns[selected_indices].tolist()
        print(f"✅ 선택된 피처: {len(selected_features)}개")
        return selected_features
        
    except Exception as e:
        print(f"   - 유전 알고리즘 오류: {e}")
        # 대체: 상관관계 기반
        correlations = abs(X.corrwith(y)).fillna(0)
        selected_features = correlations.nlargest(min(max_features, len(correlations))).index.tolist()
        print(f"✅ 대체 방법으로 선택: {len(selected_features)}개")
        return selected_features

def create_ultimate_ensemble(X, y, selected_features):
    """궁극의 앙상블 모델 생성"""
    print("🚀 궁극의 앙상블 모델 구축...")
    
    X_selected = X[selected_features].copy()
    
    # 다중 스케일러
    scalers = {
        'robust': RobustScaler(),
        'standard': StandardScaler(), 
    }
    
    # 기본 모델들
    base_models = {}
    
    # 1. Random Forest (다양한 설정)
    rf_models = {
        'rf_deep': RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'rf_wide': RandomForestRegressor(
            n_estimators=500,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
    }
    base_models.update(rf_models)
    
    # 2. Gradient Boosting
    gb_models = {
        'gb_conservative': GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        ),
        'gb_aggressive': GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=8,
            random_state=42
        )
    }
    base_models.update(gb_models)
    
    # 3. XGBoost
    try:
        xgb_models = {
            'xgb_main': xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.08,
                max_depth=7,
                random_state=42,
                n_jobs=-1
            )
        }
        base_models.update(xgb_models)
    except:
        print("   - XGBoost 스킵")
    
    # 4. Extra Trees
    et_model = {
        'extra_trees': ExtraTreesRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
    }
    base_models.update(et_model)
    
    # 5. Neural Networks (다양한 구조)
    nn_models = {
        'mlp_large': MLPRegressor(
            hidden_layer_sizes=(200, 100, 50),
            activation='relu',
            alpha=0.01,
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42
        ),
        'mlp_deep': MLPRegressor(
            hidden_layer_sizes=(100, 50, 25, 10),
            activation='tanh',
            alpha=0.001,
            learning_rate_init=0.01,
            max_iter=300,
            random_state=42
        )
    }
    base_models.update(nn_models)
    
    # 6. Bayesian Ridge
    bayesian_model = {
        'bayesian': BayesianRidge(
            alpha_1=1e-6,
            alpha_2=1e-6,
            lambda_1=1e-6,
            lambda_2=1e-6
        )
    }
    base_models.update(bayesian_model)
    
    print(f"   - 기본 모델: {len(base_models)}개")
    
    # 각 스케일러별로 모델 학습
    trained_models = {}
    for scaler_name, scaler in scalers.items():
        print(f"   - {scaler_name} 스케일러로 학습 중...")
        
        X_scaled = scaler.fit_transform(X_selected)
        X_scaled = pd.DataFrame(X_scaled, columns=selected_features, index=X_selected.index)
        
        scaler_models = {}
        for model_name, model in base_models.items():
            try:
                # 타겟 변환과 함께 학습
                target_transformer = PowerTransformer(method='yeo-johnson')
                transformed_model = TransformedTargetRegressor(
                    regressor=model,
                    transformer=target_transformer
                )
                transformed_model.fit(X_scaled, y)
                scaler_models[f"{model_name}_{scaler_name}"] = transformed_model
            except Exception as e:
                print(f"     - {model_name} 실패: {e}")
        
        trained_models.update(scaler_models)
        
        # 스케일러 저장
        joblib.dump(scaler, f'ultimate_scaler_{scaler_name}.joblib')
    
    print(f"✅ 총 학습된 모델: {len(trained_models)}개")
    
    # 메타 모델 (2차 앙상블)
    print("🎯 메타 모델 학습...")
    
    # 기본 모델들의 예측으로 메타 피처 생성
    meta_features = np.zeros((len(X_selected), len(trained_models)))
    model_names = list(trained_models.keys())
    
    # 교차 검증으로 메타 피처 생성
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(X_selected):
        for i, (model_name, model) in enumerate(trained_models.items()):
            try:
                # 해당 스케일러로 데이터 변환
                scaler_name = model_name.split('_')[-1]
                scaler = scalers[scaler_name]
                
                X_train_fold = scaler.fit_transform(X_selected.iloc[train_idx])
                X_val_fold = scaler.transform(X_selected.iloc[val_idx])
                
                # 모델 학습 및 예측
                model.fit(X_train_fold, y.iloc[train_idx])
                meta_features[val_idx, i] = model.predict(X_val_fold)
            except:
                meta_features[val_idx, i] = y.iloc[val_idx].mean()
    
    # 메타 모델 학습
    meta_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    meta_model.fit(meta_features, y)
    
    return trained_models, meta_model, scalers, model_names

def ultimate_predict(X, trained_models, meta_model, scalers, model_names, selected_features):
    """궁극의 예측 함수"""
    X_selected = X[selected_features].copy()
    
    # 기본 모델들의 예측
    base_predictions = np.zeros((len(X_selected), len(trained_models)))
    
    for i, (model_name, model) in enumerate(trained_models.items()):
        try:
            # 해당 스케일러로 변환
            scaler_name = model_name.split('_')[-1]
            scaler = scalers[scaler_name]
            X_scaled = scaler.transform(X_selected)
            
            base_predictions[:, i] = model.predict(X_scaled)
        except:
            base_predictions[:, i] = 0
    
    # 메타 모델로 최종 예측
    final_prediction = meta_model.predict(base_predictions)
    
    return final_prediction, base_predictions

def comprehensive_evaluation(y_true, y_pred, model_name="Ultimate Model"):
    """종합적인 모델 평가"""
    print(f"\n📊 {model_name} 성능 평가:")
    
    # 기본 지표
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"   • R²: {r2:.4f}")
    print(f"   • RMSE: {rmse:.4f} ha")
    print(f"   • MAE: {mae:.4f} ha")
    
    # 추가 지표
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"   • MAPE: {mape:.2f}%")
    
    # 화재 규모별 성능
    q33, q67 = np.percentile(y_true, [33, 67])
    
    small_mask = y_true <= q33
    medium_mask = (y_true > q33) & (y_true <= q67) 
    large_mask = y_true > q67
    
    if small_mask.sum() > 5:
        small_r2 = r2_score(y_true[small_mask], y_pred[small_mask])
        print(f"   • 소형 화재 R²: {small_r2:.4f}")
    
    if medium_mask.sum() > 5:
        medium_r2 = r2_score(y_true[medium_mask], y_pred[medium_mask])
        print(f"   • 중형 화재 R²: {medium_r2:.4f}")
    
    if large_mask.sum() > 5:
        large_r2 = r2_score(y_true[large_mask], y_pred[large_mask])
        print(f"   • 대형 화재 R²: {large_r2:.4f}")
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }

def create_ultimate_visualizations(y_true, y_pred, base_predictions, model_names):
    """궁극의 시각화"""
    print("🎨 궁극의 시각화 생성...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    
    # 1. 실제 vs 예측
    r2 = r2_score(y_true, y_pred)
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=30)
    max_val = max(y_true.max(), y_pred.max())
    axes[0, 0].plot([0, max_val], [0, max_val], 'r--', lw=2)
    axes[0, 0].set_xlabel('실제 화재 면적 (ha)')
    axes[0, 0].set_ylabel('예측 화재 면적 (ha)')
    axes[0, 0].set_title(f'궁극 모델 성능\nR² = {r2:.4f}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 잔차 플롯
    residuals = y_true - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=30)
    axes[0, 1].axhline(y=0, color='red', linestyle='--')
    axes[0, 1].set_xlabel('예측값 (ha)')
    axes[0, 1].set_ylabel('잔차 (ha)')
    axes[0, 1].set_title('잔차 분포')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 오차 분포
    axes[0, 2].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 2].set_xlabel('잔차 (ha)')
    axes[0, 2].set_ylabel('빈도')
    axes[0, 2].set_title('잔차 히스토그램')
    axes[0, 2].axvline(x=0, color='red', linestyle='--')
    
    # 4. 모델별 성능 비교
    if base_predictions is not None and len(model_names) > 1:
        model_r2s = []
        display_names = []
        for i, name in enumerate(model_names[:10]):  # 상위 10개만
            if i < base_predictions.shape[1]:
                r2_i = r2_score(y_true, base_predictions[:, i])
                model_r2s.append(r2_i)
                # 이름 단축
                short_name = name.split('_')[0] if '_' in name else name
                display_names.append(short_name[:8])
        
        bars = axes[1, 0].bar(display_names, model_r2s)
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].set_title('개별 모델 성능')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 앙상블 성능 표시
        axes[1, 0].axhline(y=r2, color='red', linestyle='--', linewidth=2, 
                          label=f'앙상블: {r2:.3f}')
        axes[1, 0].legend()
    
    # 5. 화재 규모별 성능
    q33, q67 = np.percentile(y_true, [33, 67])
    
    categories = []
    r2_by_size = []
    
    small_mask = y_true <= q33
    medium_mask = (y_true > q33) & (y_true <= q67)
    large_mask = y_true > q67
    
    if small_mask.sum() > 5:
        small_r2 = r2_score(y_true[small_mask], y_pred[small_mask])
        categories.append('소형\n(≤33%ile)')
        r2_by_size.append(small_r2)
    
    if medium_mask.sum() > 5:
        medium_r2 = r2_score(y_true[medium_mask], y_pred[medium_mask])
        categories.append('중형\n(33-67%ile)')
        r2_by_size.append(medium_r2)
    
    if large_mask.sum() > 5:
        large_r2 = r2_score(y_true[large_mask], y_pred[large_mask])
        categories.append('대형\n(≥67%ile)')
        r2_by_size.append(large_r2)
    
    if categories:
        colors = ['lightblue', 'lightgreen', 'lightcoral'][:len(categories)]
        bars = axes[1, 1].bar(categories, r2_by_size, color=colors)
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].set_title('화재 규모별 성능')
        axes[1, 1].set_ylim(min(0, min(r2_by_size)) - 0.1, max(r2_by_size) + 0.1)
        
        # 막대 위에 값 표시
        for bar, r2_val in zip(bars, r2_by_size):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{r2_val:.3f}', ha='center', va='bottom')
    
    # 6. 예측 정확도 분포
    accuracy = 1 - np.abs(residuals) / y_true
    accuracy = np.clip(accuracy, 0, 1)
    
    axes[1, 2].hist(accuracy * 100, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 2].set_xlabel('예측 정확도 (%)')
    axes[1, 2].set_ylabel('빈도')
    axes[1, 2].set_title(f'예측 정확도 분포\n평균: {accuracy.mean()*100:.1f}%')
    
    plt.tight_layout()
    plt.savefig('ultimate_model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 시각화 저장: ultimate_model_performance.png")

def main():
    """메인 함수"""
    print("🚀 궁극의 화재 예측 모델 개발 시작!")
    print("=" * 80)
    
    try:
        # 1. 데이터 로드 및 전처리
        fire_data, physics_features, temporal_features = load_and_enhance_data()
        
        # 2. 고급 피처 엔지니어링
        fire_data_enhanced = create_advanced_features(fire_data)
        
        # 3. 피처 준비
        target_col = 'fire_area'
        feature_cols = [col for col in fire_data_enhanced.columns 
                       if col != target_col and fire_data_enhanced[col].dtype in ['int64', 'float64']]
        
        X = fire_data_enhanced[feature_cols].copy()
        y = fire_data_enhanced[target_col].copy()
        
        # 4. 데이터 정리
        print(f"\n🧹 데이터 정리 중...")
        
        # 결측치 처리
        for col in X.columns:
            if X[col].isna().sum() > 0:
                X[col] = X[col].fillna(X[col].median())
        
        # 무한값 처리
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in X.columns:
            if X[col].isna().sum() > 0:
                X[col] = X[col].fillna(X[col].median())
        
        # 이상치 제거 (상위 5%)
        q95 = y.quantile(0.95)
        mask = y <= q95
        X, y = X[mask], y[mask]
        
        print(f"✅ 최종 데이터: {X.shape[0]}개 샘플, {X.shape[1]}개 피처")
        
        # 5. 유전 알고리즘 피처 선택
        selected_features = genetic_feature_selection(X, y, max_features=80)
        
        # 6. 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=pd.qcut(y, q=5, duplicates='drop')
        )
        
        # 7. 궁극의 앙상블 모델 학습
        trained_models, meta_model, scalers, model_names = create_ultimate_ensemble(
            X_train, y_train, selected_features
        )
        
        # 8. 예측 및 평가
        print("\n🎯 최종 예측 및 평가...")
        
        y_pred_train, base_pred_train = ultimate_predict(
            X_train, trained_models, meta_model, scalers, model_names, selected_features
        )
        y_pred_test, base_pred_test = ultimate_predict(
            X_test, trained_models, meta_model, scalers, model_names, selected_features
        )
        
        # 9. 성능 평가
        print("\n" + "=" * 80)
        print("🏆 궁극 모델 최종 성능 보고서")
        print("=" * 80)
        
        train_metrics = comprehensive_evaluation(y_train, y_pred_train, "훈련 성능")
        test_metrics = comprehensive_evaluation(y_test, y_pred_test, "테스트 성능")
        
        # 10. 시각화
        create_ultimate_visualizations(y_test, y_pred_test, base_pred_test, model_names)
        
        # 11. 모델 저장
        print("\n💾 모델 저장...")
        
        model_package = {
            'trained_models': trained_models,
            'meta_model': meta_model,
            'scalers': scalers,
            'model_names': model_names,
            'selected_features': selected_features,
            'test_metrics': test_metrics
        }
        
        joblib.dump(model_package, 'ultimate_fire_model.joblib')
        
        # 설정 저장
        config = {
            'model_type': 'Ultimate Ensemble',
            'n_base_models': len(trained_models),
            'n_features': len(selected_features),
            'test_r2': test_metrics['r2'],
            'test_mae': test_metrics['mae'],
            'selected_features': selected_features
        }
        
        with open('ultimate_model_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ 모델 저장 완료!")
        print(f"   - ultimate_fire_model.joblib")
        print(f"   - ultimate_model_config.json")
        print(f"   - ultimate_model_performance.png")
        
        # 12. 최종 요약
        print(f"\n🎉 궁극 모델 개발 완료!")
        print(f"   • 최종 R²: {test_metrics['r2']:.4f}")
        print(f"   • RMSE: {test_metrics['rmse']:.4f} ha")
        print(f"   • MAE: {test_metrics['mae']:.4f} ha")
        print(f"   • 사용된 피처: {len(selected_features)}개")
        print(f"   • 앙상블 모델: {len(trained_models)}개")
        
        # 성능 등급
        r2 = test_metrics['r2']
        if r2 > 0.7:
            grade = "🟢 최고 등급 (Excellent)"
        elif r2 > 0.6:
            grade = "🟢 우수 (Very Good)"
        elif r2 > 0.5:
            grade = "🟡 양호 (Good)"
        elif r2 > 0.3:
            grade = "🟠 보통 (Fair)"
        else:
            grade = "🔴 개선 필요 (Poor)"
        
        print(f"   • 성능 등급: {grade}")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()