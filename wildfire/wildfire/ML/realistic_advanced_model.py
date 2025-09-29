#!/usr/bin/env python3
"""
현실적인 고급 화재 예측 모델
- 엄격한 오버피팅 방지
- 시간적 분할 (Temporal Split)
- 보수적인 피처 선택
- 신뢰구간 제공
- 실용적 성능 목표 (R² 0.7-0.8)
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import TransformedTargetRegressor
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """데이터 로드 및 시간적 정렬"""
    print("📅 시간적 데이터 준비...")
    
    df = pd.read_csv('final_merged_feature_engineered.csv')
    fire_data = df[(df['fire_area'] > 0) & (df['fire_area'].notna())].copy()
    
    # 시간 정보 확인
    time_cols = [col for col in fire_data.columns if 'year' in col.lower() or 'month' in col.lower() or 'day' in col.lower()]
    print(f"   시간 관련 컬럼: {time_cols}")
    
    # 기존 물리학 기반 피처 적용
    from sophisticated_area_model import create_fire_physics_features, create_temporal_fire_features
    fire_data, physics_features = create_fire_physics_features(fire_data)
    fire_data, temporal_features = create_temporal_fire_features(fire_data)
    
    return fire_data, physics_features + temporal_features

def conservative_feature_engineering(df, base_features):
    """보수적인 피처 엔지니어링 - 물리적 의미가 있는 피처만"""
    print("🧬 보수적 피처 엔지니어링...")
    
    df_conservative = df.copy()
    new_features = base_features.copy()
    
    # 1. 검증된 상호작용만 추가
    climate_vars = ['t2m_0h', 'rh2m_0h', 'ws10m_0h', 'precip_0h']
    existing_climate = [v for v in climate_vars if v in df.columns]
    
    if len(existing_climate) >= 2:
        # 온도-습도 상호작용 (화재 위험도)
        if 't2m_0h' in df.columns and 'rh2m_0h' in df.columns:
            df_conservative['temp_humidity_risk'] = df['t2m_0h'] / (df['rh2m_0h'] + 1)
            new_features.append('temp_humidity_risk')
        
        # 바람-습도 상호작용
        if 'ws10m_0h' in df.columns and 'rh2m_0h' in df.columns:
            df_conservative['wind_drying_effect'] = df['ws10m_0h'] * (100 - df['rh2m_0h']) / 100
            new_features.append('wind_drying_effect')
    
    # 2. 계절성 강화 (물리적 근거)
    if 'month' in df.columns:
        # 화재 계절 강도
        fire_season_map = {
            1: 0.8, 2: 0.9, 3: 0.7,    # 겨울-봄 전환
            4: 0.8, 5: 0.9, 6: 0.6,    # 봄-여름
            7: 0.4, 8: 0.5, 9: 0.7,    # 여름-가을  
            10: 0.8, 11: 0.9, 12: 0.9  # 가을-겨울
        }
        df_conservative['fire_season_risk'] = df['month'].map(fire_season_map)
        new_features.append('fire_season_risk')
    
    # 3. 지형적 화재 전파 요인
    terrain_features = [col for col in df.columns if any(x in col.lower() for x in ['elevation', 'slope', 'aspect'])]
    if len(terrain_features) >= 2:
        # 경사-고도 상호작용 (단순화)
        slope_cols = [col for col in terrain_features if 'slope' in col.lower()]
        elev_cols = [col for col in terrain_features if 'elevation' in col.lower()]
        
        if slope_cols and elev_cols:
            slope_col = slope_cols[0]
            elev_col = elev_cols[0]
            df_conservative['terrain_fire_risk'] = np.sqrt(df[slope_col] * df[elev_col] / 1000)
            new_features.append('terrain_fire_risk')
    
    print(f"   추가된 피처: {len(new_features) - len(base_features)}개")
    print(f"   총 피처: {len(new_features)}개")
    
    return df_conservative, new_features

def intelligent_feature_selection(X, y, max_features=50):
    """지능적이고 보수적인 피처 선택"""
    print("🎯 지능적 피처 선택...")
    
    # 1. 상관관계 기반 초기 선별
    correlations = abs(X.corrwith(y)).fillna(0)
    high_corr_features = correlations[correlations > 0.05].index.tolist()
    
    print(f"   상관관계 > 0.05: {len(high_corr_features)}개")
    
    # 2. Random Forest 중요도 기반 선택
    if len(high_corr_features) > max_features:
        X_reduced = X[high_corr_features]
        
        # 결측치 처리
        for col in X_reduced.columns:
            if X_reduced[col].isna().sum() > 0:
                X_reduced[col] = X_reduced[col].fillna(X_reduced[col].median())
        
        rf_selector = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        rf_selector.fit(X_reduced, y)
        
        # 중요도 기준 선택
        feature_importance = pd.Series(
            rf_selector.feature_importances_,
            index=X_reduced.columns
        )
        selected_features = feature_importance.nlargest(max_features).index.tolist()
    else:
        selected_features = high_corr_features
    
    print(f"   최종 선택: {len(selected_features)}개")
    return selected_features

def create_temporal_split(df, test_ratio=0.2):
    """시간적 분할 - 과거 데이터로 훈련, 미래 데이터로 테스트"""
    print("⏰ 시간적 데이터 분할...")
    
    # 시간 순서로 정렬 시도
    if 'year' in df.columns and 'month' in df.columns:
        df_sorted = df.sort_values(['year', 'month', 'day'] if 'day' in df.columns else ['year', 'month'])
        
        split_idx = int(len(df_sorted) * (1 - test_ratio))
        train_data = df_sorted.iloc[:split_idx]
        test_data = df_sorted.iloc[split_idx:]
        
        print(f"   훈련: {len(train_data)}개 ({train_data['year'].min() if 'year' in df.columns else '?'}-{train_data['year'].max() if 'year' in df.columns else '?'})")
        print(f"   테스트: {len(test_data)}개 ({test_data['year'].min() if 'year' in df.columns else '?'}-{test_data['year'].max() if 'year' in df.columns else '?'})")
        
        return train_data, test_data
    else:
        print("   시간 정보 없음 - 무작위 분할 사용")
        return train_test_split(df, test_size=test_ratio, random_state=42)

def create_robust_ensemble(X_train, y_train, selected_features):
    """견고한 앙상블 모델"""
    print("🏗️ 견고한 앙상블 구축...")
    
    X_selected = X_train[selected_features].copy()
    
    # 결측치 처리
    for col in X_selected.columns:
        if X_selected[col].isna().sum() > 0:
            X_selected[col] = X_selected[col].fillna(X_selected[col].median())
    
    # 무한값 처리
    X_selected = X_selected.replace([np.inf, -np.inf], np.nan)
    for col in X_selected.columns:
        if X_selected[col].isna().sum() > 0:
            X_selected[col] = X_selected[col].fillna(X_selected[col].median())
    
    # 스케일러
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_selected)
    X_scaled = pd.DataFrame(X_scaled, columns=selected_features, index=X_selected.index)
    
    # 보수적인 모델들
    models = {}
    
    # 1. Random Forest (오버피팅 방지)
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,           # 깊이 제한
        min_samples_split=20,   # 분할 최소 샘플 증가
        min_samples_leaf=10,    # 리프 최소 샘플 증가
        max_features='sqrt',    # 피처 서브샘플링
        random_state=42,
        n_jobs=-1
    )
    
    # 타겟 변환과 함께 학습
    target_transformer = PowerTransformer(method='yeo-johnson')
    rf_transformed = TransformedTargetRegressor(
        regressor=rf_model,
        transformer=target_transformer
    )
    rf_transformed.fit(X_scaled, y_train)
    models['random_forest'] = rf_transformed
    
    # 2. Gradient Boosting (학습률 낮춤)
    gb_model = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,     # 낮은 학습률
        max_depth=5,            # 얕은 트리
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,          # 배깅 효과
        random_state=42
    )
    
    gb_transformed = TransformedTargetRegressor(
        regressor=gb_model,
        transformer=PowerTransformer(method='yeo-johnson')
    )
    gb_transformed.fit(X_scaled, y_train)
    models['gradient_boosting'] = gb_transformed
    
    # 3. Bayesian Ridge (정규화)
    bayesian_model = BayesianRidge(
        alpha_1=1e-6,
        alpha_2=1e-6,
        lambda_1=1e-6,
        lambda_2=1e-6,
        compute_score=True
    )
    
    bayesian_transformed = TransformedTargetRegressor(
        regressor=bayesian_model,
        transformer=PowerTransformer(method='yeo-johnson')
    )
    bayesian_transformed.fit(X_scaled, y_train)
    models['bayesian_ridge'] = bayesian_transformed
    
    print(f"   학습된 모델: {len(models)}개")
    
    return models, scaler

def ensemble_predict_with_uncertainty(X, models, scaler, selected_features):
    """불확실성을 포함한 앙상블 예측"""
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
    
    # 각 모델의 예측
    predictions = []
    for name, model in models.items():
        try:
            pred = model.predict(X_scaled)
            predictions.append(pred)
        except:
            predictions.append(np.zeros(len(X_scaled)))
    
    predictions = np.array(predictions)
    
    # 앙상블 예측 (평균)
    ensemble_pred = np.mean(predictions, axis=0)
    
    # 불확실성 추정 (표준편차)
    uncertainty = np.std(predictions, axis=0)
    
    return ensemble_pred, uncertainty, predictions

def comprehensive_evaluation_with_confidence(y_true, y_pred, uncertainty):
    """신뢰구간을 포함한 종합 평가"""
    print("📊 종합 성능 평가:")
    
    # 기본 지표
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"   • R²: {r2:.4f}")
    print(f"   • RMSE: {rmse:.4f} ha")  
    print(f"   • MAE: {mae:.4f} ha")
    
    # 불확실성 지표
    avg_uncertainty = np.mean(uncertainty)
    print(f"   • 평균 불확실성: ±{avg_uncertainty:.4f} ha")
    
    # 신뢰구간 내 정확도
    within_1_std = np.sum(np.abs(y_true - y_pred) <= uncertainty) / len(y_true)
    within_2_std = np.sum(np.abs(y_true - y_pred) <= 2 * uncertainty) / len(y_true)
    
    print(f"   • 1σ 신뢰구간 정확도: {within_1_std:.1%}")
    print(f"   • 2σ 신뢰구간 정확도: {within_2_std:.1%}")
    
    # 실용성 평가
    relative_error = np.abs(y_true - y_pred) / y_true
    good_predictions = (relative_error <= 0.5).sum() / len(relative_error)
    
    print(f"   • 50% 이내 정확도: {good_predictions:.1%}")
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'uncertainty': avg_uncertainty,
        'confidence_1std': within_1_std,
        'confidence_2std': within_2_std,
        'practical_accuracy': good_predictions
    }

def create_realistic_visualizations(y_true, y_pred, uncertainty, predictions_individual):
    """현실적인 시각화"""
    print("🎨 현실적 시각화 생성...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 실제 vs 예측 (신뢰구간 포함)
    r2 = r2_score(y_true, y_pred)
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=30)
    
    # 신뢰구간 표시
    axes[0, 0].errorbar(y_true, y_pred, yerr=uncertainty, fmt='none', 
                       alpha=0.3, color='gray', capsize=2)
    
    max_val = max(y_true.max(), y_pred.max())
    axes[0, 0].plot([0, max_val], [0, max_val], 'r--', lw=2)
    axes[0, 0].set_xlabel('실제 화재 면적 (ha)')
    axes[0, 0].set_ylabel('예측 화재 면적 (ha)')
    axes[0, 0].set_title(f'현실적 모델 성능\nR² = {r2:.4f}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 잔차 vs 불확실성
    residuals = y_true - y_pred
    axes[0, 1].scatter(uncertainty, np.abs(residuals), alpha=0.6)
    axes[0, 1].set_xlabel('예측 불확실성 (ha)')
    axes[0, 1].set_ylabel('절댓값 잔차 (ha)')
    axes[0, 1].set_title('불확실성 vs 오차')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 신뢰구간 커버리지
    coverage_1std = np.abs(residuals) <= uncertainty
    coverage_2std = np.abs(residuals) <= 2 * uncertainty
    
    coverage_data = [
        coverage_1std.sum() / len(coverage_1std),
        coverage_2std.sum() / len(coverage_2std)
    ]
    expected_coverage = [0.68, 0.95]
    
    x_pos = [0, 1]
    width = 0.35
    
    axes[0, 2].bar([x - width/2 for x in x_pos], coverage_data, width, 
                  label='실제', alpha=0.7)
    axes[0, 2].bar([x + width/2 for x in x_pos], expected_coverage, width,
                  label='이론값', alpha=0.7)
    axes[0, 2].set_xlabel('신뢰구간')
    axes[0, 2].set_ylabel('커버리지 비율')
    axes[0, 2].set_title('신뢰구간 검증')
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(['1σ (68%)', '2σ (95%)'])
    axes[0, 2].legend()
    
    # 4. 개별 모델 성능
    if predictions_individual.shape[0] > 1:
        model_names = ['RF', 'GB', 'BR'][:predictions_individual.shape[0]]
        model_r2s = []
        
        for i in range(predictions_individual.shape[0]):
            r2_i = r2_score(y_true, predictions_individual[i])
            model_r2s.append(r2_i)
        
        bars = axes[1, 0].bar(model_names, model_r2s)
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].set_title('개별 모델 성능')
        
        # 앙상블 성능 표시
        axes[1, 0].axhline(y=r2, color='red', linestyle='--', 
                          label=f'앙상블: {r2:.3f}')
        axes[1, 0].legend()
    
    # 5. 화재 규모별 성능
    q33, q67 = np.percentile(y_true, [33, 67])
    
    small_mask = y_true <= q33
    medium_mask = (y_true > q33) & (y_true <= q67)
    large_mask = y_true > q67
    
    categories = []
    r2_by_size = []
    
    if small_mask.sum() > 5:
        small_r2 = r2_score(y_true[small_mask], y_pred[small_mask])
        categories.append('소형')
        r2_by_size.append(small_r2)
    
    if medium_mask.sum() > 5:
        medium_r2 = r2_score(y_true[medium_mask], y_pred[medium_mask])
        categories.append('중형')
        r2_by_size.append(medium_r2)
    
    if large_mask.sum() > 5:
        large_r2 = r2_score(y_true[large_mask], y_pred[large_mask])
        categories.append('대형')
        r2_by_size.append(large_r2)
    
    if categories:
        colors = ['lightblue', 'lightgreen', 'lightcoral'][:len(categories)]
        bars = axes[1, 1].bar(categories, r2_by_size, color=colors)
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].set_title('화재 규모별 성능')
        
        # 값 표시
        for bar, r2_val in zip(bars, r2_by_size):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, 
                           bar.get_height() + 0.01,
                           f'{r2_val:.3f}', ha='center', va='bottom')
    
    # 6. 예측 정확도 분포
    relative_error = np.abs(residuals) / y_true
    relative_error = np.clip(relative_error, 0, 2)  # 200% 상한
    
    axes[1, 2].hist(relative_error * 100, bins=30, alpha=0.7, 
                   color='orange', edgecolor='black')
    axes[1, 2].axvline(x=50, color='red', linestyle='--', 
                      label='50% 기준선')
    axes[1, 2].set_xlabel('상대 오차 (%)')
    axes[1, 2].set_ylabel('빈도')
    axes[1, 2].set_title('예측 정확도 분포')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('realistic_advanced_model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 시각화 저장: realistic_advanced_model_performance.png")

def main():
    """메인 함수"""
    print("🎯 현실적인 고급 화재 예측 모델 개발")
    print("=" * 70)
    
    try:
        # 1. 데이터 준비
        fire_data, base_features = load_and_prepare_data()
        print(f"✅ 기본 데이터: {fire_data.shape}")
        
        # 2. 보수적 피처 엔지니어링
        fire_data_enhanced, all_features = conservative_feature_engineering(
            fire_data, base_features
        )
        
        # 3. 피처 준비
        target_col = 'fire_area'
        available_features = [col for col in all_features 
                            if col in fire_data_enhanced.columns 
                            and col != target_col]
        
        X = fire_data_enhanced[available_features].copy()
        y = fire_data_enhanced[target_col].copy()
        
        # 4. 데이터 정리
        print(f"\n🧹 데이터 정리...")
        
        # 결측치 처리
        for col in X.columns:
            if X[col].isna().sum() > 0:
                X[col] = X[col].fillna(X[col].median())
        
        # 이상치 제거 (상위 2% 만)
        q98 = y.quantile(0.98)
        mask = y <= q98
        X, y = X[mask], y[mask]
        fire_data_filtered = fire_data_enhanced[mask]
        
        print(f"✅ 정리된 데이터: {X.shape[0]}개 샘플, {X.shape[1]}개 피처")
        
        # 5. 지능적 피처 선택
        selected_features = intelligent_feature_selection(X, y, max_features=40)
        
        # 6. 시간적 데이터 분할
        train_data, test_data = create_temporal_split(fire_data_filtered)
        
        X_train = train_data[selected_features]
        y_train = train_data[target_col]
        X_test = test_data[selected_features]  
        y_test = test_data[target_col]
        
        print(f"   훈련: {X_train.shape[0]}개, 테스트: {X_test.shape[0]}개")
        
        # 7. 견고한 앙상블 학습
        models, scaler = create_robust_ensemble(X_train, y_train, selected_features)
        
        # 8. 예측 (불확실성 포함)
        print("\n🎯 예측 및 평가...")
        
        y_pred_train, uncertainty_train, individual_train = ensemble_predict_with_uncertainty(
            X_train, models, scaler, selected_features
        )
        y_pred_test, uncertainty_test, individual_test = ensemble_predict_with_uncertainty(
            X_test, models, scaler, selected_features
        )
        
        # 9. 종합 평가
        print("\n" + "=" * 70)
        print("🏆 현실적 고급 모델 성능 보고서")
        print("=" * 70)
        
        print("\n📊 훈련 성능:")
        train_metrics = comprehensive_evaluation_with_confidence(
            y_train, y_pred_train, uncertainty_train
        )
        
        print("\n📊 테스트 성능:")
        test_metrics = comprehensive_evaluation_with_confidence(
            y_test, y_pred_test, uncertainty_test
        )
        
        # 10. 시각화
        create_realistic_visualizations(
            y_test, y_pred_test, uncertainty_test, individual_test
        )
        
        # 11. 모델 저장
        print("\n💾 모델 저장...")
        
        model_package = {
            'models': models,
            'scaler': scaler,
            'selected_features': selected_features,
            'test_metrics': test_metrics
        }
        
        joblib.dump(model_package, 'realistic_advanced_model.joblib')
        
        config = {
            'model_type': 'Realistic Advanced Ensemble',
            'n_models': len(models),
            'n_features': len(selected_features),
            'test_r2': test_metrics['r2'],
            'test_mae': test_metrics['mae'],
            'uncertainty': test_metrics['uncertainty'],
            'features': selected_features
        }
        
        with open('realistic_advanced_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✅ 저장 완료!")
        
        # 12. 최종 요약
        print(f"\n🎉 현실적 고급 모델 완료!")
        print(f"   • 테스트 R²: {test_metrics['r2']:.4f}")
        print(f"   • RMSE: {test_metrics['rmse']:.4f} ha")
        print(f"   • MAE: {test_metrics['mae']:.4f} ha")
        print(f"   • 평균 불확실성: ±{test_metrics['uncertainty']:.4f} ha")
        print(f"   • 실용적 정확도: {test_metrics['practical_accuracy']:.1%}")
        print(f"   • 선택된 피처: {len(selected_features)}개")
        
        # 성능 등급
        r2 = test_metrics['r2']
        if r2 > 0.8:
            grade = "🟢 최고 등급"
        elif r2 > 0.7:
            grade = "🟢 우수"
        elif r2 > 0.6:
            grade = "🟡 양호"
        elif r2 > 0.5:
            grade = "🟡 보통"
        else:
            grade = "🟠 개선 필요"
        
        print(f"   • 성능 등급: {grade}")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()