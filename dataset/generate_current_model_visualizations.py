import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import RobustScaler
import joblib
import json
from model_definitions import EnsembleRegressor, EnsembleClassifier
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_and_prepare_data():
    """현재 모델에서 사용하는 데이터 로드 및 전처리"""
    print("📊 데이터 로딩 및 전처리...")
    
    # train_v2_model.py에서 사용하는 데이터 로드
    try:
        df = pd.read_csv("gangwon_fire_data_augmented_parallel.csv")
        print(f"   - 원본 데이터 크기: {df.shape}")
    except FileNotFoundError:
        print("⚠️  gangwon_fire_data_augmented_parallel.csv 파일이 없어서 대체 데이터 사용")
        df = pd.read_csv("final_merged_feature_engineered.csv")
        print(f"   - 대체 데이터 크기: {df.shape}")
    
    # train_v2_model.py와 동일한 피처 설정
    features = [
        'lat', 'lng',
        'duration_hours', 'total_duration_hours',
        'T2M', 'RH2M', 'WS10M', 'WD10M', 'PRECTOTCORR',
        'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI'
    ]
    
    # 타겟 변수 선택 (available한 것 사용)
    target_candidates = ['estimated_damage_area', 'fire_area']
    target = None
    for candidate in target_candidates:
        if candidate in df.columns:
            target = candidate
            break
    
    if target is None:
        raise ValueError("타겟 변수를 찾을 수 없습니다.")
    
    print(f"   - 사용할 타겟 변수: {target}")
    print(f"   - 사용할 피처 수: {len(features)}")
    
    # 결측값 제거 및 이상치 처리 (train_v2_model.py와 동일)
    initial_size = len(df)
    df.dropna(subset=features + [target], inplace=True)
    df = df[(df[target] > 0) & (df[target] < df[target].quantile(0.99))]
    print(f"   - 전처리 후 데이터 크기: {df.shape} (제거된 데이터: {initial_size - len(df)})")
    
    X = df[features]
    y = df[target]  # 로그 변환은 나중에 적용
    
    return X, y, features, target

def create_model_comparison_plots(X, y, features, target):
    """현재 모델들의 성능 비교 시각화"""
    print("📈 모델 성능 비교 시각화 생성...")
    
    # 로그 변환 적용
    y_log = np.log1p(y)
    
    # 스케일링
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_log, test_size=0.2, random_state=42
    )
    
    # 현재 사용 중인 모델들 학습
    print("   - 모델 학습 중...")
    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse')
    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    gb_model = GradientBoostingRegressor(random_state=42)
    
    xgb_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    
    # 예측
    pred_xgb_log = xgb_model.predict(X_test)
    pred_rf_log = rf_model.predict(X_test)
    pred_gb_log = gb_model.predict(X_test)
    pred_ensemble_log = (pred_xgb_log + pred_rf_log + pred_gb_log) / 3
    
    # 원래 스케일로 변환
    y_test_orig = np.expm1(y_test)
    pred_xgb = np.expm1(pred_xgb_log)
    pred_rf = np.expm1(pred_rf_log)
    pred_gb = np.expm1(pred_gb_log)
    pred_ensemble = np.expm1(pred_ensemble_log)
    
    # 1. 실제 vs 예측값 산점도 (현재 모델들)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('현재 모델들의 실제 vs 예측 피해 면적 비교', fontsize=16, fontweight='bold')
    
    models = [
        ('XGBoost', pred_xgb, 'tab:blue'),
        ('Random Forest', pred_rf, 'tab:orange'), 
        ('Gradient Boosting', pred_gb, 'tab:green'),
        ('Ensemble (평균)', pred_ensemble, 'tab:red')
    ]
    
    for idx, (name, pred, color) in enumerate(models):
        ax = axes[idx // 2, idx % 2]
        
        # 산점도
        ax.scatter(y_test_orig, pred, alpha=0.6, color=color, s=30)
        
        # 이상적인 예측선
        max_val = max(y_test_orig.max(), pred.max())
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
        
        # 성능 지표 계산
        rmse = np.sqrt(mean_squared_error(y_test_orig, pred))
        mae = mean_absolute_error(y_test_orig, pred)
        r2 = r2_score(y_test_orig, pred)
        
        ax.set_xlabel('실제 피해 면적 (ha)')
        ax.set_ylabel('예측 피해 면적 (ha)')
        ax.set_title(f'{name}\nRMSE: {rmse:.1f}, MAE: {mae:.1f}, R²: {r2:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/mmymacymac/Developer/Projects/WildFire_projects/머신러닝 결과 보고서/current_models_actual_vs_predicted.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return xgb_model, rf_model, gb_model, X_train, y_train, scaler

def create_feature_importance_plot(xgb_model, rf_model, features):
    """피처 중요도 비교 시각화"""
    print("📊 피처 중요도 시각화 생성...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # XGBoost 피처 중요도
    xgb_importance = xgb_model.feature_importances_
    xgb_indices = np.argsort(xgb_importance)[::-1]
    
    axes[0].barh(range(len(features)), xgb_importance[xgb_indices], color='steelblue')
    axes[0].set_yticks(range(len(features)))
    axes[0].set_yticklabels([features[i] for i in xgb_indices])
    axes[0].set_xlabel('중요도')
    axes[0].set_title('XGBoost 피처 중요도')
    axes[0].grid(True, alpha=0.3)
    
    # Random Forest 피처 중요도
    rf_importance = rf_model.feature_importances_
    rf_indices = np.argsort(rf_importance)[::-1]
    
    axes[1].barh(range(len(features)), rf_importance[rf_indices], color='darkorange')
    axes[1].set_yticks(range(len(features)))
    axes[1].set_yticklabels([features[i] for i in rf_indices])
    axes[1].set_xlabel('중요도')
    axes[1].set_title('Random Forest 피처 중요도')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/mmymacymac/Developer/Projects/WildFire_projects/머신러닝 결과 보고서/current_models_feature_importance.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_kfold_performance_plot(X, y, features):
    """K-Fold 교차검증 성능 시각화"""
    print("🔄 K-Fold 교차검증 성능 시각화 생성...")
    
    y_log = np.log1p(y)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 결과 저장용
    results = {'Fold': [], 'Model': [], 'RMSE': [], 'R2': [], 'MAE': []}
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]
        
        # 모델들
        models = {
            'XGBoost': xgb.XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse'),
            'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(random_state=42)
        }
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            pred_log = model.predict(X_val)
            
            # 원래 스케일로 변환
            y_val_orig = np.expm1(y_val)
            pred_orig = np.expm1(pred_log)
            
            rmse = np.sqrt(mean_squared_error(y_val_orig, pred_orig))
            r2 = r2_score(y_val_orig, pred_orig)
            mae = mean_absolute_error(y_val_orig, pred_orig)
            
            results['Fold'].append(fold + 1)
            results['Model'].append(name)
            results['RMSE'].append(rmse)
            results['R2'].append(r2)
            results['MAE'].append(mae)
        
        # 앙상블 계산
        ensemble_pred_log = sum([models[name].predict(X_val) for name in models.keys()]) / len(models)
        ensemble_pred_orig = np.expm1(ensemble_pred_log)
        
        rmse_ens = np.sqrt(mean_squared_error(y_val_orig, ensemble_pred_orig))
        r2_ens = r2_score(y_val_orig, ensemble_pred_orig)
        mae_ens = mean_absolute_error(y_val_orig, ensemble_pred_orig)
        
        results['Fold'].append(fold + 1)
        results['Model'].append('Ensemble')
        results['RMSE'].append(rmse_ens)
        results['R2'].append(r2_ens)
        results['MAE'].append(mae_ens)
    
    # DataFrame으로 변환
    results_df = pd.DataFrame(results)
    
    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # RMSE 박스플롯
    sns.boxplot(data=results_df, x='Model', y='RMSE', ax=axes[0])
    axes[0].set_title('K-Fold 교차검증 RMSE 분포')
    axes[0].tick_params(axis='x', rotation=45)
    
    # R² 박스플롯  
    sns.boxplot(data=results_df, x='Model', y='R2', ax=axes[1])
    axes[1].set_title('K-Fold 교차검증 R² 분포')
    axes[1].tick_params(axis='x', rotation=45)
    
    # MAE 박스플롯
    sns.boxplot(data=results_df, x='Model', y='MAE', ax=axes[2])
    axes[2].set_title('K-Fold 교차검증 MAE 분포')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('/Users/mmymacymac/Developer/Projects/WildFire_projects/머신러닝 결과 보고서/current_models_kfold_performance.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 평균 성능 출력
    print("\n📊 K-Fold 평균 성능:")
    avg_performance = results_df.groupby('Model')[['RMSE', 'R2', 'MAE']].mean()
    for model in avg_performance.index:
        rmse, r2, mae = avg_performance.loc[model]
        print(f"  - {model:15s}: RMSE={rmse:.1f}, R²={r2:.3f}, MAE={mae:.1f}")

def create_target_distribution_plot(y, target):
    """타겟 변수 분포 시각화"""
    print("📊 타겟 변수 분포 시각화 생성...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 원본 분포
    axes[0].hist(y, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].set_xlabel(f'{target} (ha)')
    axes[0].set_ylabel('빈도')
    axes[0].set_title(f'원본 {target} 분포')
    axes[0].axvline(y.mean(), color='red', linestyle='--', label=f'평균: {y.mean():.1f}')
    axes[0].axvline(y.median(), color='orange', linestyle='--', label=f'중앙값: {y.median():.1f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 로그 변환 후 분포
    y_log = np.log1p(y)
    axes[1].hist(y_log, bins=50, alpha=0.7, color='darkorange', edgecolor='black')
    axes[1].set_xlabel(f'log1p({target})')
    axes[1].set_ylabel('빈도')
    axes[1].set_title(f'로그 변환 후 {target} 분포')
    axes[1].axvline(y_log.mean(), color='red', linestyle='--', label=f'평균: {y_log.mean():.2f}')
    axes[1].axvline(y_log.median(), color='orange', linestyle='--', label=f'중앙값: {y_log.median():.2f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/mmymacymac/Developer/Projects/WildFire_projects/머신러닝 결과 보고서/current_target_distribution.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 기술통계 출력
    print(f"\n📈 {target} 기술통계:")
    print(f"  - 평균: {y.mean():.1f} ha")
    print(f"  - 중앙값: {y.median():.1f} ha")  
    print(f"  - 표준편차: {y.std():.1f} ha")
    print(f"  - 최소값: {y.min():.1f} ha")
    print(f"  - 최대값: {y.max():.1f} ha")
    print(f"  - 75분위수: {y.quantile(0.75):.1f} ha")
    print(f"  - 95분위수: {y.quantile(0.95):.1f} ha")

def create_residual_analysis_plot(X, y, features):
    """잔차 분석 시각화"""
    print("📊 잔차 분석 시각화 생성...")
    
    y_log = np.log1p(y)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_log, test_size=0.2, random_state=42
    )
    
    # 앙상블 모델 학습
    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse')
    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    gb_model = GradientBoostingRegressor(random_state=42)
    
    xgb_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    
    # 앙상블 예측
    pred_log = (xgb_model.predict(X_test) + rf_model.predict(X_test) + gb_model.predict(X_test)) / 3
    
    y_test_orig = np.expm1(y_test)
    pred_orig = np.expm1(pred_log)
    
    residuals = y_test_orig - pred_orig
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('현재 앙상블 모델 잔차 분석', fontsize=16, fontweight='bold')
    
    # 1. 잔차 vs 예측값
    axes[0, 0].scatter(pred_orig, residuals, alpha=0.6, color='steelblue')
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('예측값 (ha)')
    axes[0, 0].set_ylabel('잔차 (ha)')
    axes[0, 0].set_title('잔차 vs 예측값')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 잔차 히스토그램
    axes[0, 1].hist(residuals, bins=30, alpha=0.7, color='darkorange', edgecolor='black')
    axes[0, 1].set_xlabel('잔차 (ha)')
    axes[0, 1].set_ylabel('빈도')
    axes[0, 1].set_title('잔차 분포')
    axes[0, 1].axvline(residuals.mean(), color='red', linestyle='--', label=f'평균: {residuals.mean():.1f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Q-Q 플롯 (정규성 검정)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (정규성 검정)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 실제값 vs 잔차
    axes[1, 1].scatter(y_test_orig, residuals, alpha=0.6, color='green')
    axes[1, 1].axhline(y=0, color='red', linestyle='--')
    axes[1, 1].set_xlabel('실제값 (ha)')
    axes[1, 1].set_ylabel('잔차 (ha)')
    axes[1, 1].set_title('실제값 vs 잔차')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/mmymacymac/Developer/Projects/WildFire_projects/머신러닝 결과 보고서/current_models_residual_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """메인 실행 함수"""
    print("🔥 현재 모델 기반 시각화 생성 시작!")
    print("=" * 50)
    
    try:
        # 1. 데이터 로드 및 전처리
        X, y, features, target = load_and_prepare_data()
        
        # 2. 모델 비교 시각화
        xgb_model, rf_model, gb_model, X_train, y_train, scaler = create_model_comparison_plots(X, y, features, target)
        
        # 3. 피처 중요도 시각화
        create_feature_importance_plot(xgb_model, rf_model, features)
        
        # 4. K-Fold 성능 시각화
        create_kfold_performance_plot(X, y, features)
        
        # 5. 타겟 분포 시각화
        create_target_distribution_plot(y, target)
        
        # 6. 잔차 분석 시각화
        create_residual_analysis_plot(X, y, features)
        
        print("\n" + "=" * 50)
        print("✅ 모든 시각화 생성 완료!")
        print("📁 저장 위치: /Users/mmymacymac/Developer/Projects/WildFire_projects/머신러닝 결과 보고서/")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()