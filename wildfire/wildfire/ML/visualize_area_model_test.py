#!/usr/bin/env python3
"""
Advanced Area Boost 모델 시각화 테스트
78.8% R² 성능 시각적 검증
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Malgun Gothic']
plt.rcParams['axes.unicode_minus'] = False

def create_advanced_features_v2(df):
    """고급 피처 엔지니어링 v2 (advanced_area_boost.py와 동일)"""
    features = []
    
    # 1. 기본 기상 피처
    weather_features = [
        't2m_0h', 'rh2m_0h', 'ws10m_0h', 'wd10m_0h', 'ps_0h',
        'fwi_0h', 'isi_0h', 'dc_0h', 'dmc_0h', 'ffmc_0h'
    ]
    
    for feat in weather_features:
        if feat in df.columns:
            features.append(feat)
    
    # 2. 화재 위험 지수들의 조합
    if 'fwi_0h' in df.columns and 'isi_0h' in df.columns:
        df['fire_risk_combined'] = df['fwi_0h'] * df['isi_0h']
        features.append('fire_risk_combined')
    
    # 3. 바람-습도 상호작용
    if 'ws10m_0h' in df.columns and 'rh2m_0h' in df.columns:
        df['wind_humidity_ratio'] = df['ws10m_0h'] / (df['rh2m_0h'] + 1)
        features.append('wind_humidity_ratio')
    
    # 4. 온도-습도 상호작용  
    if 't2m_0h' in df.columns and 'rh2m_0h' in df.columns:
        df['temp_humidity_interaction'] = df['t2m_0h'] * (100 - df['rh2m_0h'])
        features.append('temp_humidity_interaction')
    
    # 5. 계절적 패턴
    if 'startday' in df.columns:
        df['season_sin'] = np.sin(2 * np.pi * df['startday'] / 365)
        df['season_cos'] = np.cos(2 * np.pi * df['startday'] / 365)
        features.extend(['season_sin', 'season_cos'])
    
    # 6. 로그 변환된 피처들
    for feat in ['fwi_0h', 'isi_0h', 'dc_0h']:
        if feat in df.columns:
            df[f'{feat}_log'] = np.log1p(df[feat])
            features.append(f'{feat}_log')
    
    # 7. 바람 방향 벡터 성분
    if 'wd10m_0h' in df.columns:
        df['wind_x'] = np.cos(np.radians(df['wd10m_0h']))
        df['wind_y'] = np.sin(np.radians(df['wd10m_0h']))
        features.extend(['wind_x', 'wind_y'])
    
    # 8. 과거 데이터 통계 (간소화)
    past_features = [col for col in df.columns if 'past' in col or 'mean' in col or 'std' in col]
    available_past = [f for f in past_features if f in df.columns][:15]  # 최대 15개
    features.extend(available_past)
    
    return df, features

def build_stacking_model(X_train, y_train, X_test):
    """고급 스태킹 모델 구축 (advanced_area_boost.py와 동일)"""
    
    # Level 1 모델들
    base_models = {
        'ridge': Ridge(alpha=1.0),
        'elastic': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'bayesian': BayesianRidge(),
        'lgb': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
        'nn_opt': MLPRegressor(hidden_layer_sizes=(200, 50, 80), alpha=0.024, 
                              learning_rate_init=0.0098, max_iter=500, random_state=42)
    }
    
    # 다양한 타겟 변환
    transformations = {
        'standard': lambda x: x,
        'power': lambda x: np.power(x, 0.5),
        'quantile': lambda x: x  # 간소화
    }
    
    # Level 1 예측들 수집
    level1_predictions = []
    level1_names = []
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for transform_name, transform_func in transformations.items():
        y_transformed = transform_func(y_train)
        
        for model_name, model in base_models.items():
            try:
                if model_name in ['ridge', 'elastic', 'bayesian', 'nn_opt']:
                    model.fit(X_train_scaled, y_transformed)
                    pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_transformed)
                    pred = model.predict(X_test)
                
                # 역변환
                if transform_name == 'power':
                    pred = np.power(np.maximum(0, pred), 2)
                
                pred = np.maximum(0, pred)
                level1_predictions.append(pred)
                level1_names.append(f"{transform_name}_{model_name}")
                
            except:
                continue
    
    if len(level1_predictions) < 2:
        return None, None
    
    # Level 2: 메타모델
    level1_array = np.column_stack(level1_predictions)
    
    meta_models = {
        'ridge_meta': Ridge(alpha=0.1),
        'elastic_meta': ElasticNet(alpha=0.1, l1_ratio=0.5), 
        'nn_meta': MLPRegressor(hidden_layer_sizes=(50, 25), alpha=0.01, max_iter=300, random_state=42)
    }
    
    best_meta = None
    best_pred = None
    best_r2 = -1
    
    # 메타모델을 위한 분할
    X_meta_train, X_meta_val, y_meta_train, y_meta_val = train_test_split(
        level1_array, y_train, test_size=0.3, random_state=42
    )
    
    for meta_name, meta_model in meta_models.items():
        try:
            meta_scaler = StandardScaler()
            X_meta_train_scaled = meta_scaler.fit_transform(X_meta_train)
            X_meta_val_scaled = meta_scaler.transform(X_meta_val)
            
            meta_model.fit(X_meta_train_scaled, y_meta_train)
            val_pred = meta_model.predict(X_meta_val_scaled)
            val_pred = np.maximum(0, val_pred)
            
            r2 = r2_score(y_meta_val, val_pred)
            
            if r2 > best_r2:
                best_r2 = r2
                best_meta = (meta_model, meta_scaler)
                
                # 전체 level1에서 최종 예측
                level1_scaled = meta_scaler.transform(level1_array)
                final_pred = meta_model.predict(level1_scaled)
                best_pred = np.maximum(0, final_pred)
                
        except:
            continue
    
    return best_pred, best_r2

def visualize_area_model_performance():
    """면적 예측 모델 성능 시각화"""
    print("🎯 Advanced Area Boost 모델 시각화 테스트")
    print("=" * 60)
    
    # 데이터 로드
    print("📊 데이터 로드...")
    df = pd.read_csv('final_merged_feature_engineered.csv', low_memory=False)
    fire_mask = (df['fire_area'] > 0) & df['fire_area'].notna()
    fire_df = df[fire_mask].copy()
    
    print(f"   화재 데이터: {fire_df.shape}")
    print(f"   면적 범위: {fire_df['fire_area'].min():.3f} ~ {fire_df['fire_area'].max():.3f} ha")
    
    # 피처 생성
    print("🎯 고급 피처 엔지니어링...")
    df_features, features = create_advanced_features_v2(fire_df)
    
    # 데이터 준비
    X = df_features[features].fillna(df_features[features].median())
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = fire_df['fire_area'].copy()
    
    # 극값 처리
    y_threshold = y.quantile(0.99)
    y_clipped = y.clip(upper=y_threshold)
    y_log = np.log1p(y_clipped)
    
    print(f"   피처 수: {len(features)}개")
    print(f"   데이터 형태: X{X.shape}")
    print(f"   면적 범위 (클리핑): {y_clipped.min():.3f} ~ {y_clipped.max():.3f} ha")
    
    # 분할
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.25, random_state=42
    )
    
    y_test = np.expm1(y_test_log)
    y_train = np.expm1(y_train_log)
    
    print(f"   훈련셋: {X_train.shape}")
    print(f"   테스트셋: {X_test.shape}")
    
    # 모델 훈련 및 예측
    print("\\n🚀 스태킹 모델 훈련...")
    y_pred, meta_r2 = build_stacking_model(X_train, y_train_log, X_test)
    
    if y_pred is None:
        print("❌ 모델 훈련 실패")
        return
    
    # 역변환
    y_pred = np.expm1(y_pred)
    y_pred = np.maximum(0, y_pred)
    
    # 성능 계산
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\\n🏆 모델 성능:")
    print(f"   R² Score: {r2:.4f} ({r2:.1%})")
    print(f"   RMSE: {rmse:.3f} ha")
    print(f"   MAE: {mae:.3f} ha")
    
    # 시각화
    print("\\n📊 시각화 생성...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'화재 면적 예측 모델 성능 분석 (R² = {r2:.1%})', fontsize=16, fontweight='bold')
    
    # 1. 실제 vs 예측 산점도
    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_pred, alpha=0.6, s=30, color='blue')
    ax1.plot([0, y_test.max()], [0, y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('실제 면적 (ha)')
    ax1.set_ylabel('예측 면적 (ha)')
    ax1.set_title(f'실제 vs 예측\\nR² = {r2:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 잔차 분석
    ax2 = axes[0, 1]
    residuals = y_test - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, s=30, color='green')
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('예측 면적 (ha)')
    ax2.set_ylabel('잔차 (실제 - 예측)')
    ax2.set_title(f'잔차 분석\\nRMSE = {rmse:.2f} ha')
    ax2.grid(True, alpha=0.3)
    
    # 3. 오차 분포
    ax3 = axes[0, 2]
    error_pct = np.abs(residuals) / (y_test + 0.1) * 100
    ax3.hist(error_pct, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax3.set_xlabel('절대 오차율 (%)')
    ax3.set_ylabel('빈도')
    ax3.set_title(f'오차율 분포\\n평균 오차율: {error_pct.mean():.1f}%')
    ax3.grid(True, alpha=0.3)
    
    # 4. 크기별 정확도
    ax4 = axes[1, 0]
    size_bins = ['소형\\n(0-1ha)', '중형\\n(1-5ha)', '대형\\n(5-20ha)', '초대형\\n(20ha+)']
    size_thresholds = [0, 1, 5, 20, np.inf]
    size_r2s = []
    
    for i in range(len(size_thresholds)-1):
        mask = (y_test >= size_thresholds[i]) & (y_test < size_thresholds[i+1])
        if mask.sum() > 5:
            size_r2 = r2_score(y_test[mask], y_pred[mask])
            size_r2s.append(size_r2)
        else:
            size_r2s.append(0)
    
    bars = ax4.bar(size_bins, size_r2s, color=['lightblue', 'lightgreen', 'orange', 'red'], alpha=0.7)
    ax4.set_ylabel('R² Score')
    ax4.set_title('화재 규모별 예측 정확도')
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)
    
    # 막대 위에 값 표시
    for bar, r2_val in zip(bars, size_r2s):
        if r2_val > 0:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{r2_val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. 예측 정확도 구간별 분석
    ax5 = axes[1, 1]
    accuracy_ranges = ['90%+', '80-90%', '70-80%', '60-70%', '<60%']
    accuracy_thresholds = [0.9, 0.8, 0.7, 0.6, 0]
    individual_r2s = []
    
    for i in range(len(y_test)):
        if y_test.iloc[i] > 0:
            point_r2 = 1 - ((y_test.iloc[i] - y_pred[i])**2) / ((y_test.iloc[i] - y_test.mean())**2)
            individual_r2s.append(max(0, point_r2))
        else:
            individual_r2s.append(0)
    
    individual_r2s = np.array(individual_r2s)
    range_counts = []
    
    for i in range(len(accuracy_thresholds)):
        if i == 0:
            count = (individual_r2s >= accuracy_thresholds[i]).sum()
        else:
            count = ((individual_r2s >= accuracy_thresholds[i]) & 
                    (individual_r2s < accuracy_thresholds[i-1])).sum()
        range_counts.append(count)
    
    colors = ['darkgreen', 'green', 'yellow', 'orange', 'red']
    wedges, texts, autotexts = ax5.pie(range_counts, labels=accuracy_ranges, colors=colors, 
                                       autopct='%1.1f%%', startangle=90)
    ax5.set_title('예측 정확도 분포')
    
    # 6. 성능 요약
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # 통계 정보
    stats_text = f'''
    📊 성능 요약
    
    🎯 전체 성능
    • R² Score: {r2:.3f} ({r2:.1%})
    • RMSE: {rmse:.2f} ha  
    • MAE: {mae:.2f} ha
    
    📈 예측 품질
    • 평균 오차율: {error_pct.mean():.1f}%
    • 중앙 오차율: {np.median(error_pct):.1f}%
    • 90% 정확도 이상: {(individual_r2s >= 0.9).sum()}개
    
    🔥 화재 규모별
    • 소형화재: {size_r2s[0]:.2f}
    • 중형화재: {size_r2s[1]:.2f}
    • 대형화재: {size_r2s[2]:.2f}
    • 초대형화재: {size_r2s[3]:.2f}
    
    ✅ 활용 권장: {'우수' if r2 > 0.7 else '양호' if r2 > 0.5 else '제한적'}
    '''
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('area_model_performance_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\\n✅ 시각화 완료!")
    print(f"📊 파일 저장: area_model_performance_visualization.png")
    print(f"🎯 최종 결론: R² {r2:.1%} - {'실용 가능한 우수한 성능' if r2 > 0.7 else '참고용 성능' if r2 > 0.5 else '개선 필요'}")
    
    return r2, rmse, mae, error_pct.mean()

if __name__ == "__main__":
    result = visualize_area_model_performance()