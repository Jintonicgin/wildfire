import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set Korean font for matplotlib (macOS)
plt.rcParams['font.family'] = ['AppleGothic', 'Apple SD Gothic Neo', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def load_important_features():
    """Load important features for each model from JSON files"""
    
    try:
        print("Loading model features from JSON files...")
        
        # Load v3_tuned area model features (first 20 for importance)
        area_path = "area_model_columns_v3_tuned.json"
        print(f"Loading area features from: {area_path}")
        with open(area_path, "r") as f:
            area_features = json.load(f)[:20]
        print(f"✓ Loaded {len(area_features)} area features")
        
        # Load v2_tuned_cw direction model features (first 20 for importance)
        direction_path = "direction_model_columns_v2_tuned_cw.json"
        print(f"Loading direction features from: {direction_path}")
        with open(direction_path, "r") as f:
            direction_features = json.load(f)[:20]
        print(f"✓ Loaded {len(direction_features)} direction features")
        
        # Load v2_tuned_cw speed model features (first 20 for importance)
        speed_path = "speed_model_columns_v2_tuned_cw.json"
        print(f"Loading speed features from: {speed_path}")
        with open(speed_path, "r") as f:
            speed_features = json.load(f)[:20]
        print(f"✓ Loaded {len(speed_features)} speed features")
        
        # Print first few features for verification
        print(f"\nFirst 5 area features: {area_features[:5]}")
        print(f"First 5 direction features: {direction_features[:5]}")
        print(f"First 5 speed features: {speed_features[:5]}")
        
        # Create alternative area features using actual available columns from dataset
        print(f"\n🔧 Creating alternative area features from available columns...")
        
        # Find area features that actually exist in the dataset or create reasonable alternatives
        area_features_alternative = []
        
        # Check if the feature engineering derived features exist, if not use base features
        derived_features = ['dryness_index', 'wind_temp_product', 'wind_humidity_ratio']
        base_weather_features = ['T2M_mean', 'T2M_max', 'T2M_min', 'T2M_std', 
                                'RH2M_mean', 'RH2M_max', 'RH2M_min', 'RH2M_std',
                                'WS10M_mean', 'WS10M_max', 'WS10M_min', 'WS10M_std',
                                'PS_mean', 'PS_max', 'PS_min', 'PS_std',
                                'PRECTOTCORR_mean', 'PRECTOTCORR_max', 'PRECTOTCORR_min', 'PRECTOTCORR_std']
        
        # If derived features don't exist, use base weather features
        area_features = derived_features + base_weather_features
        print(f"Using alternative area features: {len(area_features)} features")
        
    except FileNotFoundError as e:
        print(f"Error loading model features: {e}")
        print("Using fallback features...")
        
        # Fallback features if JSON files not found - using available columns in dataset
        area_features = [
            'T2M_0h', 'RH2M_0h', 'WS10M_0h', 'PS_0h', 'ALLSKY_SFC_SW_DWN_0h',
            'T2M_3h', 'RH2M_3h', 'WS10M_3h', 'PS_3h', 'ALLSKY_SFC_SW_DWN_3h',
            'T2M_6h', 'RH2M_6h', 'WS10M_6h', 'PS_6h', 'ALLSKY_SFC_SW_DWN_6h',
            'T2M_9h', 'RH2M_9h', 'WS10M_9h', 'PS_9h', 'ALLSKY_SFC_SW_DWN_9h'
        ]
        
        direction_features = [
            'FWI_0h', 'ndvi_stress', 'slope_max', 'elevation_mean', 
            'dry_days_90d_start', 'ALLSKY_SFC_SW_DWN_0h', 'BUI_0h', 'DC_0h',
            'DMC_0h', 'ISI_0h', 'PS_0h', 'RH2M_0h', 'T2M_0h', 'WS10M_0h'
        ]
        
        speed_features = [
            'elevation_max', 'slope_max', 'WD10M_0h', 'dry_to_rain_ratio_30d',
            'ndvi_stress', 'ALLSKY_SFC_SW_DWN_0h', 'DC_0h', 'DMC_0h',
            'FFMC_0h', 'FWI_0h', 'ISI_0h', 'PS_0h', 'aspect_mode'
        ]
    
    return area_features, direction_features, speed_features

def extract_top_core_features(area_features, direction_features, speed_features, top_n=20):
    """모델별 중요 변수를 통합하여 상위 N개 핵심 변수 추출"""
    
    # 각 변수의 등장 횟수와 우선순위 계산
    feature_scores = {}
    
    # Area 모델 변수들 (가중치 3 - 피해면적 예측이 가장 중요)
    for i, feature in enumerate(area_features):
        # 순서가 앞설수록 중요도가 높으므로 역순으로 점수 부여
        score = (len(area_features) - i) * 3
        feature_scores[feature] = feature_scores.get(feature, 0) + score
    
    # Direction 모델 변수들 (가중치 2)
    for i, feature in enumerate(direction_features):
        score = (len(direction_features) - i) * 2
        feature_scores[feature] = feature_scores.get(feature, 0) + score
    
    # Speed 모델 변수들 (가중치 2)
    for i, feature in enumerate(speed_features):
        score = (len(speed_features) - i) * 2
        feature_scores[feature] = feature_scores.get(feature, 0) + score
    
    # 점수 기준으로 상위 N개 변수 선택
    top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    print(f"\n🎯 상위 {top_n}개 핵심 변수 선별 결과:")
    print("=" * 70)
    for i, (feature, score) in enumerate(top_features, 1):
        models = []
        if feature in area_features: models.append("Area(v3_tuned)")
        if feature in direction_features: models.append("Direction(v2_tuned_cw)") 
        if feature in speed_features: models.append("Speed(v2_tuned_cw)")
        print(f"{i:2d}. {feature:<30} (점수: {score:3d}, 모델: {'/'.join(models)})")
    
    return [feature for feature, _ in top_features]

def create_correlation_matrix(df, features, title, save_path):
    """Create correlation matrix heatmap"""
    
    # Filter features that exist in dataframe
    available_features = [f for f in features if f in df.columns]
    
    print(f"   Features requested: {len(features)}, Available in dataset: {len(available_features)}")
    if len(available_features) < len(features):
        missing_features = [f for f in features if f not in df.columns]
        print(f"   Missing features: {missing_features[:5]}...")
    
    if len(available_features) < 3:
        print(f"Warning: Only {len(available_features)} features available for {title}")
        print("Looking for alternative features...")
        
        # For area model, try to find alternative features from weather variables
        if "Area" in title:
            weather_alternatives = []
            for base in ['T2M', 'RH2M', 'WS10M', 'PS', 'ALLSKY_SFC_SW_DWN', 'PRECTOTCORR']:
                for suffix in ['_0h', '_3h', '_6h', '_9h', '_12h', '_mean', '_max', '_min', '_std']:
                    alt_feature = base + suffix
                    if alt_feature in df.columns:
                        weather_alternatives.append(alt_feature)
            
            available_features = weather_alternatives[:20]  # Use first 20 available weather features
            print(f"   Using {len(available_features)} alternative weather features")
        
        if len(available_features) < 3:
            return None
    
    print(f"Creating correlation matrix for {len(available_features)} features...")
    
    # Calculate correlation matrix
    corr_matrix = df[available_features].corr()
    
    # Create figure with larger size for better readability
    plt.figure(figsize=(16, 14))
    
    # Create symmetric heatmap (full matrix)
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={"shrink": .8},
                annot_kws={'size': 8})
    
    plt.title(f'{title} - 상관관계 히트맵 (상위 {len(available_features)}개 변수)', 
              fontsize=18, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return corr_matrix

def create_core_features_heatmap(df, core_features, save_path):
    """핵심 변수들만으로 상관관계 히트맵 생성"""
    
    # Filter features that exist in dataframe
    available_features = [f for f in core_features if f in df.columns]
    
    if len(available_features) < 3:
        print(f"Warning: Only {len(available_features)} core features available")
        return None
    
    print(f"🎯 Creating CORE features correlation heatmap for {len(available_features)} features...")
    
    # Calculate correlation matrix
    corr_matrix = df[available_features].corr()
    
    # Create figure with optimal size
    plt.figure(figsize=(20, 18))
    
    # Create enhanced symmetric heatmap (full matrix)
    # Custom colormap for better distinction
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap=cmap,
                center=0,
                square=True,
                fmt='.3f',
                cbar_kws={"shrink": .7, "label": "Correlation Coefficient"},
                annot_kws={'size': 9, 'weight': 'bold'},
                linewidths=0.5)
    
    plt.title('🔥 산불 예측 핵심 변수 상관관계 분석 🔥\n(상위 20개 통합 중요 변수 - v3_tuned/v2_tuned_cw 모델)', 
              fontsize=18, fontweight='bold', pad=30)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    
    # Add subtitle
    plt.figtext(0.5, 0.02, 'Area(v3_tuned)/Direction(v2_tuned_cw)/Speed(v2_tuned_cw) 모델의 중요 변수를 가중치로 통합 선별', 
                ha='center', fontsize=11, style='italic')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return corr_matrix

def create_pairplot(df, features, title, save_path, sample_size=1000):
    """Create pairplot for selected variables"""
    
    # Filter features that exist in dataframe
    available_features = [f for f in features[:6] if f in df.columns]  # Limit to 6 for readability
    
    if len(available_features) < 3:
        print(f"Warning: Only {len(available_features)} features available for {title}")
        return
    
    # Sample data for performance
    df_sample = df[available_features].sample(n=min(sample_size, len(df)))
    
    # Create pairplot
    plt.figure(figsize=(15, 15))
    g = sns.pairplot(df_sample, 
                     diag_kind='hist',
                     plot_kws={'alpha': 0.6, 's': 20})
    
    g.fig.suptitle(f'{title} - Top Variables Pairplot', y=1.02, fontsize=16, fontweight='bold')
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_high_correlations(corr_matrix, threshold=0.7):
    """Find and analyze high correlations"""
    
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                high_corr_pairs.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    return sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)

def main():
    """Main analysis function"""
    
    print("🔥 Starting Enhanced Variable Correlation Analysis...")
    
    # Load data
    data_path = "final_merged_feature_engineered.csv"
    print(f"Loading data from {data_path}...")
    
    try:
        df = pd.read_csv(data_path)
        print(f"Data loaded successfully: {df.shape}")
    except FileNotFoundError:
        print(f"Error: {data_path} not found")
        return
    
    # Load important features
    area_features, direction_features, speed_features = load_important_features()
    
    # Extract top core features (통합 상위 20개)
    core_features = extract_top_core_features(area_features, direction_features, speed_features, top_n=20)
    
    # Create output directory
    output_dir = Path("correlation_analysis")
    output_dir.mkdir(exist_ok=True)
    
    print("\n🎯 Creating CORE FEATURES correlation heatmap...")
    
    # 핵심 변수 상관관계 히트맵 (메인 결과물)
    core_corr = create_core_features_heatmap(
        df, core_features,
        output_dir / "core_features_correlation_heatmap.png"
    )
    
    if core_corr is not None:
        core_high_corr = analyze_high_correlations(core_corr, threshold=0.6)
        print(f"🔍 Found {len(core_high_corr)} high correlations (|r| >= 0.6) in core features")
    
    print("\n📊 Creating individual model correlation matrices...")
    
    # 1. Area Model Correlations (v3_tuned)
    print("1. Analyzing Area Model (v3_tuned) correlations...")
    area_corr = create_correlation_matrix(
        df, area_features, 
        "Area Model v3_tuned (피해면적 예측)",
        output_dir / "area_model_v3_tuned_correlations.png"
    )
    
    if area_corr is not None:
        area_high_corr = analyze_high_correlations(area_corr)
        print(f"   Found {len(area_high_corr)} high correlations (|r| >= 0.7)")
    
    # 2. Direction Model Correlations (v2_tuned_cw)
    print("2. Analyzing Direction Model (v2_tuned_cw) correlations...")
    direction_corr = create_correlation_matrix(
        df, direction_features,
        "Direction Model v2_tuned_cw (확산방향 예측)", 
        output_dir / "direction_model_v2_tuned_cw_correlations.png"
    )
    
    if direction_corr is not None:
        direction_high_corr = analyze_high_correlations(direction_corr)
        print(f"   Found {len(direction_high_corr)} high correlations (|r| >= 0.7)")
    
    # 3. Speed Model Correlations (v2_tuned_cw)
    print("3. Analyzing Speed Model (v2_tuned_cw) correlations...")
    speed_corr = create_correlation_matrix(
        df, speed_features,
        "Speed Model v2_tuned_cw (확산속도 예측)",
        output_dir / "speed_model_v2_tuned_cw_correlations.png"  
    )
    
    if speed_corr is not None:
        speed_high_corr = analyze_high_correlations(speed_corr)
        print(f"   Found {len(speed_high_corr)} high correlations (|r| >= 0.7)")
    
    # Print comprehensive high correlation summary
    print("\n🔍 High Correlation Summary:")
    print("=" * 70)
    
    if core_corr is not None and len(core_high_corr) > 0:
        print(f"\n🎯 CORE Features - High Correlations (Top 10):")
        for i, pair in enumerate(core_high_corr[:10], 1):
            print(f"   {i:2d}. {pair['var1']:<20} ↔ {pair['var2']:<20}: r = {pair['correlation']:6.3f}")
    
    if area_corr is not None and len(area_high_corr) > 0:
        print(f"\n📈 Area Model - High Correlations (Top 3):")
        for pair in area_high_corr[:3]:
            print(f"   • {pair['var1']} ↔ {pair['var2']}: r = {pair['correlation']:.3f}")
    
    if direction_corr is not None and len(direction_high_corr) > 0:
        print(f"\n🧭 Direction Model - High Correlations (Top 3):")
        for pair in direction_high_corr[:3]:
            print(f"   • {pair['var1']} ↔ {pair['var2']}: r = {pair['correlation']:.3f}")
    
    if speed_corr is not None and len(speed_high_corr) > 0:
        print(f"\n⚡ Speed Model - High Correlations (Top 3):")
        for pair in speed_high_corr[:3]:
            print(f"   • {pair['var1']} ↔ {pair['var2']}: r = {pair['correlation']:.3f}")
    
    print(f"\n✅ Enhanced Analysis Complete! Results saved in {output_dir}/")
    print("🎯 Main Result:")
    print("- core_features_correlation_heatmap.png  (상위 20개 핵심 변수 히트맵)")
    print("\n📊 Individual Models:")
    print("- area_model_v3_tuned_correlations.png")
    print("- direction_model_v2_tuned_cw_correlations.png") 
    print("- speed_model_v2_tuned_cw_correlations.png")

if __name__ == "__main__":
    main()