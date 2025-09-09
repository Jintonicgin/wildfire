import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend and font
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'DejaVu Sans'

def create_optimized_variable_set():
    """Create optimized variable set after removing redundant terrain variables"""
    
    print("🔧 Creating Optimized Variable Set...")
    
    # Load data
    df = pd.read_csv("final_merged_feature_engineered.csv")
    print(f"Original data: {df.shape}")
    
    # Optimized variable set (removing only redundant terrain variables)
    optimized_vars = [
        # Target variable
        'fire_area',
        
        # All FWI indices (keep all - domain knowledge important)
        'FWI_0h', 'ISI_0h', 'DC_0h', 'DMC_0h', 'BUI_0h', 'FFMC_0h',
        
        # Key climate variables
        'T2M_max', 'T2M_min', 'T2M_std',
        'RH2M_std', 'PS_max', 'PS_min',
        'WS10M_max', 'WS10M_min', 'WS10M_std',
        'WD10M_0h',
        
        # Vegetation stress
        'ndvi_stress',
        
        # Terrain (keep max values only, remove means)
        'elevation_max',  # Remove elevation_mean (r=0.999)
        'slope_max',      # Remove slope_mean (r=0.924)
        
        # Derived features
        'dryness_index', 'wind_temp_product',
        'dry_days_90d_start', 'dry_to_rain_ratio_30d',
        
        # Change variables (key temporal patterns)
        't2m_change_0_3h', 'rh2m_change_3_6h', 'ws10m_change_3_6h'
    ]
    
    # Filter available variables
    available_vars = [v for v in optimized_vars if v in df.columns]
    
    # Show what was removed vs kept
    removed_vars = ['elevation_mean', 'slope_mean']
    kept_terrain = ['elevation_max', 'slope_max'] 
    all_fwi = ['FWI_0h', 'ISI_0h', 'DC_0h', 'DMC_0h', 'BUI_0h', 'FFMC_0h']
    
    print(f"\n📊 Variable Selection Summary:")
    print(f"   Total optimized variables: {len(available_vars)}")
    print(f"   🚫 Removed (high correlation): {removed_vars}")
    print(f"   ✅ Kept terrain variables: {kept_terrain}")
    print(f"   ✅ All FWI indices preserved: {len([v for v in all_fwi if v in available_vars])}/{len(all_fwi)}")
    
    return df, available_vars

def analyze_optimized_correlations(df, variables):
    """Analyze correlations after optimization"""
    
    print(f"\n🔍 Analyzing Optimized Correlations...")
    
    # Select numeric data
    numeric_data = df[variables].select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numeric_data.corr()
    
    # Find remaining high correlations
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= 0.7:
                high_corr_pairs.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    # Display results
    print(f"\n🚨 Remaining High Correlations (|r| >= 0.7):")
    if high_corr_pairs:
        high_corr_pairs = sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)
        for idx, pair in enumerate(high_corr_pairs, 1):
            correlation_type = "🔥 FWI-related" if any(fwi in pair['var1'] or fwi in pair['var2'] 
                                                      for fwi in ['FWI', 'ISI', 'DC', 'DMC', 'BUI', 'FFMC']) else "🌍 Other"
            print(f"{idx:2d}. {pair['var1']} ↔ {pair['var2']}: r = {pair['correlation']:.3f} {correlation_type}")
    else:
        print("   ✅ No high correlations found!")
    
    # Medium correlations
    medium_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if 0.5 <= abs(corr_val) < 0.7:
                medium_corr_pairs.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    print(f"\n📊 Medium Correlations (0.5 <= |r| < 0.7): {len(medium_corr_pairs)}")
    if medium_corr_pairs and len(medium_corr_pairs) <= 10:
        medium_corr_pairs = sorted(medium_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)
        for idx, pair in enumerate(medium_corr_pairs, 1):
            print(f"{idx:2d}. {pair['var1']} ↔ {pair['var2']}: r = {pair['correlation']:.3f}")
    
    return corr_matrix, high_corr_pairs

def create_final_heatmap(df, variables, corr_matrix):
    """Create final optimized correlation heatmap"""
    
    print(f"\n🎨 Creating Final Correlation Heatmap...")
    
    # Create output directory
    output_dir = Path("correlation_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(16, 14))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Enhanced heatmap with better styling
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={"shrink": .8},
                linewidths=0.3,
                annot_kws={'size': 8})
    
    plt.title('🔥 Optimized Variable Correlation Matrix\n(FWI Indices Preserved, Redundant Terrain Variables Removed)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save final plot
    final_path = output_dir / "final_optimized_correlation_heatmap.png"
    plt.savefig(final_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Final heatmap saved: {final_path}")
    
    return final_path

def create_variable_guide(variables, high_corr_pairs):
    """Create variable selection guideline"""
    
    print(f"\n📋 Creating Variable Selection Guide...")
    
    # Categorize variables
    fwi_vars = [v for v in variables if any(fwi in v for fwi in ['FWI', 'ISI', 'DC', 'DMC', 'BUI', 'FFMC'])]
    climate_vars = [v for v in variables if any(climate in v for climate in ['T2M', 'RH2M', 'PS', 'WS10M', 'WD10M'])]
    terrain_vars = [v for v in variables if any(terrain in v for terrain in ['elevation', 'slope'])]
    derived_vars = [v for v in variables if any(derived in v for derived in ['dryness', 'wind_temp', 'dry_days', 'dry_to_rain', 'change'])]
    other_vars = [v for v in variables if v not in fwi_vars + climate_vars + terrain_vars + derived_vars and v != 'fire_area']
    
    guide_text = f"""
🔥 WILDFIRE PREDICTION - OPTIMIZED VARIABLE SELECTION GUIDE

📊 TOTAL VARIABLES: {len(variables)}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 TARGET VARIABLE:
   • fire_area - 산불 피해 면적

🔥 FWI INDICES ({len(fwi_vars)} variables) - ALL PRESERVED:
   • FWI_0h - Fire Weather Index (종합 화재 위험도)
   • ISI_0h - Initial Spread Index (초기 확산 지수) 
   • DC_0h - Drought Code (가뭄 코드)
   • DMC_0h - Duff Moisture Code (부식토 습도 코드)
   • BUI_0h - Buildup Index (연료 축적 지수)
   • FFMC_0h - Fine Fuel Moisture Code (세부 연료 습도 코드)

🌡️ CLIMATE VARIABLES ({len(climate_vars)} variables):
   • Temperature: T2M_max, T2M_min, T2M_std
   • Humidity: RH2M_std  
   • Pressure: PS_max, PS_min
   • Wind: WS10M_max, WS10M_min, WS10M_std, WD10M_0h

⛰️ TERRAIN VARIABLES ({len(terrain_vars)} variables) - OPTIMIZED:
   • elevation_max (kept) - 최대 고도
   • slope_max (kept) - 최대 경사도
   ❌ elevation_mean (removed) - r=0.999 with elevation_max
   ❌ slope_mean (removed) - r=0.924 with slope_max

🧮 DERIVED FEATURES ({len(derived_vars)} variables):
   • dryness_index - 건조도 지수
   • wind_temp_product - 바람×온도 상호작용
   • dry_days_90d_start - 90일 건조일수
   • dry_to_rain_ratio_30d - 30일 건조/강수 비율
   • t2m_change_0_3h - 3시간 온도 변화
   • rh2m_change_3_6h - 습도 변화
   • ws10m_change_3_6h - 풍속 변화

🌿 OTHER VARIABLES ({len(other_vars)} variables):
   • ndvi_stress - 식생 스트레스 지수

⚠️ MULTICOLLINEARITY STATUS:
   • High correlations (|r|>=0.7): {len(high_corr_pairs)}
   • Action taken: Removed only redundant terrain variables
   • FWI correlations: PRESERVED (domain knowledge priority)

✅ RECOMMENDATION:
   • Use ALL variables in this optimized set
   • Apply regularization (Ridge/Lasso) for FWI correlations
   • Monitor model performance vs. interpretability trade-off
"""
    
    # Save guide
    output_dir = Path("correlation_analysis")
    guide_path = output_dir / "variable_selection_guide.txt"
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write(guide_text)
    
    print(f"✅ Variable guide saved: {guide_path}")
    print(guide_text)
    
    return guide_path

def main():
    """Main optimization workflow"""
    
    print("🚀 WILDFIRE VARIABLE OPTIMIZATION WORKFLOW")
    print("=" * 50)
    
    # Step 1: Create optimized variable set
    df, optimized_vars = create_optimized_variable_set()
    
    # Step 2: Analyze correlations
    corr_matrix, high_corr_pairs = analyze_optimized_correlations(df, optimized_vars)
    
    # Step 3: Create final heatmap
    heatmap_path = create_final_heatmap(df, optimized_vars, corr_matrix)
    
    # Step 4: Create guide
    guide_path = create_variable_guide(optimized_vars, high_corr_pairs)
    
    # Step 5: Summary
    print(f"\n🎉 OPTIMIZATION COMPLETE!")
    print(f"📁 Files generated:")
    print(f"   • {heatmap_path}")
    print(f"   • {guide_path}")
    print(f"\n💡 Next steps:")
    print(f"   1. Use optimized variable set for model training")
    print(f"   2. Apply Ridge/Lasso regularization for FWI correlations") 
    print(f"   3. Compare model performance before/after optimization")

if __name__ == "__main__":
    main()