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

def analyze_detailed_correlations():
    """Detailed correlation analysis with expanded variable set"""
    
    print("🔍 Detailed Correlation Analysis...")
    
    # Load data
    df = pd.read_csv("final_merged_feature_engineered.csv")
    print(f"Data loaded: {df.shape}")
    
    # Expanded set of important variables
    important_vars = [
        # Target
        'fire_area',
        
        # Climate variables
        'T2M_max', 'T2M_mean', 'T2M_min', 'T2M_std',
        'RH2M_std', 'PS_max', 'PS_min',
        'WS10M_max', 'WS10M_mean', 'WS10M_min', 'WS10M_std',
        
        # Fire weather indices
        'FWI_0h', 'ISI_0h', 'DC_0h', 'DMC_0h', 'BUI_0h', 'FFMC_0h',
        
        # Vegetation and terrain
        'ndvi_stress', 'elevation_max', 'elevation_mean', 'slope_max', 'slope_mean',
        
        # Derived features
        'dryness_index', 'wind_temp_product',
        'dry_days_90d_start', 'dry_to_rain_ratio_30d',
        
        # Wind direction
        'WD10M_0h',
        
        # Change variables
        't2m_change_0_3h', 'rh2m_change_3_6h', 'ws10m_change_3_6h'
    ]
    
    # Filter available variables
    available_vars = [v for v in important_vars if v in df.columns]
    print(f"Available variables: {len(available_vars)}")
    print("Variables:", available_vars[:10], "..." if len(available_vars) > 10 else "")
    
    # Select numeric columns only
    numeric_data = df[available_vars].select_dtypes(include=[np.number])
    print(f"Numeric variables: {len(numeric_data.columns)}")
    
    # Calculate correlation matrix
    corr_matrix = numeric_data.corr()
    
    # Find high correlations
    print("\n🚨 HIGH CORRELATIONS (|r| >= 0.7):")
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
    
    if high_corr_pairs:
        high_corr_pairs = sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)
        for idx, pair in enumerate(high_corr_pairs, 1):
            print(f"{idx:2d}. {pair['var1']} ↔ {pair['var2']}: r = {pair['correlation']:.3f}")
    else:
        print("   No high correlations found")
    
    # Medium correlations
    print(f"\n📊 MEDIUM CORRELATIONS (0.5 <= |r| < 0.7):")
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
    
    if medium_corr_pairs:
        medium_corr_pairs = sorted(medium_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)
        for idx, pair in enumerate(medium_corr_pairs[:10], 1):  # Top 10
            print(f"{idx:2d}. {pair['var1']} ↔ {pair['var2']}: r = {pair['correlation']:.3f}")
    
    # Create enhanced heatmap
    output_dir = Path("correlation_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Filter to most interesting variables for visualization
    viz_vars = available_vars[:15] if len(available_vars) >= 15 else available_vars
    viz_corr = df[viz_vars].select_dtypes(include=[np.number]).corr()
    
    plt.figure(figsize=(14, 12))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(viz_corr, dtype=bool))
    
    # Enhanced heatmap
    sns.heatmap(viz_corr, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={"shrink": .8},
                linewidths=0.5)
    
    plt.title('Enhanced Variable Correlation Matrix', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save enhanced plot
    enhanced_path = output_dir / "enhanced_correlation_heatmap.png"
    plt.savefig(enhanced_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Enhanced heatmap saved: {enhanced_path}")
    
    # Summary statistics
    print(f"\n📈 CORRELATION SUMMARY:")
    print(f"   Total variable pairs analyzed: {len(corr_matrix.columns) * (len(corr_matrix.columns)-1) // 2}")
    print(f"   High correlations (|r|>=0.7): {len(high_corr_pairs)}")
    print(f"   Medium correlations (0.5<=|r|<0.7): {len(medium_corr_pairs)}")
    print(f"   Mean absolute correlation: {np.abs(corr_matrix.values).mean():.3f}")
    
    # Identify potential multicollinearity issues
    if high_corr_pairs:
        print(f"\n⚠️  MULTICOLLINEARITY WARNING:")
        print("   The following variable pairs may cause multicollinearity issues:")
        for pair in high_corr_pairs:
            print(f"   - Consider removing one of: {pair['var1']} or {pair['var2']}")

if __name__ == "__main__":
    analyze_detailed_correlations()