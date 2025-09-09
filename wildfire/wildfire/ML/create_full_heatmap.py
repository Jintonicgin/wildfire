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

def create_full_symmetric_heatmap():
    """Create full symmetric correlation heatmap"""
    
    print("🎨 Creating Full Symmetric Correlation Heatmap...")
    
    # Load data
    df = pd.read_csv("final_merged_feature_engineered.csv")
    
    # Use optimized variable set
    optimized_vars = [
        'fire_area',
        'FWI_0h', 'ISI_0h', 'DC_0h', 'DMC_0h', 'BUI_0h', 'FFMC_0h',
        'ndvi_stress', 'elevation_max', 'slope_max',
        'dryness_index', 'wind_temp_product',
        'dry_days_90d_start'
    ]
    
    # Filter available variables  
    available_vars = [v for v in optimized_vars if v in df.columns]
    
    # Select numeric data and calculate correlation
    numeric_data = df[available_vars].select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()
    
    print(f"Variables included: {len(available_vars)}")
    
    # Create output directory
    output_dir = Path("correlation_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Create full symmetric heatmap
    plt.figure(figsize=(14, 12))
    
    # NO MASK - show full symmetric matrix
    sns.heatmap(corr_matrix,
                annot=True,
                cmap='RdBu_r',
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={"shrink": .8},
                linewidths=0.5,
                annot_kws={'size': 9})
    
    plt.title('🔥 Complete Symmetric Correlation Matrix\n(Optimized Variable Set - FWI Preserved)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save full heatmap
    full_path = output_dir / "full_symmetric_correlation_heatmap.png"
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Full symmetric heatmap saved: {full_path}")
    
    # Also create a compact version with better readability
    plt.figure(figsize=(16, 14))
    
    # Full heatmap with enhanced styling
    sns.heatmap(corr_matrix,
                annot=True,
                cmap='RdBu_r',
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={"shrink": .7, "pad": .02},
                linewidths=0.3,
                annot_kws={'size': 8, 'weight': 'bold'})
    
    plt.title('🔥 WILDFIRE PREDICTION - COMPLETE CORRELATION MATRIX\n' + 
             '(All FWI Indices Preserved | Redundant Terrain Variables Removed)',
              fontsize=18, fontweight='bold', pad=25)
    
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    # Save enhanced version
    enhanced_path = output_dir / "enhanced_full_correlation_heatmap.png"
    plt.savefig(enhanced_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Enhanced full heatmap saved: {enhanced_path}")
    
    # Print correlation summary for reference
    print(f"\n📊 Correlation Matrix Summary:")
    print(f"   Matrix size: {corr_matrix.shape[0]} × {corr_matrix.shape[1]}")
    print(f"   Variables: {list(corr_matrix.columns)}")
    
    # Show highest and lowest correlations
    # Get upper triangle values (excluding diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    upper_tri_values = corr_matrix.where(mask)
    
    # Find max and min correlations
    max_corr_idx = np.unravel_index(np.nanargmax(upper_tri_values.values), upper_tri_values.shape)
    min_corr_idx = np.unravel_index(np.nanargmin(upper_tri_values.values), upper_tri_values.shape)
    
    max_corr = upper_tri_values.iloc[max_corr_idx]
    min_corr = upper_tri_values.iloc[min_corr_idx]
    
    print(f"\n🔝 Highest correlation: {corr_matrix.columns[max_corr_idx[1]]} ↔ {corr_matrix.columns[max_corr_idx[0]]}: r = {max_corr:.3f}")
    print(f"🔻 Lowest correlation: {corr_matrix.columns[min_corr_idx[1]]} ↔ {corr_matrix.columns[min_corr_idx[0]]}: r = {min_corr:.3f}")
    
    return enhanced_path

if __name__ == "__main__":
    create_full_symmetric_heatmap()