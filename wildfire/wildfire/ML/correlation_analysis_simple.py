import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend to Agg for non-interactive plotting
import matplotlib
matplotlib.use('Agg')

# Fix font issue
plt.rcParams['font.family'] = 'DejaVu Sans'

def main():
    """Simple correlation analysis"""
    
    print("🔥 Starting Variable Correlation Analysis...")
    
    # Load data
    data_path = "final_merged_feature_engineered.csv"
    print(f"Loading data from {data_path}...")
    
    try:
        df = pd.read_csv(data_path)
        print(f"Data loaded successfully: {df.shape}")
    except FileNotFoundError:
        print(f"Error: {data_path} not found")
        return
    
    # Select key variables for correlation analysis
    key_variables = [
        'fire_area',
        'dryness_index', 'T2M_max', 'WS10M_max', 'wind_temp_product',
        'FWI_0h', 'ndvi_stress', 'elevation_max', 'slope_max',
        'PS_max', 'RH2M_std', 'dry_days_90d_start',
        'WD10M_0h', 'dry_to_rain_ratio_30d'
    ]
    
    # Filter available variables
    available_vars = [v for v in key_variables if v in df.columns]
    print(f"Available variables: {len(available_vars)}")
    
    if len(available_vars) < 3:
        print("Not enough variables available for correlation analysis")
        return
    
    # Create output directory
    output_dir = Path("correlation_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Calculate correlation matrix
    print("📊 Calculating correlation matrix...")
    corr_data = df[available_vars].select_dtypes(include=[np.number])
    corr_matrix = corr_data.corr()
    
    # Create correlation heatmap
    print("🎨 Creating correlation heatmap...")
    plt.figure(figsize=(12, 10))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Generate heatmap
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={"shrink": .8})
    
    plt.title('Key Variables Correlation Matrix', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save plot
    heatmap_path = output_dir / "variables_correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Correlation heatmap saved: {heatmap_path}")
    
    # Find high correlations
    print("\n🔍 High Correlations (|r| >= 0.7):")
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
        for pair in high_corr_pairs:
            print(f"   {pair['var1']} ↔ {pair['var2']}: r = {pair['correlation']:.3f}")
    else:
        print("   No high correlations found (|r| >= 0.7)")
    
    # Create scatter plot matrix for top variables
    print("\n📈 Creating scatter plot matrix...")
    top_vars = available_vars[:6]  # Top 6 variables
    
    if len(top_vars) >= 3:
        # Sample data for performance
        sample_size = min(1000, len(df))
        df_sample = df[top_vars].sample(n=sample_size)
        
        # Create pairplot
        plt.figure(figsize=(15, 15))
        g = sns.pairplot(df_sample, diag_kind='hist', plot_kws={'alpha': 0.6, 's': 20})
        g.fig.suptitle('Top Variables Scatter Plot Matrix', y=1.02, fontsize=16)
        
        # Save pairplot
        pairplot_path = output_dir / "variables_pairplot.png"
        plt.savefig(pairplot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Pairplot saved: {pairplot_path}")
    
    # Summary statistics
    print(f"\n📊 Correlation Summary:")
    print(f"   Variables analyzed: {len(available_vars)}")
    print(f"   High correlations (|r|>=0.7): {len(high_corr_pairs)}")
    print(f"   Mean absolute correlation: {np.abs(corr_matrix.values).mean():.3f}")
    
    print(f"\n✅ Analysis complete! Files saved in {output_dir}/")

if __name__ == "__main__":
    main()