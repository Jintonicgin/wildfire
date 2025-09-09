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

# Set up matplotlib for English output
plt.style.use('default')
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_palette("husl")

def load_and_prepare_data():
    """Load and prepare data for current models"""
    print("📊 Loading and preprocessing data...")
    
    # Load data using train_v2_model.py approach
    try:
        df = pd.read_csv("gangwon_fire_data_augmented_parallel.csv")
        print(f"   - Original data size: {df.shape}")
    except FileNotFoundError:
        print("⚠️  Using alternative data file")
        df = pd.read_csv("final_merged_feature_engineered.csv")
        print(f"   - Alternative data size: {df.shape}")
    
    # Features as defined in train_v2_model.py
    features = [
        'lat', 'lng',
        'duration_hours', 'total_duration_hours',
        'T2M', 'RH2M', 'WS10M', 'WD10M', 'PRECTOTCORR',
        'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI'
    ]
    
    # Target variable selection
    target_candidates = ['estimated_damage_area', 'fire_area']
    target = None
    for candidate in target_candidates:
        if candidate in df.columns:
            target = candidate
            break
    
    if target is None:
        raise ValueError("Target variable not found")
    
    print(f"   - Target variable: {target}")
    print(f"   - Number of features: {len(features)}")
    
    # Remove missing values and outliers (same as train_v2_model.py)
    initial_size = len(df)
    df.dropna(subset=features + [target], inplace=True)
    df = df[(df[target] > 0) & (df[target] < df[target].quantile(0.99))]
    print(f"   - Data size after preprocessing: {df.shape} (removed: {initial_size - len(df)})")
    
    X = df[features]
    y = df[target]  # Log transformation applied later
    
    return X, y, features, target

def create_model_comparison_plots(X, y, features, target):
    """Create model performance comparison visualizations"""
    print("📈 Creating model performance comparison plots...")
    
    # Apply log transformation
    y_log = np.log1p(y)
    
    # Scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_log, test_size=0.2, random_state=42
    )
    
    # Train current models
    print("   - Training models...")
    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse')
    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    gb_model = GradientBoostingRegressor(random_state=42)
    
    xgb_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    
    # Predictions
    pred_xgb_log = xgb_model.predict(X_test)
    pred_rf_log = rf_model.predict(X_test)
    pred_gb_log = gb_model.predict(X_test)
    pred_ensemble_log = (pred_xgb_log + pred_rf_log + pred_gb_log) / 3
    
    # Convert back to original scale
    y_test_orig = np.expm1(y_test)
    pred_xgb = np.expm1(pred_xgb_log)
    pred_rf = np.expm1(pred_rf_log)
    pred_gb = np.expm1(pred_gb_log)
    pred_ensemble = np.expm1(pred_ensemble_log)
    
    # 1. Actual vs Predicted scatter plots (current models)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Current Models: Actual vs Predicted Damage Area', fontsize=16, fontweight='bold')
    
    models = [
        ('XGBoost', pred_xgb, 'tab:blue'),
        ('Random Forest', pred_rf, 'tab:orange'), 
        ('Gradient Boosting', pred_gb, 'tab:green'),
        ('Ensemble (Average)', pred_ensemble, 'tab:red')
    ]
    
    for idx, (name, pred, color) in enumerate(models):
        ax = axes[idx // 2, idx % 2]
        
        # Scatter plot
        ax.scatter(y_test_orig, pred, alpha=0.6, color=color, s=30)
        
        # Perfect prediction line
        max_val = max(y_test_orig.max(), pred.max())
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
        
        # Calculate performance metrics
        rmse = np.sqrt(mean_squared_error(y_test_orig, pred))
        mae = mean_absolute_error(y_test_orig, pred)
        r2 = r2_score(y_test_orig, pred)
        
        ax.set_xlabel('Actual Damage Area (ha)')
        ax.set_ylabel('Predicted Damage Area (ha)')
        ax.set_title(f'{name}\nRMSE: {rmse:.1f}, MAE: {mae:.1f}, R²: {r2:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/mmymacymac/Developer/Projects/WildFire_projects/머신러닝 결과 보고서/english_models_actual_vs_predicted.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return xgb_model, rf_model, gb_model, X_train, y_train, scaler

def create_feature_importance_plot(xgb_model, rf_model, features):
    """Create feature importance comparison visualization"""
    print("📊 Creating feature importance visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # XGBoost feature importance
    xgb_importance = xgb_model.feature_importances_
    xgb_indices = np.argsort(xgb_importance)[::-1]
    
    axes[0].barh(range(len(features)), xgb_importance[xgb_indices], color='steelblue')
    axes[0].set_yticks(range(len(features)))
    axes[0].set_yticklabels([features[i] for i in xgb_indices])
    axes[0].set_xlabel('Importance')
    axes[0].set_title('XGBoost Feature Importance')
    axes[0].grid(True, alpha=0.3)
    
    # Random Forest feature importance
    rf_importance = rf_model.feature_importances_
    rf_indices = np.argsort(rf_importance)[::-1]
    
    axes[1].barh(range(len(features)), rf_importance[rf_indices], color='darkorange')
    axes[1].set_yticks(range(len(features)))
    axes[1].set_yticklabels([features[i] for i in rf_indices])
    axes[1].set_xlabel('Importance')
    axes[1].set_title('Random Forest Feature Importance')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/mmymacymac/Developer/Projects/WildFire_projects/머신러닝 결과 보고서/english_models_feature_importance.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_kfold_performance_plot(X, y, features):
    """Create K-Fold cross-validation performance visualization"""
    print("🔄 Creating K-Fold cross-validation performance visualization...")
    
    y_log = np.log1p(y)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store results
    results = {'Fold': [], 'Model': [], 'RMSE': [], 'R2': [], 'MAE': []}
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]
        
        # Models
        models = {
            'XGBoost': xgb.XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse'),
            'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(random_state=42)
        }
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            pred_log = model.predict(X_val)
            
            # Convert to original scale
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
        
        # Ensemble calculation
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
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # RMSE boxplot
    sns.boxplot(data=results_df, x='Model', y='RMSE', ax=axes[0])
    axes[0].set_title('K-Fold Cross-Validation RMSE Distribution')
    axes[0].tick_params(axis='x', rotation=45)
    
    # R² boxplot
    sns.boxplot(data=results_df, x='Model', y='R2', ax=axes[1])
    axes[1].set_title('K-Fold Cross-Validation R² Distribution')
    axes[1].tick_params(axis='x', rotation=45)
    
    # MAE boxplot
    sns.boxplot(data=results_df, x='Model', y='MAE', ax=axes[2])
    axes[2].set_title('K-Fold Cross-Validation MAE Distribution')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('/Users/mmymacymac/Developer/Projects/WildFire_projects/머신러닝 결과 보고서/english_models_kfold_performance.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print average performance
    print("\n📊 K-Fold Average Performance:")
    avg_performance = results_df.groupby('Model')[['RMSE', 'R2', 'MAE']].mean()
    for model in avg_performance.index:
        rmse, r2, mae = avg_performance.loc[model]
        print(f"  - {model:15s}: RMSE={rmse:.1f}, R²={r2:.3f}, MAE={mae:.1f}")

def create_target_distribution_plot(y, target):
    """Create target variable distribution visualization"""
    print("📊 Creating target variable distribution visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original distribution
    axes[0].hist(y, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].set_xlabel(f'{target} (ha)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Original {target} Distribution')
    axes[0].axvline(y.mean(), color='red', linestyle='--', label=f'Mean: {y.mean():.1f}')
    axes[0].axvline(y.median(), color='orange', linestyle='--', label=f'Median: {y.median():.1f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Log-transformed distribution
    y_log = np.log1p(y)
    axes[1].hist(y_log, bins=50, alpha=0.7, color='darkorange', edgecolor='black')
    axes[1].set_xlabel(f'log1p({target})')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Log-transformed {target} Distribution')
    axes[1].axvline(y_log.mean(), color='red', linestyle='--', label=f'Mean: {y_log.mean():.2f}')
    axes[1].axvline(y_log.median(), color='orange', linestyle='--', label=f'Median: {y_log.median():.2f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/mmymacymac/Developer/Projects/WildFire_projects/머신러닝 결과 보고서/english_target_distribution.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print descriptive statistics
    print(f"\n📈 {target} Descriptive Statistics:")
    print(f"  - Mean: {y.mean():.1f} ha")
    print(f"  - Median: {y.median():.1f} ha")  
    print(f"  - Std Dev: {y.std():.1f} ha")
    print(f"  - Min: {y.min():.1f} ha")
    print(f"  - Max: {y.max():.1f} ha")
    print(f"  - 75th percentile: {y.quantile(0.75):.1f} ha")
    print(f"  - 95th percentile: {y.quantile(0.95):.1f} ha")

def create_residual_analysis_plot(X, y, features):
    """Create residual analysis visualization"""
    print("📊 Creating residual analysis visualization...")
    
    y_log = np.log1p(y)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_log, test_size=0.2, random_state=42
    )
    
    # Train ensemble model
    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse')
    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    gb_model = GradientBoostingRegressor(random_state=42)
    
    xgb_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    
    # Ensemble prediction
    pred_log = (xgb_model.predict(X_test) + rf_model.predict(X_test) + gb_model.predict(X_test)) / 3
    
    y_test_orig = np.expm1(y_test)
    pred_orig = np.expm1(pred_log)
    
    residuals = y_test_orig - pred_orig
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Current Ensemble Model Residual Analysis', fontsize=16, fontweight='bold')
    
    # 1. Residuals vs Predicted values
    axes[0, 0].scatter(pred_orig, residuals, alpha=0.6, color='steelblue')
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Predicted Values (ha)')
    axes[0, 0].set_ylabel('Residuals (ha)')
    axes[0, 0].set_title('Residuals vs Predicted Values')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residual histogram
    axes[0, 1].hist(residuals, bins=30, alpha=0.7, color='darkorange', edgecolor='black')
    axes[0, 1].set_xlabel('Residuals (ha)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Residual Distribution')
    axes[0, 1].axvline(residuals.mean(), color='red', linestyle='--', label=f'Mean: {residuals.mean():.1f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Q-Q plot (normality test)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normality Test)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Actual values vs Residuals
    axes[1, 1].scatter(y_test_orig, residuals, alpha=0.6, color='green')
    axes[1, 1].axhline(y=0, color='red', linestyle='--')
    axes[1, 1].set_xlabel('Actual Values (ha)')
    axes[1, 1].set_ylabel('Residuals (ha)')
    axes[1, 1].set_title('Actual Values vs Residuals')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/mmymacymac/Developer/Projects/WildFire_projects/머신러닝 결과 보고서/english_models_residual_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main execution function"""
    print("🔥 Creating English visualizations for current models!")
    print("=" * 50)
    
    try:
        # 1. Load and preprocess data
        X, y, features, target = load_and_prepare_data()
        
        # 2. Model comparison visualization
        xgb_model, rf_model, gb_model, X_train, y_train, scaler = create_model_comparison_plots(X, y, features, target)
        
        # 3. Feature importance visualization
        create_feature_importance_plot(xgb_model, rf_model, features)
        
        # 4. K-Fold performance visualization
        create_kfold_performance_plot(X, y, features)
        
        # 5. Target distribution visualization
        create_target_distribution_plot(y, target)
        
        # 6. Residual analysis visualization
        create_residual_analysis_plot(X, y, features)
        
        print("\n" + "=" * 50)
        print("✅ All English visualizations created successfully!")
        print("📁 Save location: /Users/mmymacymac/Developer/Projects/WildFire_projects/머신러닝 결과 보고서/")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()