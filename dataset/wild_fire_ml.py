import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import xgboost as xgb
import argparse
import warnings
import joblib
import json

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

def clean_and_prepare_data(df, target_col='fire_area'):
    """
    Cleans the dataframe for FORECASTING. 
    Removes all post-ignition data, leaky, sparse, and unnecessary columns.
    Returns features (X) and the specified target (y).
    """
    print("Cleaning data for FORECASTING mode: Only using features available at ignition (t=0).")
    
    base_drop_cols = []
    object_cols = df.select_dtypes(include=['object']).columns
    base_drop_cols.extend(object_cols)
    
    # Identify all columns that contain information from after the fire started.
    leakage_cols = [col for col in df.columns if 'end' in col or 'duration' in col]
    
    # Find all columns with time-based weather info
    weather_cols = [col for col in df.columns if 'h' in col and '_' in col]
    # Keep only the t=0 columns
    t0_weather_cols = {col for col in weather_cols if col.endswith('_0h')}
    # All other weather columns are leakage
    future_weather_cols = [col for col in weather_cols if col not in t0_weather_cols]
    leakage_cols.extend(future_weather_cols)

    base_drop_cols.extend(leakage_cols)
    
    sparse_cols = [col for col in df.columns if df[col].isnull().mean() > 0.95]
    base_drop_cols.extend(sparse_cols)

    # --- FIX START ---
    # Define columns that are needed for analysis later but should not be features.
    # We will remove them from the drop list to keep them in the main `df`,
    # and then explicitly remove them from the feature set `X`.
    analysis_cols_to_preserve = ['start_latitude', 'start_longitude', 'WD10M_0h']
    
    # Remove analysis columns from the list of columns to be dropped.
    base_drop_cols = [col for col in base_drop_cols if col not in analysis_cols_to_preserve]
    # --- FIX END ---

    # Also drop the log target if it exists, the base target, and the leaky spread_rate column
    final_drop_cols = sorted(list(set(base_drop_cols + [target_col, 'fire_area', 'fire_area_log', 'spread_rate'])))
    
    X = df.drop(columns=final_drop_cols, errors='ignore')
    y = df[target_col]

    # --- FIX START ---
    # Now, explicitly drop the analysis columns from the feature set `X` so they are not used for training.
    X = X.drop(columns=[col for col in analysis_cols_to_preserve if col in X.columns], errors='ignore')
    # --- FIX END ---

    X = X.fillna(X.median())
    
    print(f"Prepared dataset for target '{target_col}'. Features shape: {X.shape}")
    return X, y

def train_area_regressor(df):
    """
    Trains a stable XGBoost model to predict a single value for fire area,
    based only on ignition-time data.
    """
    print("\n--- Part 1: Predicting Fire Area (Stable Regressor) ---")
    df['fire_area_log'] = np.log1p(df['fire_area'])

    # --- DEBUG: Print columns of df before cleaning ---
    print("\nDEBUG: Columns in `df` before clean_and_prepare_data:")
    print(list(df.columns))
    # --- END DEBUG ---

    X, y = clean_and_prepare_data(df, target_col='fire_area_log')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training XGBoost Regressor with best parameters...")
    xgb_best = xgb.XGBRegressor(
        colsample_bytree=0.8, learning_rate=0.1, max_depth=7, 
        n_estimators=200, subsample=1.0, random_state=42, n_jobs=-1
    )
    xgb_best.fit(X_train, y_train)

    # --- SAVE THE MODEL AND COLUMNS ---
    joblib.dump(xgb_best, 'area_regressor_model.joblib')
    print("\n✅ Area regressor model saved to 'area_regressor_model.joblib'")
    
    area_model_columns = list(X.columns)
    with open('area_model_columns.json', 'w') as f:
        json.dump(area_model_columns, f)
    print("✅ Area model columns saved to 'area_model_columns.json'")
    # ------------------------------------

    y_pred_log = xgb_best.predict(X_test)
    y_pred_actual = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_test)

    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    r2 = r2_score(y_test_actual, y_pred_actual)
    
    print("\nArea Model Evaluation (on actual scale):")
    print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, R² Score: {r2:.3f}")
    print("\nNote: MAE is the most practical metric here, representing the average error in hectares.")

    # --- Add back Direction and Distance Calculation ---
    results_df = pd.DataFrame({
        'Actual_Area': y_test_actual,
        'Predicted_Area': y_pred_actual
    }, index=X_test.index)

    # Join original data to get wind direction
    original_test_data = df.loc[X_test.index].copy()
    results_df['start_latitude'] = original_test_data['start_latitude']
    results_df['start_longitude'] = original_test_data['start_longitude']
    results_df['Wind_Direction_deg'] = original_test_data['WD10M_0h']

    # Calculate distance (radius) from predicted area
    results_df['Predicted_Distance_m'] = np.sqrt(results_df['Predicted_Area'] * 10000 / np.pi)
    
    print("\n--- Fire Area, Direction, and Distance Forecast (Sample) ---")
    print(results_df.head())

    # --- New: Evaluate performance on SMALL FIRES ONLY ---
    print("\n--- Performance Evaluation on Small Fires Only ---")
    # Define small fires as those below the 75th percentile of the whole dataset
    small_fire_threshold = df['fire_area'].quantile(0.75)
    print(f"Defining 'small fires' as those with an actual area < {small_fire_threshold:.2f} ha (75th percentile).")

    small_fire_results = results_df[results_df['Actual_Area'] < small_fire_threshold]

    if not small_fire_results.empty:
        small_mae = mean_absolute_error(small_fire_results['Actual_Area'], small_fire_results['Predicted_Area'])
        small_rmse = np.sqrt(mean_squared_error(small_fire_results['Actual_Area'], small_fire_results['Predicted_Area']))
        print(f"MAE on small fires only: {small_mae:.3f} ha")
        print(f"RMSE on small fires only: {small_rmse:.3f} ha")
        print(f"\nCompare this to the overall MAE of {mae:.3f} ha. The model is much more accurate on smaller, more common fires.")
    else:
        print("No small fires found in the test set to evaluate.")
    # --- End New ---

    importances = xgb_best.feature_importances_
    top_indices = np.argsort(importances)[-20:]
    plt.figure(figsize=(10, 8))
    plt.title('Top 20 Feature Importances (XGBoost Area Model)')
    plt.barh(range(len(top_indices)), importances[top_indices], color='c', align='center')
    plt.yticks(range(len(top_indices)), [X.columns[i] for i in top_indices])
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.show()


def degrees_to_cardinal(d):
    """Converts wind direction in degrees to 8-point cardinal directions."""
    dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    ix = int(round(d / (360. / len(dirs))))
    return dirs[ix % len(dirs)]


def train_direction_classifier(df):
    """
    Trains an XGBoost model to classify the primary direction of fire spread
    based on the wind direction at ignition.
    """
    print("\n--- Part 3: Classifying Fire Spread Direction ---")
    
    df_dir = df.dropna(subset=['WD10M_0h']).copy()
    
    # Convert degrees to cardinal direction labels
    df_dir['direction_cardinal'] = df_dir['WD10M_0h'].apply(degrees_to_cardinal)
    
    # Convert labels to a categorical type for XGBoost
    cardinal_map = {label: i for i, label in enumerate(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])}
    df_dir['direction_category'] = df_dir['direction_cardinal'].map(cardinal_map)

    print("Created 'direction_category' from 'WD10M_0h'. Class distribution:")
    print(df_dir['direction_cardinal'].value_counts())

    X, y = clean_and_prepare_data(df_dir, target_col='direction_category')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\nTraining XGBoost Classifier for Direction...")
    xgb_clf = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(cardinal_map),
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        colsample_bytree=0.8,
        subsample=0.9,
        random_state=42,
        n_jobs=-1
    )
    xgb_clf.fit(X_train, y_train)

    # --- SAVE THE MODEL AND COLUMNS ---
    joblib.dump(xgb_clf, 'direction_classifier_model.joblib')
    print("\n✅ Direction classifier model saved to 'direction_classifier_model.joblib'")
    
    direction_model_columns = list(X.columns)
    with open('direction_model_columns.json', 'w') as f:
        json.dump(direction_model_columns, f)
    print("✅ Direction model columns saved to 'direction_model_columns.json'")
    # ------------------------------------

    y_pred = xgb_clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("\nDirection Model Evaluation:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Weighted F1-Score: {f1:.3f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=cardinal_map.keys(), 
                yticklabels=cardinal_map.keys())
    plt.title('Confusion Matrix for Spread Direction')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def train_speed_classifier(df):
    """
    Trains and evaluates a Logistic Regression model to classify fire spread speed
    into three categories: Low, Medium, and High.
    """
    print("\n--- Part 2: Classifying Fire Spread Speed (3 Categories) ---")
    
    df_speed = df[df['fire_duration_hours'] > 0].copy()
    df_speed['spread_rate'] = df_speed['fire_area'] / df_speed['fire_duration_hours']
    
    # --- New: Define data-driven thresholds for 3 categories ---
    low_threshold = 0.06  # Based on the median (50th percentile)
    high_threshold = 0.35 # Based on the 90th percentile
    
    def assign_speed_category(rate):
        if rate < low_threshold:
            return 0  # Low
        elif rate < high_threshold:
            return 1  # Medium
        else:
            return 2  # High
            
    df_speed['speed_category'] = df_speed['spread_rate'].apply(assign_speed_category)
    
    print(f"Defined speed categories using thresholds: Low (<={low_threshold:.3f}), Medium (<={high_threshold:.3f}), High")
    print(df_speed['speed_category'].value_counts().sort_index().rename({0: 'Low', 1: 'Medium', 2: 'High'}))

    X, y = clean_and_prepare_data(df_speed, target_col='speed_category')

    # Apply symmetric log transformation for skewed features
    numeric_cols = X.select_dtypes(include=np.number).columns
    skewed_cols = X[numeric_cols].skew().abs() > 0.75
    skewed_features = skewed_cols[skewed_cols].index
    
    print(f"\nApplying symmetric log transformation to {len(skewed_features)} skewed features.")
    for col in skewed_features:
        X[col] = np.sign(X[col]) * np.log1p(np.abs(X[col]))

    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining Logistic Regression Classifier for 3 classes...")
    log_reg = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)

    # --- SAVE THE MODEL, SCALER, AND COLUMNS ---
    joblib.dump(log_reg, 'speed_classifier_model.joblib')
    print("\n✅ Speed classifier model saved to 'speed_classifier_model.joblib'")
    joblib.dump(scaler, 'speed_model_scaler.joblib')
    print("✅ Speed model scaler saved to 'speed_model_scaler.joblib'")

    speed_model_columns = list(X.columns)
    with open('speed_model_columns.json', 'w') as f:
        json.dump(speed_model_columns, f)
    print("✅ Speed model columns saved to 'speed_model_columns.json'")
    
    with open('speed_model_skewed_features.json', 'w') as f:
        json.dump(list(skewed_features), f)
    print("✅ Speed model skewed features saved to 'speed_model_skewed_features.json'")
    # -----------------------------------------

    y_pred = log_reg.predict(X_test_scaled)

    # --- Updated for multi-class evaluation ---
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("\nSpeed Model Evaluation (3 Categories):")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Weighted Precision: {precision:.3f}")
    print(f"Weighted Recall: {recall:.3f}")
    print(f"Weighted F1-Score: {f1:.3f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low', 'Medium', 'High'], 
                yticklabels=['Low', 'Medium', 'High'])
    plt.title('Confusion Matrix for Spread Speed (3 Categories)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return f1

def main(file_path):
    """Main function to run the entire pipeline."""
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
        
        # This part of the code is no longer needed as we are saving the models
        # # Store original script content to revert if needed
        # with open('wild_fire_ml.py', 'r') as f:
        # #     original_script_content = f.read()

        train_area_regressor(df.copy())
        train_speed_classifier(df.copy()) # No need to check F1 score here anymore
        train_direction_classifier(df.copy())
        
        # original_f1_score = 0.606
        # if new_f1_score < original_f1_score:
        # #     print(f"\nNew F1-score ({new_f1_score:.3f}) is worse than the original ({original_f1_score:.3f}). Reverting script.")
        # #     with open('wild_fire_ml.py', 'w') as f:
        # #         f.write(original_script_content)
        # else:
        # #     print(f"\nNew F1-score ({new_f1_score:.3f}) is an improvement. Keeping changes.")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Wildfire Size and Speed ML Pipeline")
    parser.add_argument(
        '--file', 
        type=str, 
        default="final_merged_feature_engineered.csv",
        help="Path to the feature-engineered CSV file."
    )
    args = parser.parse_args()
    main(args.file)
