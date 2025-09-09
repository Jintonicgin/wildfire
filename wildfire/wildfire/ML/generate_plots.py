import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier # For feature importance
import json

def generate_visualizations():
    """
    Loads pre-trained models and generates matplotlib visualizations.
    This script should be run from the 'wildfire/wildfire/ML/' directory.
    """
    # --- Configuration ---
    script_path = os.path.dirname(os.path.abspath(__file__))
    base_dir = script_path # The script is in the ML directory
    model_dir = os.path.join(base_dir, "models")
    data_path = os.path.join(base_dir, "final_merged_feature_engineered.csv")
    # The report is saved two levels up from the script directory
    report_dir = os.path.abspath(os.path.join(base_dir, "..", "..", "머신러닝 결과 보고서"))
    os.makedirs(report_dir, exist_ok=True)

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    df.dropna(subset=['fire_area', 'fire_duration_hours', 'WD10M_0h'], inplace=True)

    # --- Load Models and Scalers ---
    print("Loading pre-trained models and scalers...")
    try:
        speed_model = joblib.load(os.path.join(model_dir, "speed_classifier_model.joblib"))
        speed_scaler = joblib.load(os.path.join(model_dir, "speed_scaler.joblib"))
        direction_model = joblib.load(os.path.join(model_dir, "direction_classifier_model.joblib"))
        direction_scaler = joblib.load(os.path.join(model_dir, "direction_scaler.joblib"))
        
        with open(os.path.join(base_dir, "speed_model_columns.json"), 'r') as f:
            speed_features = json.load(f)
        with open(os.path.join(base_dir, "direction_model_columns.json"), 'r') as f:
            direction_features = json.load(f)

    except FileNotFoundError as e:
        print(f"Error: Model file not found. {e}")
        print("Please ensure the models are trained and saved in the 'models' directory by running train.py first.")
        return

    # --- Prepare Data ---
    # Create target variables for evaluation
    def classify_speed(speed: float, thresholds=(0.014, 0.11)) -> int:
        low, high = thresholds
        if speed <= low: return 0
        if speed <= high: return 1
        return 2

    def convert_degree_to_direction(deg: float) -> int:
        import math
        if deg is None or (isinstance(deg, float) and math.isnan(deg)) or deg == -999: return 0
        return int(math.floor(((float(deg) + 22.5) % 360) / 45))

    df["spread_speed"] = df.apply(
        lambda row: row["fire_area"] / row["fire_duration_hours"] if row["fire_duration_hours"] > 0 else 0, axis=1
    )
    df["actual_spread_speed_class"] = df["spread_speed"].apply(classify_speed)
    df["actual_spread_direction"] = df["WD10M_0h"].apply(convert_degree_to_direction)

    # Ensure all necessary columns exist before prediction
    for col in speed_features:
        if col not in df.columns:
            df[col] = 0
    for col in direction_features:
        if col not in df.columns:
            df[col] = 0
            
    X_speed = df[speed_features].copy().fillna(0)
    X_dir = df[direction_features].copy().fillna(0)
    
    y_speed_true = df["actual_spread_speed_class"]
    y_dir_true = df["actual_spread_direction"]

    # Scale data
    X_speed_scaled = speed_scaler.transform(X_speed)
    X_dir_scaled = direction_scaler.transform(X_dir)

    # --- Generate Predictions ---
    print("Generating predictions...")
    y_speed_pred = speed_model.predict(X_speed_scaled)
    y_dir_pred = direction_model.predict(X_dir_scaled)

    # --- Generate Visualizations ---
    print("Generating and saving plots...")

    # 1. Speed Model Confusion Matrix
    cm_speed = confusion_matrix(y_speed_true, y_speed_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_speed, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
    plt.title('Speed Model Confusion Matrix (Ensemble)', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    speed_cm_path = os.path.join(report_dir, "speed_confusion_matrix.png")
    plt.savefig(speed_cm_path)
    plt.close()
    print(f"Saved speed confusion matrix to: {speed_cm_path}")

    # 2. Direction Model Confusion Matrix
    dir_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    cm_dir = confusion_matrix(y_dir_true, y_dir_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_dir, annot=True, fmt='d', cmap='Blues', xticklabels=dir_labels, yticklabels=dir_labels)
    plt.title('Direction Model Confusion Matrix (Ensemble)', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    dir_cm_path = os.path.join(report_dir, "direction_confusion_matrix.png")
    plt.savefig(dir_cm_path)
    plt.close()
    print(f"Saved direction confusion matrix to: {dir_cm_path}")

    # 3. Feature Importance (from a newly trained RandomForest for simplicity)
    # The ensemble object doesn't expose feature_importances_, so we train one model for this plot.
    print("Training a RandomForest model to get feature importances...")
    rf_for_importance = RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)
    rf_for_importance.fit(X_speed_scaled, y_speed_true)
    
    importances = rf_for_importance.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': X_speed.columns, 'importance': importances}).sort_values(by='importance', ascending=False).head(20) # Top 20
    
    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title('Top 20 Feature Importance for Speed Prediction (RandomForest)', fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    feature_importance_path = os.path.join(report_dir, "speed_feature_importance.png")
    plt.savefig(feature_importance_path)
    plt.close()
    print(f"Saved feature importance plot to: {feature_importance_path}")
    
    print("\nAll visualizations have been generated successfully.")

if __name__ == "__main__":
    generate_visualizations()
