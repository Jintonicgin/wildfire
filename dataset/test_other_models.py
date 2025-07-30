import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import math

# --- 설정 ---
SOURCE_DATA_PATH = "final_merged_feature_engineered.csv"
MODEL_PATH = "/Users/mmymacymac/Developer/Projects/eclipse/WildFire/dataset/"

# --- 앙상블 클래스 정의 (필요시 사용) ---
class EnsembleRegressor:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        preds = [model.predict(X) for model in self.models]
        return np.mean(np.column_stack(preds), axis=1)

# --- 유틸리티 함수 (prediction.js에서 가져옴) ---
def convert_degree_to_direction(deg):
    if deg == -999 or deg is None or np.isnan(deg): return 0 # 기본값 'N'에 해당하는 인덱스 0 반환
    deg = float(deg) # Ensure it's a float
    index = math.floor(((deg + 22.5) % 360) / 45)
    return int(index) # 정수 인덱스 반환

def classify_speed(speed):
    # 모델의 클래스와 일치하도록 반환 값 조정 (0:느림, 1:보통, 2:빠름)
    if speed < 100: return 0 
    if speed < 300: return 1 
    return 2 

# --- 메인 실행 로직 ---
def main():
    print(f"1. 데이터 로딩: {SOURCE_DATA_PATH}")
    df = pd.read_csv(MODEL_PATH + SOURCE_DATA_PATH)

    # 필요한 컬럼만 선택하고 NaN 값 처리
    with open(MODEL_PATH + "speed_model_columns.json") as f:
        speed_features_expected = json.load(f)
    with open(MODEL_PATH + "direction_model_columns.json") as f:
        direction_features_expected = json.load(f)

    all_required_cols = list(set(speed_features_expected + direction_features_expected + 
                                 ['fire_area', 'fire_duration_hours', 'WD10M_0h']))
    df.dropna(subset=all_required_cols, inplace=True)

    # --- 2. 확산 속도 모델 테스트 ---
    print("\n--- 확산 속도 모델 테스트 ---")
    try:
        speed_model = joblib.load(MODEL_PATH + "speed_classifier_model.joblib")
        speed_scaler = joblib.load(MODEL_PATH + "speed_model_scaler.joblib")
        
        print(f"  - 속도 모델 클래스: {speed_model.classes_}") # 모델 클래스 출력

        # 실제 확산 속도 카테고리 계산
        df['actual_spread_distance'] = np.sqrt(df['fire_area'] * 10000 / np.pi) 
        df['actual_spread_speed'] = df['actual_spread_distance'] / df['fire_duration_hours']
        df['actual_spread_speed_category'] = df['actual_spread_speed'].apply(classify_speed)

        # 모델이 기대하는 피처 이름에 맞춰 데이터 준비
        X_speed = pd.DataFrame(index=df.index)
        for feature in speed_features_expected:
            col_in_csv = feature.lower().replace('_0h', '')
            if col_in_csv in df.columns:
                X_speed[feature] = df[col_in_csv]
            else:
                X_speed[feature] = 0 

        y_speed_actual = df['actual_spread_speed_category']

        X_speed.fillna(0, inplace=True)
        y_speed_actual.dropna(inplace=True)

        # 스케일링
        X_speed_scaled = speed_scaler.transform(X_speed)

        # 예측
        y_speed_pred = speed_model.predict(X_speed_scaled)

        print("  --- 모델 성능 평가 ---")
        print(f"  - 정확도 (Accuracy): {accuracy_score(y_speed_actual, y_speed_pred):.4f}")
        print("  - 분류 리포트 (Precision, Recall, F1-Score):")
        print(classification_report(y_speed_actual, y_speed_pred, zero_division=0))
        
        # 혼동 행렬 시각화
        cm_speed = confusion_matrix(y_speed_actual, y_speed_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_speed, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=speed_model.classes_, yticklabels=speed_model.classes_)
        plt.title('Speed Model Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(MODEL_PATH + 'speed_model_confusion_matrix.png')
        plt.close()

    except Exception as e:
        print(f"❌ 확산 속도 모델 테스트 실패: {e}")

    # --- 3. 확산 방향 모델 테스트 ---
    print("\n--- 확산 방향 모델 테스트 ---")
    try:
        direction_model = joblib.load(MODEL_PATH + "direction_classifier_model.joblib")
        
        print(f"  - 방향 모델 클래스: {direction_model.classes_}") # 모델 클래스 출력

        # 실제 확산 방향 계산
        df['actual_spread_direction'] = df['WD10M_0h'].apply(convert_degree_to_direction)

        # 모델이 기대하는 피처 이름에 맞춰 데이터 준비
        X_direction = pd.DataFrame(index=df.index)
        for feature in direction_features_expected:
            col_in_csv = feature.lower().replace('_0h', '')
            if col_in_csv in df.columns:
                X_direction[feature] = df[col_in_csv]
            else:
                X_direction[feature] = 0 

        y_direction_actual = df['actual_spread_direction']

        X_direction.fillna(0, inplace=True)
        y_direction_actual.dropna(inplace=True)

        # 예측
        y_direction_pred = direction_model.predict(X_direction)

        print("  --- 모델 성능 평가 ---")
        print(f"  - 정확도 (Accuracy): {accuracy_score(y_direction_actual, y_direction_pred):.4f}")
        print("  - 분류 리포트 (Precision, Recall, F1-Score):")
        print(classification_report(y_direction_actual, y_direction_pred, zero_division=0))

        # 혼동 행렬 시각화
        cm_direction = confusion_matrix(y_direction_actual, y_direction_pred, labels=direction_model.classes_)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_direction, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=direction_model.classes_, yticklabels=direction_model.classes_)
        plt.title('Direction Model Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(MODEL_PATH + 'direction_model_confusion_matrix.png')
        plt.close()

    except Exception as e:
        print(f"❌ 확산 방향 모델 테스트 실패: {e}")

if __name__ == "__main__":
    main()
