import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import math
from imblearn.over_sampling import SMOTE
from collections import Counter

SOURCE_DATA_PATH = "final_merged_feature_engineered.csv"
MODEL_PATH = "/Users/mmymacymac/Developer/Projects/eclipse/WildFire/dataset/"

class EnsembleRegressor:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        preds = [model.predict(X) for model in self.models]
        return np.mean(np.column_stack(preds), axis=1)

class EnsembleClassifier:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions)

def convert_degree_to_direction(deg):
    if deg == -999 or deg is None or np.isnan(deg): return 0
    deg = float(deg)
    return int(math.floor(((deg + 22.5) % 360) / 45))

def classify_speed(speed):
    if speed < 100: return 0
    if speed < 300: return 1
    return 2

def main():
    print(f"1. 데이터 로딩: {SOURCE_DATA_PATH}")
    df = pd.read_csv(MODEL_PATH + SOURCE_DATA_PATH)

    with open(MODEL_PATH + "speed_model_columns.json") as f:
        speed_features_expected = json.load(f)
    with open(MODEL_PATH + "direction_model_columns.json") as f:
        direction_features_expected = json.load(f)

    all_required_cols = list(set(speed_features_expected + direction_features_expected + ['fire_area', 'fire_duration_hours', 'WD10M_0h']))
    df.dropna(subset=all_required_cols, inplace=True)

    print("\n--- 확산 속도 모델 재학습 (현재 데이터셋으로는 불가능) ---")
    print("\n--- 확산 방향 모델 재학습 ---")

    try:
        df['actual_spread_direction'] = df['WD10M_0h'].apply(convert_degree_to_direction)

        time_intervals = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42]
        new_direction_features = []

        for i in range(len(time_intervals) - 1):
            def gen_feat(prefix): return f"{prefix}_{time_intervals[i]}h", f"{prefix}_{time_intervals[i+1]}h"

            for prefix, name in [("WD10M", "wind_direction_change"), ("WS10M", "wind_speed_change"),
                                 ("T2M", "temp_change"), ("RH2M", "humidity_change")]:
                col1, col2 = gen_feat(prefix)
                new_col = f"{name}_{time_intervals[i]}_{time_intervals[i+1]}h"
                if col1 in df.columns and col2 in df.columns:
                    delta = np.abs(df[col1] - df[col2])
                    if "direction" in name:
                        delta = delta.apply(lambda x: min(x, 360 - x) if not np.isnan(x) else x)
                    df[new_col] = delta
                    new_direction_features.append(new_col)

        direction_features_expected.extend(new_direction_features)
        direction_features_expected = list(set(direction_features_expected))

        X_direction = pd.DataFrame(index=df.index)
        for feature in direction_features_expected:
            col = feature.lower().replace('_0h', '')
            X_direction[feature] = df[col] if col in df.columns else 0

        y_direction = df['actual_spread_direction'].dropna()
        X_direction.fillna(0, inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X_direction, y_direction, test_size=0.2, random_state=42, stratify=y_direction)

        print(f"  - SMOTE 적용 전 방향 클래스 분포: {Counter(y_train)}")
        k_neighbors = max(min(Counter(y_train).values()) - 1, 1)
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        print(f"  - SMOTE 적용 후 방향 클래스 분포: {Counter(y_resampled)}")

        print("  - 개별 분류 모델 학습 시작...")
        classifiers = {
            "xgb": xgb.XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='mlogloss'),
            "rf": RandomForestClassifier(random_state=42, n_jobs=-1),
            "gb": GradientBoostingClassifier(random_state=42),
            "lgbm": lgb.LGBMClassifier(random_state=42, n_jobs=-1),
            "cat": CatBoostClassifier(random_state=42, verbose=0, allow_writing_files=False)
        }

        param_grid_xgb = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
        gs_xgb = GridSearchCV(classifiers["xgb"], param_grid_xgb, cv=3, scoring='accuracy', n_jobs=-1)
        gs_xgb.fit(X_resampled, y_resampled)
        best_xgb = gs_xgb.best_estimator_
        print(f"  - XGBoost 최적 하이퍼파라미터: {gs_xgb.best_params_}")

        param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [5, 10]}
        gs_rf = GridSearchCV(classifiers["rf"], param_grid_rf, cv=3, scoring='accuracy', n_jobs=-1)
        gs_rf.fit(X_resampled, y_resampled)
        best_rf = gs_rf.best_estimator_
        print(f"  - RandomForest 최적 하이퍼파라미터: {gs_rf.best_params_}")

        best_lgbm = classifiers["lgbm"].fit(X_resampled, y_resampled)
        best_cat = classifiers["cat"].fit(X_resampled, y_resampled)
        best_gb = classifiers["gb"].fit(X_resampled, y_resampled)

        ensemble = EnsembleClassifier([best_xgb, best_rf, best_gb, best_lgbm, best_cat])
        print("  - 앙상블 모델 생성 완료.")

        y_pred = ensemble.predict(X_test)
        print("  --- 앙상블 모델 성능 평가 ---")
        print(f"  - 정확도 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))

        joblib.dump(ensemble, MODEL_PATH + "direction_classifier_model.joblib")
        print("  ✅ 확산 방향 앙상블 모델 저장 완료.")

    except Exception as e:
        print(f"❌ 확산 방향 모델 재학습 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()