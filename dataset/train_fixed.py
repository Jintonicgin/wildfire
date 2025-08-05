import os
import json
import math
from collections import Counter

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

try:
    import xgboost as xgb
except ImportError:
    xgb = None  # optional
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None


def classify_speed(speed: float) -> int:
    """
    속도 값을 세 구간으로 나누는 함수.

    - speed <= 0.014  → 0
    - 0.014 < speed <= 0.11 → 1
    - speed > 0.11 → 2
    """
    if speed <= 0.014:
        return 0
    if speed <= 0.11:
        return 1
    return 2


def convert_degree_to_direction(deg: float) -> int:
    """
    풍향(0~360°)을 8방위(0~7)로 변환한다.

    deg가 결측치이거나 -999로 표기된 경우 0으로 처리한다.
    """
    if deg is None or (isinstance(deg, float) and math.isnan(deg)) or deg == -999:
        return 0
    # 22.5도 단위로 구간 나누기
    index = math.floor(((float(deg) + 22.5) % 360) / 45)
    return int(index)


def find_matching_column(feature_name: str, columns: pd.Index) -> str | None:
    """데이터프레임 컬럼 목록 중 feature_name과 가장 잘 매칭되는 컬럼명을 반환한다.

    * exact match가 있으면 그대로 반환
    * 대소문자를 구분하지 않는 매칭을 시도한다
    * `_0h` 접미사를 제거한 뒤 매칭을 시도한다
    일치하는 컬럼이 없으면 None을 반환한다.
    """
    # 1. 정확히 일치
    if feature_name in columns:
        return feature_name
    # 2. 대소문자 무시한 매칭
    lower_map = {col.lower(): col for col in columns}
    if feature_name.lower() in lower_map:
        return lower_map[feature_name.lower()]
    # 3. `_0h` 제거 후 매칭
    name_no_0h = feature_name.replace("_0h", "")
    if name_no_0h in columns:
        return name_no_0h
    if name_no_0h.lower() in lower_map:
        return lower_map[name_no_0h.lower()]
    return None


class EnsembleClassifier:
    """여러 분류기를 앙상블하여 다수결로 예측하는 클래스."""

    def __init__(self, models: list):
        self.models = models

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = np.column_stack([model.predict(X) for model in self.models])
        return np.apply_along_axis(lambda arr: np.bincount(arr).argmax(), axis=1, arr=predictions)


def load_feature_list(path: str) -> list[str] | None:
    """JSON 파일에서 feature 목록을 불러온다. 존재하지 않으면 None을 반환."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def build_feature_map(features_expected: list[str] | None, df: pd.DataFrame) -> list[tuple[str, str]]:
    """
    예상하는 feature 목록과 실제 데이터프레임 컬럼을 매핑한다.

    반환되는 리스트는 [(feature_name, matched_column), ...] 형태이며, matched_column이 None이면 해당 feature는 사용할 수 없다.
    만약 features_expected가 None이면 데이터프레임에서 숫자형 컬럼을 모두 선택한다.
    """
    feature_map: list[tuple[str, str]] = []
    if features_expected:
        for feat in features_expected:
            match = find_matching_column(feat, df.columns)
            if match is not None:
                feature_map.append((feat, match))
            else:
                # 일치하는 컬럼이 없으면 None을 매핑하여 나중에 0으로 채운다
                feature_map.append((feat, None))
    else:
        # 전체 수치형 컬럼을 사용하되, 몇몇 타깃 변수는 제외한다
        exclude = {"fire_area", "fire_duration_hours", "spread_speed", "actual_spread_speed_class", "actual_spread_direction"}
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in exclude:
                feature_map.append((col, col))
    return feature_map


def build_X(df: pd.DataFrame, feature_map: list[tuple[str, str]]) -> pd.DataFrame:
    """feature_map을 이용해 모델 입력 데이터를 생성한다."""
    X = pd.DataFrame(index=df.index)
    for feat_name, match in feature_map:
        if match is not None:
            X[feat_name] = df[match]
        else:
            # 해당 feature가 존재하지 않으면 0으로 채우기
            X[feat_name] = 0
    return X


def main() -> None:
    # 현재 스크립트와 동일한 디렉터리 기준으로 경로를 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_path = os.path.join(script_dir, "final_merged_feature_engineered.csv")
    # 모델과 scaler 저장 경로. 존재하지 않으면 생성.
    model_dir = os.path.join(script_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    print(f"데이터 로딩: {source_path}")
    df = pd.read_csv(source_path)

    # 특징 목록 JSON 파일 읽기 (선택적)
    speed_features_file = os.path.join(script_dir, "speed_model_columns.json")
    direction_features_file = os.path.join(script_dir, "direction_model_columns.json")

    speed_features_expected = load_feature_list(speed_features_file)
    direction_features_expected = load_feature_list(direction_features_file)

    # 널 값 제거. 분석에 사용할 필수 컬럼들을 지정한다. 없으면 전체 숫자형 사용.
    # 필수 컬럼만 결측치를 제거한다. 속도 계산에 필요한 fire_area, fire_duration_hours,
    # 방향 계산에 필요한 WD10M_0h 만 dropna 대상으로 지정하여, 예측 특성의 NaN은 이후 0으로 채운다.
    essential_cols = ["fire_area", "fire_duration_hours", "WD10M_0h"]
    df.dropna(subset=essential_cols, inplace=True)

    # ---------- 확산 속도 모델 ----------
    print("\n[확산 속도 모델]")
    # spread_speed 및 클래스 정의
    df["spread_speed"] = df.apply(lambda row: row["fire_area"] / row["fire_duration_hours"] if row["fire_duration_hours"] > 0 else 0, axis=1)
    df["actual_spread_speed_class"] = df["spread_speed"].apply(classify_speed)

    # feature 매핑 생성
    speed_feature_map = build_feature_map(speed_features_expected, df)
    X_speed = build_X(df, speed_feature_map)
    y_speed = df["actual_spread_speed_class"]

    # 결측치 처리: 현재 build_X에서 결측치를 0으로 채웠지만, 실제 값이 NaN인 경우도 있으므로 다시 채운다
    X_speed = X_speed.fillna(0)
    y_speed = y_speed.fillna(0)

    # 스케일러 학습
    scaler_speed = RobustScaler()
    X_speed_scaled = scaler_speed.fit_transform(X_speed)
    # 스케일된 DataFrame으로 변환
    X_speed_scaled = pd.DataFrame(X_speed_scaled, columns=X_speed.columns, index=X_speed.index)
    joblib.dump(scaler_speed, os.path.join(model_dir, "speed_scaler.joblib"))
    print("  - 속도 모델 스케일러 저장 완료")

    # 학습/테스트 분리
    X_train_speed, X_test_speed, y_train_speed, y_test_speed = train_test_split(
        X_speed_scaled, y_speed, test_size=0.2, random_state=42, stratify=y_speed
    )

    # 클래스 불균형에 대해 SMOTE 적용 (선택 사항)
    if SMOTE is not None:
        print(f"  - SMOTE 적용 전 속도 클래스 분포: {Counter(y_train_speed)}")
        smote_speed = SMOTE(random_state=42)
        X_train_speed, y_train_speed = smote_speed.fit_resample(X_train_speed, y_train_speed)
        print(f"  - SMOTE 적용 후 속도 클래스 분포: {Counter(y_train_speed)}")
    else:
        print("  - imblearn이 설치되어 있지 않아 SMOTE를 적용하지 않습니다.")

    # 개별 모델 학습
    models_speed = []
    # XGBoost
    if xgb is not None:
        models_speed.append(xgb.XGBClassifier(random_state=42, n_estimators=200, learning_rate=0.1, max_depth=5, n_jobs=-1, use_label_encoder=False, eval_metric="mlogloss"))
    # RandomForest
    models_speed.append(RandomForestClassifier(random_state=42, n_estimators=300, max_depth=None, n_jobs=-1))
    # GradientBoosting
    models_speed.append(GradientBoostingClassifier(random_state=42))
    # LightGBM
    if lgb is not None:
        # LightGBM 출력 억제를 위해 verbosity=-1 설정
        models_speed.append(lgb.LGBMClassifier(random_state=42,
                                              n_estimators=200,
                                              learning_rate=0.1,
                                              max_depth=-1,
                                              n_jobs=-1,
                                              verbosity=-1))
    # CatBoost
    if CatBoostClassifier is not None:
        models_speed.append(CatBoostClassifier(random_state=42, depth=6, learning_rate=0.1, n_estimators=200, verbose=0, allow_writing_files=False))

    # 모델 학습
    for model in models_speed:
        model.fit(X_train_speed, y_train_speed)

    # 앙상블 생성
    speed_model = EnsembleClassifier(models_speed)
    # 예측 및 평가
    y_pred_speed = speed_model.predict(X_test_speed)
    print("  - 속도 모델 정확도: {:.4f}".format(accuracy_score(y_test_speed, y_pred_speed)))
    print("  - 속도 모델 분류 리포트:\n", classification_report(y_test_speed, y_pred_speed, zero_division=0))

    # 모델 저장
    joblib.dump(speed_model, os.path.join(model_dir, "speed_classifier_model.joblib"))
    print("  ✅ 확산 속도 모델 저장 완료")

    # ---------- 확산 방향 모델 ----------
    print("\n[확산 방향 모델]")
    # 풍향을 8방위 클래스로 변환
    df["actual_spread_direction"] = df["WD10M_0h"].apply(convert_degree_to_direction)
    y_direction = df["actual_spread_direction"]

    # 방향 특성 계산: 기존 코드와 동일하게 시간 간격 별 변화량 및 평균/표준편차 계산
    time_intervals = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42]
    new_direction_features: list[str] = []

    # 변화량 특성 생성
    for i in range(len(time_intervals) - 1):
        t_start = time_intervals[i]
        t_end = time_intervals[i + 1]
        # 풍향 변화
        col1_wd = f"WD10M_{t_start}h"
        col2_wd = f"WD10M_{t_end}h"
        new_col_name_wd = f"wind_direction_change_{t_start}_{t_end}h"
        if col1_wd in df.columns and col2_wd in df.columns:
            df[new_col_name_wd] = (df[col1_wd] - df[col2_wd]).abs().apply(lambda x: min(x, 360 - x))
            new_direction_features.append(new_col_name_wd)
        # 풍속 변화
        col1_ws = f"WS10M_{t_start}h"
        col2_ws = f"WS10M_{t_end}h"
        new_col_name_ws = f"wind_speed_change_{t_start}_{t_end}h"
        if col1_ws in df.columns and col2_ws in df.columns:
            df[new_col_name_ws] = (df[col1_ws] - df[col2_ws]).abs()
            new_direction_features.append(new_col_name_ws)
        # 온도 변화
        col1_t = f"T2M_{t_start}h"
        col2_t = f"T2M_{t_end}h"
        new_col_name_t = f"temp_change_{t_start}_{t_end}h"
        if col1_t in df.columns and col2_t in df.columns:
            df[new_col_name_t] = (df[col1_t] - df[col2_t]).abs()
            new_direction_features.append(new_col_name_t)
        # 습도 변화
        col1_rh = f"RH2M_{t_start}h"
        col2_rh = f"RH2M_{t_end}h"
        new_col_name_rh = f"humidity_change_{t_start}_{t_end}h"
        if col1_rh in df.columns and col2_rh in df.columns:
            df[new_col_name_rh] = (df[col1_rh] - df[col2_rh]).abs()
            new_direction_features.append(new_col_name_rh)

    # 평균/표준편차 특성 생성
    weather_params = ["T2M", "RH2M", "WS10M", "WD10M", "PRECTOTCORR", "PS", "ALLSKY_SFC_SW_DWN"]
    for param in weather_params:
        cols_for_param = [f"{param}_{interval}h" for interval in time_intervals if f"{param}_{interval}h" in df.columns]
        if cols_for_param:
            mean_name = f"{param}_mean"
            std_name = f"{param}_std"
            df[mean_name] = df[cols_for_param].mean(axis=1)
            df[std_name] = df[cols_for_param].std(axis=1)
            new_direction_features.extend([mean_name, std_name])

    # 방향 feature 목록 합치기
    direction_features_complete: list[str] = []
    if direction_features_expected:
        direction_features_complete.extend(direction_features_expected)
    direction_features_complete.extend(new_direction_features)
    # 중복 제거
    direction_features_complete = list(dict.fromkeys(direction_features_complete))

    # feature 매핑 생성
    direction_feature_map = build_feature_map(direction_features_complete, df)
    X_direction = build_X(df, direction_feature_map)
    X_direction = X_direction.fillna(0)

    # 스케일링
    scaler_direction = RobustScaler()
    X_direction_scaled = scaler_direction.fit_transform(X_direction)
    X_direction_scaled = pd.DataFrame(X_direction_scaled, columns=X_direction.columns, index=X_direction.index)
    joblib.dump(scaler_direction, os.path.join(model_dir, "direction_scaler.joblib"))
    print("  - 방향 모델 스케일러 저장 완료")

    # 학습/테스트 분리
    X_train_dir, X_test_dir, y_train_dir, y_test_dir = train_test_split(
        X_direction_scaled, y_direction, test_size=0.2, random_state=42, stratify=y_direction
    )

    # SMOTE 적용
    if SMOTE is not None:
        print(f"  - SMOTE 적용 전 방향 클래스 분포: {Counter(y_train_dir)}")
        # SMOTE 기본 설정: 각 클래스 최소 샘플 수보다 k_neighbors를 작게 설정
        min_samples = min(Counter(y_train_dir).values())
        k_neighbors = max(1, min_samples - 1)
        smote_dir = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_dir, y_train_dir = smote_dir.fit_resample(X_train_dir, y_train_dir)
        print(f"  - SMOTE 적용 후 방향 클래스 분포: {Counter(y_train_dir)}")
    else:
        print("  - imblearn이 설치되어 있지 않아 SMOTE를 적용하지 않습니다.")

    # 개별 모델 학습
    models_dir: list = []
    if xgb is not None:
        models_dir.append(xgb.XGBClassifier(random_state=42, n_estimators=200, learning_rate=0.1, max_depth=5, n_jobs=-1, use_label_encoder=False, eval_metric="mlogloss"))
    models_dir.append(RandomForestClassifier(random_state=42, n_estimators=300, max_depth=None, n_jobs=-1))
    models_dir.append(GradientBoostingClassifier(random_state=42))
    if lgb is not None:
        models_dir.append(lgb.LGBMClassifier(random_state=42,
                                            n_estimators=200,
                                            learning_rate=0.1,
                                            max_depth=-1,
                                            n_jobs=-1,
                                            verbosity=-1))
    if CatBoostClassifier is not None:
        models_dir.append(CatBoostClassifier(random_state=42, depth=6, learning_rate=0.1, n_estimators=200, verbose=0, allow_writing_files=False))

    for model in models_dir:
        model.fit(X_train_dir, y_train_dir)

    direction_model = EnsembleClassifier(models_dir)
    y_pred_dir = direction_model.predict(X_test_dir)
    print("  - 방향 모델 정확도: {:.4f}".format(accuracy_score(y_test_dir, y_pred_dir)))
    print("  - 방향 모델 분류 리포트:\n", classification_report(y_test_dir, y_pred_dir, zero_division=0))

    joblib.dump(direction_model, os.path.join(model_dir, "direction_classifier_model.joblib"))
    print("  ✅ 확산 방향 모델 저장 완료")


if __name__ == "__main__":
    main()