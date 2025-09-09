from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import os
import json
import math
from collections import Counter

from model_definitions import EnsembleClassifier, EnsembleRegressor

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor
)

try:
    import xgboost as xgb
except ImportError:
    xgb = None
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

from fwi_calc import fwi_calc


def classify_speed(speed: float, thresholds=(0.014, 0.11)) -> int:
    low, high = thresholds
    if speed <= low:
        return 0
    if speed <= high:
        return 1
    return 2


def convert_degree_to_direction(deg: float) -> int:
    if deg is None or (isinstance(deg, float) and math.isnan(deg)) or deg == -999:
        return 0
    return int(math.floor(((float(deg) + 22.5) % 360) / 45))


def find_matching_column(feature_name: str, columns: pd.Index) -> str | None:
    if feature_name in columns:
        return feature_name
    lower_map = {col.lower(): col for col in columns}
    if feature_name.lower() in lower_map:
        return lower_map[feature_name.lower()]
    name_no_0h = feature_name.replace("_0h", "")
    if name_no_0h in columns:
        return name_no_0h
    if name_no_0h.lower() in lower_map:
        return lower_map[name_no_0h.lower()]
    return None


def load_feature_list(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def build_feature_map(features_expected, df):
    feature_map = []
    if features_expected:
        for feat in features_expected:
            match = find_matching_column(feat, df.columns)
            feature_map.append((feat, match))
    else:
        exclude = {"fire_area", "fire_duration_hours",
                   "spread_speed", "actual_spread_speed_class",
                   "actual_spread_direction"}
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in exclude:
                feature_map.append((col, col))
    return feature_map


def build_X(df, feature_map):
    data = {}
    for feat_name, match in feature_map:
        data[feat_name] = df[match] if match is not None else 0
    return pd.DataFrame(data, index=df.index)


def engineer_features(df):
    """🔧 예측 시와 호환되는 피처 생성 함수"""
    print("새로운 파생변수 생성 중...")
    new_features = []

    # 1. 시간 간격 찾기
    time_intervals = []
    for c in df.columns:
        if c.startswith('WS10M_') and c.endswith('h'):
            try:
                hour = int(c.split('_')[1].replace('h', ''))
                time_intervals.append(hour)
            except ValueError:
                continue
    time_intervals = sorted(list(set(time_intervals)))

    print(f"발견된 시간 간격: {time_intervals}")

    # 2. 시간적 변화량 피처 (예측 시와 동일)
    for i in range(len(time_intervals) - 1):
        t0, t1 = time_intervals[i], time_intervals[i + 1]
        for prefix in ["WS10M", "T2M", "RH2M"]:
            c1, c2 = f"{prefix}_{t0}h", f"{prefix}_{t1}h"
            new_col = f"{prefix.lower()}_change_{t0}_{t1}h"
            if c1 in df.columns and c2 in df.columns:
                df[new_col] = (df[c1] - df[c2]).abs()
                new_features.append(new_col)
                print(f"   생성: {new_col}")

    # 3. 🔧 핵심 수정: _mean_all 대신 _mean 등 단순한 접미사 사용
    weather_params = ["T2M", "RH2M", "WS10M", "PRECTOTCORR", "PS", "ALLSKY_SFC_SW_DWN"]
    for param in weather_params:
        cols = [f"{param}_{t}h" for t in time_intervals if f"{param}_{t}h" in df.columns]
        if cols:
            # 🔧 예측 시와 동일한 접미사 사용
            df[f"{param}_mean"] = df[cols].mean(axis=1)
            df[f"{param}_std"] = df[cols].std(axis=1).fillna(0)
            df[f"{param}_max"] = df[cols].max(axis=1)
            df[f"{param}_min"] = df[cols].min(axis=1)

            new_features.extend([f"{param}_mean", f"{param}_std", f"{param}_max", f"{param}_min"])
            print(f"   생성: {param} 통계 피처들 (_mean, _std, _max, _min)")

    # 4. 조합 피처 (접미사 수정)
    if "T2M_mean" in df.columns and "RH2M_mean" in df.columns and "WS10M_mean" in df.columns:
        df["dryness_index"] = df["T2M_mean"] * (100 - df["RH2M_mean"]) / 100
        df["wind_humidity_ratio"] = df["WS10M_mean"] / (df["RH2M_mean"] + 1e-5)
        df["wind_temp_product"] = df["WS10M_mean"] * df["T2M_mean"]
        new_features.extend(["dryness_index", "wind_humidity_ratio", "wind_temp_product"])
        print("   생성: 조합 피처들")

    # 5. FWI 계산 (기존과 동일)
    fwi_inputs = {
        'T': 'T2M_0h',
        'RH': 'RH2M_0h',
        'W': 'WS10M_0h',
        'P': 'PRECTOTCORR_0h',
        'month': 'month'
    }
    if all(col in df.columns for col in fwi_inputs.values()):
        print("   FWI 계산 중...")
        fwi_results = df.apply(
            lambda row: fwi_calc(
                T=row[fwi_inputs['T']],
                RH=row[fwi_inputs['RH']],
                W=row[fwi_inputs['W']],
                P=row[fwi_inputs['P']],
                month=row[fwi_inputs['month']]
            ),
            axis=1
        )
        fwi_df = pd.DataFrame(fwi_results.tolist(), index=df.index)
        for col in fwi_df.columns:
            df[col] = fwi_df[col]
            new_features.append(col)

    print(f"총 {len(new_features)}개의 새로운 피처 생성됨")
    return new_features


def create_core_features_if_missing(df):
    """핵심 피처가 없을 경우 기본값으로 생성"""
    core_features_to_check = [
        'T2M_mean', 'RH2M_mean', 'WS10M_mean', 'PRECTOTCORR_mean',
        'PS_mean', 'ALLSKY_SFC_SW_DWN_mean',
        'T2M_std', 'RH2M_std', 'WS10M_std', 'PRECTOTCORR_std',
        'PS_std', 'ALLSKY_SFC_SW_DWN_std',
        'T2M_max', 'RH2M_max', 'WS10M_max', 'PRECTOTCORR_max',
        'PS_max', 'ALLSKY_SFC_SW_DWN_max',
        'T2M_min', 'RH2M_min', 'WS10M_min', 'PRECTOTCORR_min',
        'PS_min', 'ALLSKY_SFC_SW_DWN_min'
    ]

    created_features = []
    for feat in core_features_to_check:
        if feat not in df.columns:
            # 기본값 설정
            if '_mean' in feat or '_max' in feat or '_min' in feat:
                if 'T2M' in feat:
                    df[feat] = 20.0
                elif 'RH2M' in feat:
                    df[feat] = 50.0
                elif 'WS10M' in feat:
                    df[feat] = 2.0
                elif 'PRECTOTCORR' in feat:
                    df[feat] = 0.0
                elif 'PS' in feat:
                    df[feat] = 1013.0
                elif 'ALLSKY_SFC_SW_DWN' in feat:
                    df[feat] = 200.0
                else:
                    df[feat] = 0.0
            elif '_std' in feat:
                df[feat] = 1.0  # 표준편차는 1.0으로
            else:
                df[feat] = 0.0
            created_features.append(feat)

    if created_features:
        print(f"   기본값으로 생성된 피처: {len(created_features)}개")

    return created_features


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "final_merged_feature_engineered.csv")
    model_dir = os.path.join(script_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    print(f"데이터 로딩: {data_path}")
    df = pd.read_csv(data_path)
    df.dropna(subset=["fire_area", "fire_duration_hours", "WD10M_0h"], inplace=True)

    print(f"데이터 크기: {df.shape}")
    print(f"컬럼 샘플: {list(df.columns[:10])}")

    # 기본 타겟 변수 생성
    df["spread_speed"] = df.apply(
        lambda row: row["fire_area"] / row["fire_duration_hours"]
        if row["fire_duration_hours"] > 0 else 0,
        axis=1
    )
    df["actual_spread_speed_class"] = df["spread_speed"].apply(classify_speed)

    # 새 파생 변수 생성
    newly_engineered_features = engineer_features(df)

    # 🔧 추가: 핵심 피처가 누락된 경우 기본값으로 생성
    core_created_features = create_core_features_if_missing(df)
    newly_engineered_features.extend(core_created_features)

    print(f"\n생성된 파생 피처 샘플: {newly_engineered_features[:10]}")

    # 속도 모델 학습
    print("\n[속도 모델 K-Fold 교차검증 시작]")

    # 기존 피처 목록 로드 (있다면)
    speed_features_expected = load_feature_list(os.path.join(script_dir, "speed_model_columns.json"))

    if speed_features_expected is not None:
        print(f"기존 피처 목록 로드됨: {len(speed_features_expected)}개")
        speed_features_expected.extend(newly_engineered_features)
        speed_features_expected = list(dict.fromkeys(speed_features_expected))  # 중복 제거
    else:
        # 🔧 핵심: 기본 피처 세트 정의
        base_features = [
            # 기본 위치/시간 정보
            'lat', 'lng', 'duration_hours', 'total_duration_hours',
            'startyear', 'month', 'startday',

            # 현재 기상 조건
            'T2M_0h', 'RH2M_0h', 'WS10M_0h', 'WD10M_0h', 'PRECTOTCORR_0h', 'PS_0h', 'ALLSKY_SFC_SW_DWN_0h',

            # FWI 관련
            'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI',

            # 계절 정보
            'is_spring', 'is_summer', 'is_autumn', 'is_winter',

            # 강수량 관련
            'total_precip_7d_start', 'total_precip_14d_start', 'total_precip_30d_start',
            'dry_days_7d_start', 'dry_days_14d_start', 'dry_days_30d_start',
            'consecutive_dry_days_start',

            # 추가 지표
            'dry_windy_combo', 'fuel_combo', 'potential_spread_index',
            'wind_steady_flag', 'dry_to_rain_ratio_30d', 'ndvi_stress', 'terrain_var_effect'
        ]
        speed_features_expected = base_features + newly_engineered_features
        speed_features_expected = list(dict.fromkeys(speed_features_expected))

    print(f"최종 속도 모델 피처 수: {len(speed_features_expected)}")

    # 피처 매핑 및 데이터 준비
    speed_feature_map = build_feature_map(speed_features_expected, df)
    X_speed = build_X(df, speed_feature_map).fillna(0)
    y_speed = df["actual_spread_speed_class"]

    print(f"속도 모델 입력 데이터 크기: {X_speed.shape}")
    print(f"클래스 분포: {y_speed.value_counts().to_dict()}")

    # 스케일링
    scaler_s = RobustScaler()
    X_speed_scaled = pd.DataFrame(
        scaler_s.fit_transform(X_speed),
        columns=X_speed.columns, index=df.index
    )

    # K-Fold 교차검증
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    acc_rf_list, acc_gb_list, acc_xgb_list = [], [], []
    acc_lgb_list, acc_cat_list, acc_ens_list = [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_speed_scaled)):
        print(f"\n📂 Fold {fold + 1}")
        X_train, X_val = X_speed_scaled.iloc[train_idx], X_speed_scaled.iloc[val_idx]
        y_train, y_val = y_speed.iloc[train_idx], y_speed.iloc[val_idx]

        if SMOTE is not None:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        models = []

        # RandomForest
        rf = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=300, max_depth=20, class_weight='balanced')
        rf.fit(X_train, y_train)
        acc_rf = accuracy_score(y_val, rf.predict(X_val))
        acc_rf_list.append(acc_rf)
        print(f"  - RF Accuracy: {acc_rf:.4f}")
        models.append(rf)

        # GradientBoosting
        gb = GradientBoostingClassifier(random_state=42)
        gb.fit(X_train, y_train)
        acc_gb = accuracy_score(y_val, gb.predict(X_val))
        acc_gb_list.append(acc_gb)
        print(f"  - GB Accuracy: {acc_gb:.4f}")
        models.append(gb)

        # XGBoost
        if xgb is not None:
            xg = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss',
                                   n_estimators=300, learning_rate=0.05, max_depth=7, n_jobs=-1)
            xg.fit(X_train, y_train)
            acc_x = accuracy_score(y_val, xg.predict(X_val))
            acc_xgb_list.append(acc_x)
            print(f"  - XGB Accuracy: {acc_x:.4f}")
            models.append(xg)

        # LightGBM
        if lgb is not None:
            lg = lgb.LGBMClassifier(random_state=42, n_estimators=300, learning_rate=0.05,
                                    num_leaves=31, class_weight="balanced", n_jobs=-1, verbosity=-1)
            lg.fit(X_train, y_train)
            acc_l = accuracy_score(y_val, lg.predict(X_val))
            acc_lgb_list.append(acc_l)
            print(f"  - LGB Accuracy: {acc_l:.4f}")
            models.append(lg)

        # CatBoost
        if CatBoostClassifier is not None:
            ct = CatBoostClassifier(random_state=42, verbose=0, allow_writing_files=False,
                                    n_estimators=300, learning_rate=0.05, depth=6)
            ct.fit(X_train, y_train)
            acc_c = accuracy_score(y_val, ct.predict(X_val))
            acc_cat_list.append(acc_c)
            print(f"  - CAT Accuracy: {acc_c:.4f}")
            models.append(ct)

        # 앙상블
        ens = EnsembleClassifier(models)
        acc_ens = accuracy_score(y_val, ens.predict(X_val))
        acc_ens_list.append(acc_ens)
        print(f"  - Ensemble Accuracy: {acc_ens:.4f}")

    print("\n📊 평균 정확도 요약")
    print(f"  - RF   평균 정확도: {np.mean(acc_rf_list):.4f}")
    print(f"  - GB   평균 정확도: {np.mean(acc_gb_list):.4f}")
    if acc_xgb_list: print(f"  - XGB  평균 정확도: {np.mean(acc_xgb_list):.4f}")
    if acc_lgb_list: print(f"  - LGB  평균 정확도: {np.mean(acc_lgb_list):.4f}")
    if acc_cat_list: print(f"  - CAT  평균 정확도: {np.mean(acc_cat_list):.4f}")
    print(f"  - 앙상블 평균 정확도: {np.mean(acc_ens_list):.4f}")

    # 최종 모델 학습
    print("\n[속도 모델 전체 학습 및 저장]")
    best_estimators = []

    if SMOTE is not None:
        smote = SMOTE(random_state=42)
        X_train_full, y_train_full = smote.fit_resample(X_speed_scaled, y_speed)
    else:
        X_train_full, y_train_full = X_speed_scaled, y_speed

    # 모델들 학습
    if xgb is not None:
        model_xgb = xgb.XGBClassifier(
            random_state=42, n_estimators=300, learning_rate=0.1, max_depth=7,
            n_jobs=-1, use_label_encoder=False, eval_metric="mlogloss"
        )
        model_xgb.fit(X_train_full, y_train_full)
        best_estimators.append(model_xgb)

    model_rf = RandomForestClassifier(
        random_state=42, n_estimators=500, max_depth=20,
        class_weight="balanced", n_jobs=-1
    )
    model_rf.fit(X_train_full, y_train_full)
    best_estimators.append(model_rf)

    model_gb = GradientBoostingClassifier(random_state=42)
    model_gb.fit(X_train_full, y_train_full)
    best_estimators.append(model_gb)

    if lgb is not None:
        model_lgb = lgb.LGBMClassifier(
            random_state=42, n_estimators=500, learning_rate=0.05,
            num_leaves=31, class_weight="balanced", n_jobs=-1, verbosity=-1
        )
        model_lgb.fit(X_train_full, y_train_full)
        best_estimators.append(model_lgb)

    if CatBoostClassifier is not None:
        model_cat = CatBoostClassifier(
            random_state=42, n_estimators=500, learning_rate=0.1,
            depth=8, verbose=0, allow_writing_files=False
        )
        model_cat.fit(X_train_full, y_train_full)
        best_estimators.append(model_cat)

    # 모델 저장
    ensemble_model = EnsembleClassifier(best_estimators)
    joblib.dump(ensemble_model, os.path.join(model_dir, "speed_classifier_model.joblib"))
    joblib.dump(scaler_s, os.path.join(model_dir, "speed_scaler.joblib"))

    with open(os.path.join(script_dir, "speed_model_columns.json"), 'w') as f:
        json.dump(X_speed.columns.tolist(), f)

    print("✅ 속도 앙상블 모델 및 컬럼 저장 완료")

    # 방향 모델 (간단화)
    print("\n[확산 방향 모델]")
    df["actual_spread_direction"] = df["WD10M_0h"].apply(convert_degree_to_direction)
    y_dir = df["actual_spread_direction"]

    # 방향 모델은 속도 모델과 같은 피처 사용
    X_dir = X_speed.copy()
    scaler_d = RobustScaler()
    X_dir_scaled = pd.DataFrame(
        scaler_d.fit_transform(X_dir),
        columns=X_dir.columns, index=X_dir.index
    )

    # 간단한 방향 모델들
    dir_models = [
        RandomForestClassifier(random_state=42, n_estimators=300, class_weight="balanced", n_jobs=-1),
        GradientBoostingClassifier(random_state=42)
    ]

    if xgb is not None:
        dir_models.append(xgb.XGBClassifier(
            random_state=42, n_estimators=300, learning_rate=0.05, max_depth=6,
            n_jobs=-1, use_label_encoder=False, eval_metric="mlogloss"
        ))

    for mdl in dir_models:
        mdl.fit(X_dir_scaled, y_dir)

    dir_ensemble = EnsembleClassifier(dir_models)
    joblib.dump(dir_ensemble, os.path.join(model_dir, "direction_classifier_model.joblib"))
    joblib.dump(scaler_d, os.path.join(model_dir, "direction_scaler.joblib"))

    with open(os.path.join(script_dir, "direction_model_columns.json"), 'w') as f:
        json.dump(X_dir.columns.tolist(), f)

    print("\n✅ 모든 모델 및 컬럼 저장 완료")
    print(f"\n📊 최종 결과:")
    print(f"   - 속도 모델 피처 수: {len(X_speed.columns)}")
    print(f"   - 방향 모델 피처 수: {len(X_dir.columns)}")
    print(f"   - 모델 저장 위치: {model_dir}")


if __name__ == "__main__":
    main()