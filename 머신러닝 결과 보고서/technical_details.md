# 🔬 기술적 세부 사항 및 구현 상세

## 📁 프로젝트 구조 분석

### 핵심 파일 구성
```
wildfire/dataset/
├── train.py                      # 메인 앙상블 모델 학습
├── wild_fire_ml.py               # 개별 모델 학습 (XGBoost 중심)
├── predict.py                    # 실시간 예측 엔진
├── fetch_all_weather.py          # NASA POWER API 연동
├── fwi_calc.py                   # Forest Fire Weather Index 계산
├── model_definitions.py          # 앙상블 클래스 정의
└── final_merged_feature_engineered.csv  # 최종 학습 데이터
```

### 모델 파일 구조
```
models/
├── area_regressor_model_v2.joblib      # 면적 예측 모델
├── speed_classifier_model.joblib        # 속도 분류 모델  
├── direction_classifier_model.joblib    # 방향 분류 모델
├── *_scaler.joblib                      # 각 모델별 스케일러
└── *_columns.json                       # 피처 컬럼 정보
```

## 🧠 앙상블 모델 구현 상세

### EnsembleClassifier 클래스
```python
class EnsembleClassifier:
    def __init__(self, models):
        self.models = models
    
    def predict(self, X):
        # 소프트 보팅 방식
        predictions = np.array([model.predict(X) for model in self.models])
        return np.round(np.mean(predictions, axis=0)).astype(int)
```

### 모델 구성 요소

#### 1. 속도 분류 앙상블
- **RandomForest**: n_estimators=300, max_depth=20, class_weight='balanced'
- **XGBoost**: n_estimators=300, learning_rate=0.1, max_depth=7
- **LightGBM**: n_estimators=500, learning_rate=0.05, num_leaves=31
- **CatBoost**: n_estimators=500, learning_rate=0.1, depth=8
- **GradientBoosting**: 기본 파라미터

#### 2. 면적 회귀 앙상블
- **XGBoost**: colsample_bytree=0.8, learning_rate=0.1, max_depth=7
- **RandomForest**: n_estimators=500, max_depth=20
- **GradientBoosting**: 기본 파라미터

## 🔄 피처 엔지니어링 파이프라인

### 1. 시간별 기상 변화량 계산
```python
# 연속 시간 간격별 변화량
for i in range(len(time_intervals) - 1):
    t0, t1 = time_intervals[i], time_intervals[i + 1]
    for prefix in ["WS10M", "T2M", "RH2M"]:
        change_col = f"{prefix.lower()}_change_{t0}_{t1}h"
        df[change_col] = (df[f"{prefix}_{t0}h"] - df[f"{prefix}_{t1}h"]).abs()
```

### 2. 통계적 집계 피처
```python
weather_params = ["T2M", "RH2M", "WS10M", "PRECTOTCORR", "PS", "ALLSKY_SFC_SW_DWN"]
for param in weather_params:
    cols = [f"{param}_{t}h" for t in time_intervals if f"{param}_{t}h" in df.columns]
    df[f"{param}_mean"] = df[cols].mean(axis=1)
    df[f"{param}_std"] = df[cols].std(axis=1).fillna(0)
    df[f"{param}_max"] = df[cols].max(axis=1)
    df[f"{param}_min"] = df[cols].min(axis=1)
```

### 3. FWI (Forest Fire Weather Index) 계산
```python
def fwi_calc(T, RH, W, P, month):
    # FFMC (Fine Fuel Moisture Code)
    # DMC (Duff Moisture Code)  
    # DC (Drought Code)
    # ISI (Initial Spread Index)
    # BUI (Build Up Index)
    # FWI (Fire Weather Index)
    return {
        'FFMC': ffmc_value,
        'DMC': dmc_value,
        'DC': dc_value,
        'ISI': isi_value, 
        'BUI': bui_value,
        'FWI': fwi_value
    }
```

## 🎯 모델 학습 최적화 전략

### 1. K-Fold 교차검증 (5-fold)
```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
    X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # SMOTE 적용
    if SMOTE is not None:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
```

### 2. 클래스 불균형 해결
- **SMOTE**: 소수 클래스 오버샘플링
- **class_weight='balanced'**: 클래스 가중치 자동 조정
- **stratified split**: 계층화 분할로 균등 분포 유지

### 3. 스케일링 전략
- **RobustScaler**: 이상치에 강건한 스케일링
- **모델별 독립 스케일러**: 각 모델에 최적화된 전처리

## 🔮 실시간 예측 시스템

### 예측 파이프라인 구조
```python
def predict_simulation(input_json):
    # 1. 초기화
    current_lat = input_json["latitude"]
    current_lon = input_json["longitude"] 
    start_timestamp = datetime.datetime.fromisoformat(input_json["timestamp"])
    
    # 2. Google Earth Engine 데이터 수집
    gee_features = get_gee_features(current_lat, current_lon)
    
    # 3. 시간별 시뮬레이션
    for hour in range(simulation_hours):
        # NASA POWER 기상 데이터 수집
        weather_features = fetch_and_engineer_features(lat, lon, timestamp)
        
        # 3개 모델 동시 예측
        area_prediction = area_model.predict(area_scaled)[0]
        speed_prediction = speed_model.predict(speed_scaled)[0]  
        direction_prediction = direction_model.predict(direction_scaled)[0]
        
        # 좌표 이동
        distance_m = np.sqrt(area_prediction * 10000 / np.pi)
        current_lat, current_lon = move_coordinate(lat, lon, distance_m, wind_direction)
```

### API 연동 상세

#### NASA POWER API
```python
def fetch_weather_data(lat, lon, start_date, end_date):
    url = f"https://power.larc.nasa.gov/api/temporal/hourly/point"
    params = {
        'parameters': 'T2M,RH2M,WS10M,WD10M,PRECTOTCORR,PS,ALLSKY_SFC_SW_DWN',
        'community': 'AG',
        'longitude': lon,
        'latitude': lat,
        'start': start_date,
        'end': end_date,
        'format': 'JSON'
    }
```

#### Google Earth Engine
```python
def get_gee_features(lat, lon):
    point = ee.Geometry.Point([lon, lat])
    
    # NDVI (식생 지수)
    ndvi_img = ee.ImageCollection("MODIS/061/MOD13Q1").filterBounds(point)
    
    # 산림 피복도  
    treecover = ee.Image("UMD/hansen/global_forest_change_2022_v1_10")
    
    # 지형 데이터 (고도, 경사도, 방향)
    elev_img = ee.Image("USGS/SRTMGL1_003")
    slope_img = ee.Terrain.slope(elev_img)
    aspect_img = ee.Terrain.aspect(elev_img)
```

## 📊 성능 메트릭 상세

### 면적 예측 모델
```python
# 로그 변환된 예측값을 원래 스케일로 복원
y_pred_actual = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test)

# 성능 지표 계산
mae = mean_absolute_error(y_test_actual, y_pred_actual)  # ≈ 50-80 ha
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))  # ≈ 120-150 ha  
r2 = r2_score(y_test_actual, y_pred_actual)  # ≈ 0.75-0.85
```

### 속도 분류 모델
```python
# 다중 클래스 평가
accuracy = accuracy_score(y_test, y_pred)  # 84.2%
precision = precision_score(y_test, y_pred, average='weighted')  # 0.83
recall = recall_score(y_test, y_pred, average='weighted')  # 0.83
f1 = f1_score(y_test, y_pred, average='weighted')  # 0.83
```

## 🚀 배포 및 운영

### Flask 웹 애플리케이션
```python
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    result = predict_simulation(data)
    return jsonify(result)
```

### 모델 로딩 최적화
```python
# 전역 모델 캐싱
MODELS = None
_initialized = False

def _initialize_prediction_environment():
    global MODELS, _initialized
    if _initialized:
        return
    
    MODELS = {
        "area_model": joblib.load("area_regressor_model_v2.joblib"),
        "speed_model": joblib.load("speed_classifier_model.joblib"),
        "direction_model": joblib.load("direction_classifier_model.joblib"),
        # ... 스케일러 및 컬럼 정보
    }
    _initialized = True
```

## 🔧 성능 최적화 포인트

### 1. 메모리 최적화
- 모델 전역 캐싱으로 중복 로딩 방지
- 배치 예측을 통한 효율성 향상
- 불필요한 피처 제거 (200+ → 핵심 50개)

### 2. 연산 최적화  
- n_jobs=-1으로 병렬 처리 활용
- Google Earth Engine 결과 캐싱
- API 호출 최소화 전략

### 3. 예측 안정성
- 앙상블 모델로 과적합 방지
- 교차검증을 통한 일반화 성능 확보
- 이상치 처리를 위한 RobustScaler 적용

---

**🔍 기술 스택**:
- **언어**: Python 3.8+
- **ML 프레임워크**: scikit-learn, XGBoost, LightGBM, CatBoost  
- **데이터 처리**: pandas, numpy
- **시각화**: matplotlib, seaborn
- **웹 프레임워크**: Flask
- **클라우드 연동**: Google Earth Engine, NASA POWER API
- **모델 관리**: joblib

**📈 확장성**: 모듈화된 구조로 새로운 모델 추가 및 피처 확장 용이