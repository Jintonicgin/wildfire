import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import json
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
# --- 개선 제안 2: LightGBM 모델 추가 ---
import lightgbm as lgb

class ComprehensiveModelRetrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        # --- 개선 제안 3: 피처 엔지니어링을 위한 컬럼명 후보 ---
        self.date_col_candidates = ['start_date', 'date', 'timestamp']
        self.temp_col_candidates = ['temp', 't2m', 'temperature']
        self.rh_col_candidates = ['humidity', 'rh']
        self.duration_col_candidates = ['fire_duration_hours', 'duration']
        self.wind_dir_col_candidates = ['wind_direction', 'wd']
        
    def find_column(self, candidates):
        """데이터프레임에 존재하는 컬럼명을 후보군에서 찾습니다."""
        for col in candidates:
            if col in self.df.columns:
                return col
        return None

    def engineer_features(self):
        """--- 개선 제안 3: 피처 엔지니어링 (시간, 상호작용) ---"""
        print("  피처 엔지니어링 수행...")
        
        # 1. 시간 관련 피처 (월, 계절)
        date_col = self.find_column(self.date_col_candidates)
        if date_col:
            print(f"    시간 관련 피처 생성 (기준: '{date_col}')")
            self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')
            self.df['month'] = self.df[date_col].dt.month
            
            # 계절 매핑 (북반구 기준)
            season_map = {1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring', 
                          6: 'summer', 7: 'summer', 8: 'summer', 9: 'autumn', 10: 'autumn', 
                          11: 'autumn', 12: 'winter'}
            self.df['season'] = self.df['month'].map(season_map)
            # One-Hot Encoding으로 변환
            self.df = pd.get_dummies(self.df, columns=['season'], prefix='season')
            print("    'month', 'season' 피처 생성 완료.")
        else:
            print("    경고: 날짜 컬럼을 찾을 수 없어 시간 관련 피처를 생성하지 못했습니다.")

        # 2. 상호작용 피처 (온도 * 습도)
        temp_col = self.find_column(self.temp_col_candidates)
        rh_col = self.find_column(self.rh_col_candidates)
        
        if temp_col and rh_col:
            print(f"    상호작용 피처 생성 ('{temp_col}' * '{rh_col}')")
            # 습도는 0-100 사이의 값으로 가정. 0으로 나누는 것을 방지.
            self.df['temp_rh_interaction'] = self.df[temp_col] / (self.df[rh_col] + 1e-6)
            print("    'temp_rh_interaction' 피처 생성 완료.")
        else:
            print(f"    경고: 온도('{temp_col}') 또는 습도('{rh_col}') 컬럼을 찾을 수 없어 상호작용 피처를 생성하지 못했습니다.")

    def load_and_prepare_data(self):
        """데이터 로딩 및 기본 전처리"""
        print("📊 데이터 로딩 및 기본 전처리...")
        
        self.df = pd.read_csv(self.data_path)
        print(f"데이터 크기: {self.df.shape}")
        
        # --- 개선 제안 3: 피처 엔지니어링 호출 ---
        self.engineer_features()
        
        # 타겟 변수들 확인
        required_cols = ['fire_area', 'speed_category', 'direction_category']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            print(f"❌ 필수 컬럼 없음: {missing_cols}")
            
            if 'speed_category' not in self.df.columns:
                print("  속도 카테고리 생성 중...")
                self.df = self.create_speed_categories(self.df)
            
            if 'direction_category' not in self.df.columns:
                print("  방향 카테고리 생성 중...")
                self.df = self.create_direction_categories(self.df)
        
        print(f"속도 카테고리 분포:")
        if 'speed_category' in self.df.columns:
            print(self.df['speed_category'].value_counts().sort_index())
        
        print(f"방향 카테고리 분포:")
        if 'direction_category' in self.df.columns:
            print(self.df['direction_category'].value_counts().sort_index())
            
        return self.df
    
    def create_speed_categories(self, df):
        """--- 개선 제안 1: 속도 카테고리 정의 개선 ---"""
        print("    화재 확산 속도 카테고리 생성...")
        
        duration_col = self.find_column(self.duration_col_candidates)
        
        # 1순위: 시간당 피해 면적으로 속도 계산
        if duration_col and 'fire_area' in df.columns:
            print(f"    '{duration_col}' 컬럼을 사용하여 시간당 피해 면적으로 속도 추정...")
            # 0 또는 매우 작은 시간 값으로 나누는 것을 방지
            df['spread_rate'] = df['fire_area'] / (df[duration_col] + 1e-6)
            
            rate_non_zero = df['spread_rate'][df['spread_rate'] > 0]
            
            if len(rate_non_zero) > 0:
                q33 = rate_non_zero.quantile(0.33)
                q67 = rate_non_zero.quantile(0.67)
                
                def categorize_speed(rate):
                    if rate <= q33: return 'slow'
                    elif rate <= q67: return 'medium'
                    else: return 'fast'
                
                df['speed_category'] = df['spread_rate'].apply(categorize_speed)
                df = df.drop(columns=['spread_rate']) # 임시 컬럼 제거
            else:
                df['speed_category'] = 'slow'
        
        # 2순위: 기존 방식 (화재 면적 기반)
        else:
            print("    경고: 지속 시간 컬럼을 찾을 수 없습니다. 기존 방식(fire_area 기반)으로 속도 카테고리를 생성합니다.")
            area = df['fire_area'].copy()
            area_non_zero = area[area > 0]
            
            if len(area_non_zero) > 0:
                q33 = area_non_zero.quantile(0.33)
                q67 = area_non_zero.quantile(0.67)
                
                def categorize_by_area(area_val):
                    if area_val <= q33: return 'slow'
                    elif area_val <= q67: return 'medium'
                    else: return 'fast'
                
                df['speed_category'] = area.apply(categorize_by_area)
            else:
                df['speed_category'] = 'slow'
            
        return df

    def create_direction_categories(self, df):
        """--- 개선 제안 1: 방향 카테고리 정의 개선 ---"""
        print("    화재 확산 방향 카테고리 생성...")
        
        wind_dir_col = self.find_column(self.wind_dir_col_candidates)

        def categorize_by_angle(angle):
            if pd.isna(angle): return 'north' # 기본값
            elif 337.5 <= angle <= 360 or 0 <= angle < 22.5: return 'north'
            elif 22.5 <= angle < 67.5: return 'northeast'
            elif 67.5 <= angle < 112.5: return 'east'
            elif 112.5 <= angle < 157.5: return 'southeast'
            elif 157.5 <= angle < 202.5: return 'south'
            elif 202.5 <= angle < 247.5: return 'southwest'
            elif 247.5 <= angle < 292.5: return 'west'
            elif 292.5 <= angle < 337.5: return 'northwest'
            else: return 'north'

        # 1순위: 풍향 데이터 사용
        if wind_dir_col:
            print(f"    '{wind_dir_col}' 컬럼을 사용하여 방향 추정 (8방위)...")
            df['direction_category'] = df[wind_dir_col].apply(categorize_by_angle)
        
        # 2순위: 경사 방향 데이터 사용
        elif 'aspect_mode' in df.columns:
            print(f"    경고: 풍향 컬럼을 찾을 수 없습니다. 'aspect_mode' 컬럼으로 방향을 추정합니다.")
            df['direction_category'] = df['aspect_mode'].apply(categorize_by_angle)
        
        # 3순위: 위경도 기반 추정 (기존 로직 유지)
        elif all(col in df.columns for col in ['start_latitude', 'start_longitude']):
            print("    경고: 풍향, 경사방향 컬럼이 없어 위경도 기반으로 방향을 추정합니다.")
            lat_median = df['start_latitude'].median()
            lon_median = df['start_longitude'].median()
            
            def geo_direction(row):
                lat, lon = row['start_latitude'], row['start_longitude']
                if lat > lat_median and lon <= lon_median: return 'north'
                elif lat <= lat_median and lon > lon_median: return 'east'
                elif lat <= lat_median and lon <= lon_median: return 'south'
                else: return 'west'
            
            df['direction_category'] = df.apply(geo_direction, axis=1)
        
        # 4순위: 랜덤 배정
        else:
            print("    경고: 방향 추정에 사용할 수 있는 컬럼이 없어 랜덤으로 방향을 배정합니다.")
            np.random.seed(42)
            directions = ['north', 'south', 'east', 'west']
            df['direction_category'] = np.random.choice(directions, size=len(df))
        
        return df
    
    def prepare_features_for_classification(self, target_type):
        """분류를 위한 피처 준비"""
        print(f"🔧 {target_type} 분류를 위한 피처 준비...")
        
        # 수치형 피처만 선택
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 타겟 변수들 제외
        exclude_cols = ['fire_area', 'speed_category', 'direction_category']
        if target_type == 'speed':
            target_col = 'speed_category'
        else:
            target_col = 'direction_category'
        
        # object, category 타입 컬럼 제외
        feature_cols = [col for col in numeric_cols if col not in self.df.select_dtypes(include=['object', 'category']).columns]
        feature_cols = [col for col in feature_cols if col not in exclude_cols]
        
        # 데이터 준비
        X = self.df[feature_cols].copy()
        y = self.df[target_col].copy()
        
        print(f"  초기 피처 수: {X.shape[1]}")
        print(f"  타겟 분포: {Counter(y)}")
        
        # 데이터 정리
        print("  데이터 정리 중...")
        
        # 1. 결측치가 많은 컬럼 제거 (50% 이상)
        missing_ratio = X.isnull().sum() / len(X)
        cols_to_drop = missing_ratio[missing_ratio > 0.5].index.tolist()
        if cols_to_drop:
            print(f"    결측치 많은 컬럼 제거: {len(cols_to_drop)}개")
            X = X.drop(columns=cols_to_drop)
        
        # 2. 나머지 결측치 중앙값으로 처리
        X = X.fillna(X.median())
        
        # 3. 무한값 처리
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # 4. 분산이 0인 컬럼 제거
        zero_var_cols = X.columns[X.var() == 0].tolist()
        if zero_var_cols:
            print(f"    분산 0인 컬럼 제거: {len(zero_var_cols)}개")
            X = X.drop(columns=zero_var_cols)
        
        # 5. 극값 처리 (보수적으로)
        for col in X.columns:
            q1 = X[col].quantile(0.01)
            q99 = X[col].quantile(0.99)
            X[col] = X[col].clip(q1, q99)
        
        print(f"  최종 피처 수: {X.shape[1]}")
        
        return X, y
    
    def select_relevant_features(self, X, y, target_type, max_features=30):
        """관련성 높은 피처 선택"""
        print(f"🎯 {target_type} 관련 피처 선택...")
        
        if target_type == 'speed':
            # 속도와 관련된 키워드들
            speed_keywords = ['wind', 'ws', 'wd', 'fwi', 'isi', 'ffmc', 'temp', 't2m', 
                             'humidity', 'rh', 'dry', 'slope', 'aspect', 'interaction'] # 상호작용 피처 추가
        else:
            # 방향과 관련된 키워드들
            direction_keywords = ['aspect', 'slope', 'wind', 'wd', 'latitude', 'longitude',
                                'elevation', 'terrain', 'north', 'south', 'east', 'west']
        
        # 키워드 기반 피처 우선 선택
        keyword_features = []
        keywords = speed_keywords if target_type == 'speed' else direction_keywords
        
        for col in X.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in keywords):
                keyword_features.append(col)
        
        print(f"  키워드 기반 피처: {len(keyword_features)}개")
        
        # 키워드 피처가 충분하지 않으면 상관관계로 보완
        if len(keyword_features) < max_features:
            # 라벨 인코딩
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            # 상관관계 계산
            correlations = []
            remaining_cols = [col for col in X.columns if col not in keyword_features]
            
            for col in remaining_cols:
                try:
                    corr = abs(np.corrcoef(X[col], y_encoded)[0, 1])
                    if not np.isnan(corr):
                        correlations.append((col, corr))
                except:
                    continue
            
            correlations.sort(key=lambda x: x[1], reverse=True)
            additional_features = [feat for feat, _ in correlations[:max_features - len(keyword_features)]]
            
            selected_features = keyword_features + additional_features
        else:
            selected_features = keyword_features[:max_features]
        
        X_selected = X[selected_features]
        print(f"  최종 선택된 피처: {len(selected_features)}개")
        print(f"  상위 5개 피처: {selected_features[:5]}")
        
        return X_selected, selected_features
    
    def train_classification_models(self, X, y, target_type):
        """분류 모델들 학습"""
        print(f"🏗️ {target_type} 분류 모델 학습...")
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {}
        
        # 1. Random Forest (기존 모델)
        print("  1. Random Forest")
        try:
            rf_params = {'n_estimators': [100, 200], 'max_depth': [5, 10], 'min_samples_split': [5, 10], 'class_weight': ['balanced']}
            rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, scoring='accuracy', n_jobs=-1)
            rf_grid.fit(X_train, y_train) # RF는 스케일링에 덜 민감하므로 원본 데이터 사용
            best_rf = rf_grid.best_estimator_
            y_pred_rf = best_rf.predict(X_test)
            models['RandomForest'] = {'model': best_rf, 'scaler': 'none', 'accuracy': accuracy_score(y_test, y_pred_rf), 'predictions': y_pred_rf, 'best_params': rf_grid.best_params_, 'classification_report': classification_report(y_test, y_pred_rf, output_dict=True)}
            print(f"    Random Forest 정확도: {models['RandomForest']['accuracy']:.4f}")
        except Exception as e:
            print(f"    Random Forest 실패: {e}")
        
        # 2. Gradient Boosting (기존 모델)
        print("  2. Gradient Boosting")
        try:
            gb_params = {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.1, 0.2]}
            gb_grid = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_params, cv=3, scoring='accuracy', n_jobs=-1)
            gb_grid.fit(X_train_scaled, y_train)
            best_gb = gb_grid.best_estimator_
            y_pred_gb = best_gb.predict(X_test_scaled)
            models['GradientBoosting'] = {'model': best_gb, 'scaler': scaler, 'accuracy': accuracy_score(y_test, y_pred_gb), 'predictions': y_pred_gb, 'best_params': gb_grid.best_params_, 'classification_report': classification_report(y_test, y_pred_gb, output_dict=True)}
            print(f"    Gradient Boosting 정확도: {models['GradientBoosting']['accuracy']:.4f}")
        except Exception as e:
            print(f"    Gradient Boosting 실패: {e}")

        # --- 개선 제안 2: LightGBM 모델 추가 ---
        print("  3. LightGBM")
        try:
            lgbm_params = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.1, 0.05],
                'class_weight': ['balanced']
            }
            lgbm_grid = GridSearchCV(lgb.LGBMClassifier(random_state=42), lgbm_params, cv=3, scoring='accuracy', n_jobs=-1)
            lgbm_grid.fit(X_train, y_train) # LGBM도 스케일링 불필요
            best_lgbm = lgbm_grid.best_estimator_
            y_pred_lgbm = best_lgbm.predict(X_test)
            models['LightGBM'] = {'model': best_lgbm, 'scaler': 'none', 'accuracy': accuracy_score(y_test, y_pred_lgbm), 'predictions': y_pred_lgbm, 'best_params': lgbm_grid.best_params_, 'classification_report': classification_report(y_test, y_pred_lgbm, output_dict=True)}
            print(f"    LightGBM 정확도: {models['LightGBM']['accuracy']:.4f}")
        except Exception as e:
            print(f"    LightGBM 실패: {e}")

        return models, y_test
    
    def prepare_features_for_regression(self):
        """회귀를 위한 피처 준비 (면적 예측)"""
        print("🔧 면적 회귀를 위한 피처 준비...")
        
        # 수치형 피처만 선택
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 타겟 변수들 제외
        exclude_cols = ['fire_area', 'speed_category', 'direction_category']
        target_col = 'fire_area'
        
        # object, category 타입 컬럼 제외
        feature_cols = [col for col in numeric_cols if col not in self.df.select_dtypes(include=['object', 'category']).columns]
        feature_cols = [col for col in feature_cols if col not in exclude_cols]
        
        # 데이터 준비
        X = self.df[feature_cols].copy()
        y = self.df[target_col].copy()
        
        print(f"  초기 피처 수: {X.shape[1]}")
        print(f"  타겟 통계: mean={y.mean():.2f}, median={y.median():.2f}, max={y.max():.2f}")
        
        # 데이터 정리 (이전과 동일)
        print("  데이터 정리 중...")
        
        # 1. 결측치가 많은 컬럼 제거 (50% 이상)
        missing_ratio = X.isnull().sum() / len(X)
        cols_to_drop = missing_ratio[missing_ratio > 0.5].index.tolist()
        if cols_to_drop:
            print(f"    결측치 많은 컬럼 제거: {len(cols_to_drop)}개")
            X = X.drop(columns=cols_to_drop)
        
        # 2. 나머지 결측치 중앙값으로 처리
        X = X.fillna(X.median())
        
        # 3. 무한값 처리
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # 4. 분산이 0인 컬럼 제거
        zero_var_cols = X.columns[X.var() == 0].tolist()
        if zero_var_cols:
            print(f"    분산 0인 컬럼 제거: {len(zero_var_cols)}개")
            X = X.drop(columns=zero_var_cols)
        
        # 5. 극값 처리 (보수적으로)
        for col in X.columns:
            q1 = X[col].quantile(0.01)
            q99 = X[col].quantile(0.99)
            X[col] = X[col].clip(q1, q99)
        
        print(f"  최종 피처 수: {X.shape[1]}")
        
        return X, y
    
    def select_area_relevant_features(self, X, y, max_features=40):
        """면적 예측 관련 피처 선택"""
        print("🎯 면적 예측 관련 피처 선택...")
        
        # 면적과 관련된 키워드들
        area_keywords = ['fwi', 'isi', 'ffmc', 'dmc', 'dc', 'bui', 'wind', 'ws', 'temp', 't2m', 
                        'humidity', 'rh', 'dry', 'slope', 'fuel', 'ndvi', 'solar', 'precip',
                        'spread', 'fire', 'weather', 'interaction'] # 상호작용 피처 추가
        
        # 키워드 기반 피처 우선 선택
        keyword_features = []
        
        for col in X.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in area_keywords):
                keyword_features.append(col)
        
        print(f"  키워드 기반 피처: {len(keyword_features)}개")
        
        # 키워드 피처가 충분하지 않으면 상관관계로 보완
        if len(keyword_features) < max_features:
            # 상관관계 계산
            correlations = []
            remaining_cols = [col for col in X.columns if col not in keyword_features]
            
            for col in remaining_cols:
                try:
                    corr = abs(np.corrcoef(X[col], y)[0, 1])
                    if not np.isnan(corr):
                        correlations.append((col, corr))
                except:
                    continue
            
            correlations.sort(key=lambda x: x[1], reverse=True)
            additional_features = [feat for feat, _ in correlations[:max_features - len(keyword_features)]]
            
            selected_features = keyword_features + additional_features
        else:
            selected_features = keyword_features[:max_features]
        
        X_selected = X[selected_features]
        print(f"  최종 선택된 피처: {len(selected_features)}개")
        print(f"  상위 5개 피처: {selected_features[:5]}")
        
        return X_selected, selected_features

    def train_regression_models(self, X, y):
        """회귀 모델들 학습 (면적 예측)"""
        print("🏗️ 면적 회귀 모델 학습...")
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        
        # 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 타겟 변환 (log1p)
        y_train_log = np.log1p(y_train)
        y_test_log = np.log1p(y_test)
        
        models = {}
        
        def evaluate_regression(y_true_log, y_pred_log):
            y_true = np.expm1(y_true_log)
            y_pred = np.expm1(y_pred_log)
            y_pred = np.maximum(y_pred, 0)
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            return r2, rmse, mae, y_pred
        
        # 1. Random Forest
        print("  1. Random Forest")
        try:
            rf_params = {'n_estimators': [100, 200], 'max_depth': [10, 15], 'min_samples_split': [5, 10]}
            rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=3, scoring='r2', n_jobs=-1)
            rf_grid.fit(X_train_scaled, y_train_log)
            best_rf = rf_grid.best_estimator_
            y_pred_rf = best_rf.predict(X_test_scaled)
            r2_rf, rmse_rf, mae_rf, y_pred_rf_orig = evaluate_regression(y_test_log, y_pred_rf)
            models['RandomForest'] = {'model': best_rf, 'scaler': scaler, 'r2': r2_rf, 'rmse': rmse_rf, 'mae': mae_rf, 'predictions': y_pred_rf_orig, 'best_params': rf_grid.best_params_}
            print(f"    Random Forest - R²: {r2_rf:.4f}, RMSE: {rmse_rf:.2f}")
        except Exception as e:
            print(f"    Random Forest 실패: {e}")
        
        # 2. Gradient Boosting
        print("  2. Gradient Boosting")
        try:
            gb_params = {'n_estimators': [100, 200], 'max_depth': [5, 8], 'learning_rate': [0.1, 0.2]}
            gb_grid = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_params, cv=3, scoring='r2', n_jobs=-1)
            gb_grid.fit(X_train_scaled, y_train_log)
            best_gb = gb_grid.best_estimator_
            y_pred_gb = best_gb.predict(X_test_scaled)
            r2_gb, rmse_gb, mae_gb, y_pred_gb_orig = evaluate_regression(y_test_log, y_pred_gb)
            models['GradientBoosting'] = {'model': best_gb, 'scaler': scaler, 'r2': r2_gb, 'rmse': rmse_gb, 'mae': mae_gb, 'predictions': y_pred_gb_orig, 'best_params': gb_grid.best_params_}
            print(f"    Gradient Boosting - R²: {r2_gb:.4f}, RMSE: {rmse_gb:.2f}")
        except Exception as e:
            print(f"    Gradient Boosting 실패: {e}")

        # --- 개선 제안 2: LightGBM 모델 추가 ---
        print("  3. LightGBM")
        try:
            lgbm_params = {'n_estimators': [100, 200], 'max_depth': [5, 8, 10], 'learning_rate': [0.1, 0.05]}
            lgbm_grid = GridSearchCV(lgb.LGBMRegressor(random_state=42), lgbm_params, cv=3, scoring='r2', n_jobs=-1)
            lgbm_grid.fit(X_train_scaled, y_train_log)
            best_lgbm = lgbm_grid.best_estimator_
            y_pred_lgbm = best_lgbm.predict(X_test_scaled)
            r2_lgbm, rmse_lgbm, mae_lgbm, y_pred_lgbm_orig = evaluate_regression(y_test_log, y_pred_lgbm)
            models['LightGBM'] = {'model': best_lgbm, 'scaler': scaler, 'r2': r2_lgbm, 'rmse': rmse_lgbm, 'mae': mae_lgbm, 'predictions': y_pred_lgbm_orig, 'best_params': lgbm_grid.best_params_}
            print(f"    LightGBM - R²: {r2_lgbm:.4f}, RMSE: {rmse_lgbm:.2f}")
        except Exception as e:
            print(f"    LightGBM 실패: {e}")

        return models, y_test_log

    def save_models_and_results(self, models, selected_features, y_test, target_type):
        """모델과 결과 저장"""
        if not models:
            print(f"❌ {target_type} 모델 학습 실패")
            return
        
        base_path = '/Users/mmymacymac/Developer/Projects/WildFire_projects/wildfire/wildfire/ML'
        
        best_model_name = max(models.keys(), key=lambda x: models[x]['accuracy'])
        best_model_info = models[best_model_name]
        
        print(f"\n🏆 {target_type} 최고 모델: {best_model_name}")
        print(f"📊 정확도: {best_model_info['accuracy']:.4f}")
        
        model_path = f'{base_path}/{target_type}_model_retrained.joblib'
        joblib.dump(best_model_info['model'], model_path)
        
        if best_model_info['scaler'] != 'none':
            scaler_path = f'{base_path}/{target_type}_scaler_retrained.joblib'
            joblib.dump(best_model_info['scaler'], scaler_path)
        else:
            scaler_path = "none"

        features_path = f'{base_path}/{target_type}_features_retrained.json'
        with open(features_path, 'w') as f: json.dump(selected_features, f, indent=2)
        
        summary = {
            'target_type': target_type,
            'best_model': best_model_name,
            'best_accuracy': float(best_model_info['accuracy']),
            'all_results': {k: {'accuracy': float(v['accuracy']), 'classification_report': v['classification_report']} for k, v in models.items()},
            'feature_count': len(selected_features),
            'features': selected_features,
            'note': f'Retrained {target_type} model with new features and models'
        }
        if 'best_params' in best_model_info: summary['best_params'] = best_model_info['best_params']
        
        summary_path = f'{base_path}/{target_type}_retrained_summary.json'
        with open(summary_path, 'w') as f: json.dump(summary, f, indent=2)
        
        plt.figure(figsize=(10, 8))
        labels = sorted(list(y_test.unique()))
        cm = confusion_matrix(y_test, best_model_info['predictions'], labels=labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f'{target_type.title()} Model Confusion Matrix\nAccuracy: {best_model_info["accuracy"]:.4f}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plot_path = f'{base_path}/{target_type}_confusion_matrix_retrained.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n💾 {target_type} 모델 저장 완료:")
        print(f"  - 모델: {model_path}")
        print(f"  - 스케일러: {scaler_path}")
        print(f"  - 피처: {features_path}")
        print(f"  - 요약: {summary_path}")
        
        return summary

    def save_regression_results(self, models, selected_features, y_test):
        """회귀 모델 결과 저장"""
        if not models:
            print("❌ 면적 모델 학습 실패")
            return
        
        base_path = '/Users/mmymacymac/Developer/Projects/WildFire_projects/wildfire/wildfire/ML'
        
        best_model_name = max(models.keys(), key=lambda x: models[x]['r2'])
        best_model_info = models[best_model_name]
        
        print(f"\n🏆 면적 최고 모델: {best_model_name}")
        print(f"📊 R²: {best_model_info['r2']:.4f}, RMSE: {best_model_info['rmse']:.2f}")
        
        if best_model_info['model'] != 'ensemble':
            model_path = f'{base_path}/area_model_retrained.joblib'
            joblib.dump(best_model_info['model'], model_path)
        else:
            model_path = "ensemble_model"
        
        scaler_path = f'{base_path}/area_scaler_retrained.joblib'
        joblib.dump(best_model_info['scaler'], scaler_path)
        
        features_path = f'{base_path}/area_features_retrained.json'
        with open(features_path, 'w') as f: json.dump(selected_features, f, indent=2)
        
        summary = {
            'target_type': 'area',
            'best_model': best_model_name,
            'best_r2': float(best_model_info['r2']),
            'best_rmse': float(best_model_info['rmse']),
            'best_mae': float(best_model_info['mae']),
            'all_results': {k: {'r2': float(v['r2']), 'rmse': float(v['rmse']), 'mae': float(v['mae'])} for k, v in models.items()},
            'feature_count': len(selected_features),
            'features': selected_features,
            'note': 'Retrained area model with new features and models'
        }
        
        if 'best_params' in best_model_info: summary['best_params'] = best_model_info['best_params']
        
        summary_path = f'{base_path}/area_retrained_summary.json'
        with open(summary_path, 'w') as f: json.dump(summary, f, indent=2)
        
        print(f"\n💾 면적 모델 저장 완료:")
        print(f"  - 모델: {model_path}")
        print(f"  - 스케일러: {scaler_path}")
        print(f"  - 피처: {features_path}")
        print(f"  - 요약: {summary_path}")
        
        return summary

    def retrain_all_models(self):
        """속도, 방향, 면적 모델 모두 재학습"""
        print("=" * 70)
        print("🔄 전체 모델 재학습 (개선된 로직 적용)")
        print("=" * 70)
        
        self.load_and_prepare_data()
        results = {}
        
        try:
            print("\n" + "=" * 50)
            print("🚀 속도 모델 재학습")
            print("=" * 50)
            X_speed, y_speed = self.prepare_features_for_classification('speed')
            X_speed_selected, speed_features = self.select_relevant_features(X_speed, y_speed, 'speed')
            speed_models, y_speed_test = self.train_classification_models(X_speed_selected, y_speed, 'speed')
            results['speed'] = self.save_models_and_results(speed_models, speed_features, y_speed_test, 'speed')
        except Exception as e:
            print(f"❌ 속도 모델 재학습 실패: {e}")
            results['speed'] = None
        
        try:
            print("\n" + "=" * 50)
            print("🧭 방향 모델 재학습")
            print("=" * 50)
            X_direction, y_direction = self.prepare_features_for_classification('direction')
            X_direction_selected, direction_features = self.select_relevant_features(X_direction, y_direction, 'direction')
            direction_models, y_direction_test = self.train_classification_models(X_direction_selected, y_direction, 'direction')
            results['direction'] = self.save_models_and_results(direction_models, direction_features, y_direction_test, 'direction')
        except Exception as e:
            print(f"❌ 방향 모델 재학습 실패: {e}")
            results['direction'] = None
        
        try:
            print("\n" + "=" * 50)
            print("🔥 면적 모델 재학습")
            print("=" * 50)
            X_area, y_area = self.prepare_features_for_regression()
            X_area_selected, area_features = self.select_area_relevant_features(X_area, y_area)
            area_models, y_area_test = self.train_regression_models(X_area_selected, y_area)
            results['area'] = self.save_regression_results(area_models, area_features, y_area_test)
        except Exception as e:
            print(f"❌ 면적 모델 재학습 실패: {e}")
            results['area'] = None
        
        print("\n" + "=" * 70)
        print("📊 전체 재학습 결과 요약")
        print("=" * 70)
        for model_type, result in results.items():
            if result:
                if model_type == 'area':
                    print(f"{model_type.title()} 모델: ✅ 성공 - {result['best_model']} (R²: {result['best_r2']:.4f}, RMSE: {result['best_rmse']:.2f})")
                else:
                    print(f"{model_type.title()} 모델: ✅ 성공 - {result['best_model']} (정확도: {result['best_accuracy']:.4f})")
            else:
                print(f"{model_type.title()} 모델: ❌ 실패")
        
        return results

def main():
    print("🔄 전체 모델 재학습 (개선된 로직 적용)")
    data_path = '/Users/mmymacymac/Developer/Projects/WildFire_projects/wildfire/wildfire/ML/clean_training_dataset.csv'
    retrainer = ComprehensiveModelRetrainer(data_path)
    results = retrainer.retrain_all_models()
    return results

if __name__ == "__main__":
    main()