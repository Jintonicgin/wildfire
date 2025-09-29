#!/usr/bin/env python3
"""
개선된 모델 훈련 - 바람 방향 데이터 누수 제거 및 지형/연료 기반 방향 예측
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import json
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ImprovedModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = {}
        
    def load_and_prepare_data(self):
        """클린 데이터 로드 및 전처리"""
        print("클린 데이터 로딩...")
        self.df = pd.read_csv(self.data_path)
        print(f"데이터 크기: {self.df.shape}")
        
        # 결측치 처리
        print("\n결측치 처리 중...")
        
        # 강수량 데이터 결측치는 0으로 대체 (강수 없음을 의미)
        if 'prectotcorr_0h' in self.df.columns:
            self.df['prectotcorr_0h'].fillna(0, inplace=True)
        
        # 다른 기상 데이터 결측치는 전후 시간 평균으로 대체
        weather_cols = [col for col in self.df.columns if any(weather in col for weather in 
                       ['t2m_', 'rh2m_', 'ws2m_', 'ws10m_', 'ps_', 'allsky_sfc_sw_dwn_'])]
        
        for col in weather_cols:
            if col in self.df.columns and self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].mean(), inplace=True)
        
        # 지형/식생 데이터 결측치 처리
        if 'ndvi_before' in self.df.columns:
            self.df['ndvi_before'].fillna(self.df['ndvi_before'].median(), inplace=True)
        if 'treecover_pre_fire_5x5' in self.df.columns:
            self.df['treecover_pre_fire_5x5'].fillna(self.df['treecover_pre_fire_5x5'].median(), inplace=True)
        
        # 복합 지수 결측치는 0으로 대체
        composite_cols = ['fuel_combo', 'potential_spread_index', 'wd10m_var_0_12h']
        for col in composite_cols:
            if col in self.df.columns:
                self.df[col].fillna(0, inplace=True)
        
        print(f"결측치 처리 후 남은 결측치: {self.df.isnull().sum().sum()}")
        
        # 타겟 변수와 피처 분리
        self.target_area = 'fire_area'
        self.X = self.df.drop(columns=[self.target_area])
        self.y_area = self.df[self.target_area]
        
        # 속도 타겟: 피해면적 기준 범주화
        area_quantiles = self.y_area.quantile([0.33, 0.67])
        self.y_speed = pd.cut(self.y_area, 
                             bins=[0, area_quantiles[0.33], area_quantiles[0.67], self.y_area.max()],
                             labels=['slow', 'medium', 'fast'],
                             include_lowest=True)
        
        # 방향 타겟: 지형 경사면 기준으로 생성 (바람 방향 대신)
        self.create_terrain_based_direction()
        
        print(f"피처 수: {self.X.shape[1]}")
        print(f"피해면적 범위: {self.y_area.min():.3f} ~ {self.y_area.max():.3f}")
        print(f"속도 분포:\\n{self.y_speed.value_counts()}")
        print(f"방향 분포:\\n{self.y_direction.value_counts()}")
        
    def create_terrain_based_direction(self):
        """지형 기반 확산 방향 생성 (바람 방향 대신)"""
        print("지형 기반 확산 방향 생성...")
        
        # 지형 정보 확인
        terrain_cols = ['aspect_mode', 'slope_mean', 'aspect_north_ratio', 'aspect_south_ratio']
        available_terrain = [col for col in terrain_cols if col in self.X.columns]
        
        if 'aspect_mode' in self.X.columns:
            # 주 향면 기준으로 확산 방향 결정
            aspect_mode = self.X['aspect_mode'].fillna(180)  # 기본값: 남향
            
            # 향면을 4방향으로 분류
            direction_labels = []
            for aspect in aspect_mode:
                if 315 <= aspect or aspect < 45:
                    direction_labels.append('north')
                elif 45 <= aspect < 135:
                    direction_labels.append('east')
                elif 135 <= aspect < 225:
                    direction_labels.append('south')
                else:  # 225 <= aspect < 315
                    direction_labels.append('west')
            
            self.y_direction = pd.Categorical(direction_labels)
            
        elif len(available_terrain) > 0:
            # 다른 지형 정보 활용
            if 'aspect_south_ratio' in self.X.columns and 'aspect_north_ratio' in self.X.columns:
                south_ratio = self.X['aspect_south_ratio'].fillna(0.25)
                north_ratio = self.X['aspect_north_ratio'].fillna(0.25)
                
                direction_labels = []
                for i in range(len(south_ratio)):
                    if south_ratio.iloc[i] > 0.6:
                        direction_labels.append('south')
                    elif north_ratio.iloc[i] > 0.6:
                        direction_labels.append('north')
                    elif south_ratio.iloc[i] > north_ratio.iloc[i]:
                        direction_labels.append('south')
                    else:
                        direction_labels.append('north')
                
                # east/west 추가를 위해 랜덤하게 일부를 변경
                np.random.seed(42)
                for i in range(len(direction_labels)):
                    if np.random.random() < 0.3:  # 30% 확률로 east/west
                        direction_labels[i] = np.random.choice(['east', 'west'])
                
                self.y_direction = pd.Categorical(direction_labels)
            else:
                # 지형 정보가 부족할 경우 위치 기반으로 생성
                print("지형 정보 부족 - 위치 기반 방향 생성")
                self.create_location_based_direction()
        else:
            print("지형 정보 없음 - 위치 기반 방향 생성")
            self.create_location_based_direction()
    
    def create_location_based_direction(self):
        """위치 기반 확산 방향 생성"""
        if 'start_latitude' in self.X.columns and 'start_longitude' in self.X.columns:
            # 위도/경도 기반으로 지역별 주요 확산 방향 추정
            lat = self.X['start_latitude'].fillna(37.5)
            lon = self.X['start_longitude'].fillna(127.8)
            
            direction_labels = []
            for i in range(len(lat)):
                # 강원도 지역 특성 반영 (산악지형, 동해안)
                if lat.iloc[i] > 37.7:  # 북부 지역
                    direction_labels.append('south')  # 남쪽으로 확산 경향
                elif lon.iloc[i] > 128.0:  # 동부 지역 (해안)
                    direction_labels.append('west')  # 서쪽 내륙으로
                elif lat.iloc[i] < 37.3:  # 남부 지역  
                    direction_labels.append('north')  # 북쪽으로 확산
                else:  # 중부 지역
                    direction_labels.append('east')   # 동쪽으로 확산
            
            self.y_direction = pd.Categorical(direction_labels)
        else:
            # 완전 랜덤 (최후 수단)
            np.random.seed(42)
            direction_labels = np.random.choice(['north', 'east', 'south', 'west'], size=len(self.df))
            self.y_direction = pd.Categorical(direction_labels)

    def identify_wind_features(self, columns):
        """바람 관련 피처 식별 (제거 대상)"""
        wind_features = []
        wind_patterns = ['wd2m', 'wd10m', 'ws2m', 'ws10m']  # 바람 방향과 속도
        
        for col in columns:
            if any(pattern in col for pattern in wind_patterns):
                wind_features.append(col)
        
        return wind_features

    def prepare_features(self, task='area'):
        """작업별 피처 준비"""
        # 카테고리 피처 인코딩
        categorical_features = ['is_spring', 'is_summer', 'is_autumn', 'is_winter']
        X_processed = self.X.copy()
        
        for cat_col in categorical_features:
            if cat_col in X_processed.columns:
                X_processed[cat_col] = X_processed[cat_col].astype(int)
        
        # 방향 예측의 경우 바람 관련 피처 제거
        if task == 'direction':
            wind_features = self.identify_wind_features(X_processed.columns)
            print(f"방향 모델에서 제거할 바람 관련 피처: {len(wind_features)}개")
            print("제거 피처 예시:", wind_features[:5] if wind_features else "없음")
            X_processed = X_processed.drop(columns=wind_features, errors='ignore')
            
            # 지형/연료 관련 피처만 강조
            preferred_features = [col for col in X_processed.columns if any(keyword in col.lower() for keyword in 
                                ['elevation', 'slope', 'aspect', 'ndvi', 'treecover', 'fuel', 'terrain'])]
            
            if len(preferred_features) > 20:
                # 지형/연료 피처가 충분하면 다른 피처는 제한적으로 사용
                other_features = [col for col in X_processed.columns if col not in preferred_features]
                # 위치, 시간, 온습도 정보만 추가로 포함
                additional_features = [col for col in other_features if any(keyword in col.lower() for keyword in 
                                     ['latitude', 'longitude', 'month', 'season', 't2m', 'rh2m', 'precip'])]
                final_features = preferred_features + additional_features[:10]  # 최대 10개 추가 피처
                X_processed = X_processed[final_features]
                print(f"방향 모델 최종 피처 수: {len(final_features)}")
        
        # 수치형 피처만 선택 (NaN이나 무한값 제거)
        X_numeric = X_processed.select_dtypes(include=[np.number])
        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
        X_numeric = X_numeric.fillna(X_numeric.mean())
        
        return X_numeric
        
    def train_area_model_improved(self):
        """개선된 피해면적 예측 모델 훈련"""
        print("\\n=== 개선된 피해면적 모델 훈련 ===")
        
        X = self.prepare_features('area')
        y = self.y_area
        
        # 로그 변환으로 스케일 문제 해결
        y_log = np.log1p(y)  # log(1+x) 변환
        
        # 학습/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_log, test_size=0.2, random_state=42
        )
        
        # 로버스트 스케일링 사용 (이상치에 덜 민감)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 여러 모델 비교
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42),
            'Ridge': Ridge(alpha=10.0)
        }
        
        best_model = None
        best_score = float('-inf')
        best_name = ""
        
        for name, model in models.items():
            # 교차 검증
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            print(f"{name} CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            if cv_scores.mean() > best_score:
                best_score = cv_scores.mean()
                best_model = model
                best_name = name
        
        print(f"\\n최고 성능 모델: {best_name}")
        
        # 최고 모델로 훈련
        best_model.fit(X_train_scaled, y_train)
        
        # 예측 및 평가
        y_pred_log = best_model.predict(X_test_scaled)
        y_pred = np.expm1(y_pred_log)  # 로그 변환 되돌리기
        y_test_original = np.expm1(y_test)
        
        mse = mean_squared_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)
        
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {np.sqrt(mse):.6f}")
        print(f"R²: {r2:.4f}")
        
        # 모델과 스케일러 저장
        self.models['area'] = best_model
        self.scalers['area'] = scaler
        self.feature_columns['area'] = X.columns.tolist()
        
        # 피처 중요도
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            feature_importance = pd.DataFrame({'feature': [], 'importance': []})
        
        return {
            'model_type': best_name,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2,
            'cv_score': best_score,
            'feature_importance': feature_importance.head(20).to_dict('records') if not feature_importance.empty else []
        }
    
    def train_direction_model_improved(self):
        """개선된 확산 방향 예측 모델 훈련 (바람 피처 제거)"""
        print("\\n=== 개선된 확산 방향 모델 훈련 (지형/연료 기반) ===")
        
        X = self.prepare_features('direction')
        y = self.y_direction.astype(str)
        
        print(f"방향 모델 피처 수: {X.shape[1]}")
        print("주요 피처들:", list(X.columns[:10]))
        
        # 학습/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 피처 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 레이블 인코딩
        encoder = LabelEncoder()
        y_train_encoded = encoder.fit_transform(y_train)
        y_test_encoded = encoder.transform(y_test)
        
        # 모델 훈련 (하이퍼파라미터 튜닝)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        }
        
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train_encoded)
        
        best_model = grid_search.best_estimator_
        print(f"최적 파라미터: {grid_search.best_params_}")
        
        # 예측 및 평가
        y_pred_encoded = best_model.predict(X_test_scaled)
        y_pred = encoder.inverse_transform(y_pred_encoded)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"정확도: {accuracy:.4f}")
        print("\\n분류 보고서:")
        print(classification_report(y_test, y_pred))
        
        # 모델, 스케일러, 인코더 저장
        self.models['direction'] = best_model
        self.scalers['direction'] = scaler
        self.encoders['direction'] = encoder
        self.feature_columns['direction'] = X.columns.tolist()
        
        # 피처 중요도
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\\n상위 10개 중요 피처:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return {
            'accuracy': accuracy,
            'best_params': grid_search.best_params_,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'feature_importance': feature_importance.head(20).to_dict('records')
        }
    
    def train_speed_model_improved(self):
        """개선된 확산 속도 예측 모델 훈련"""
        print("\\n=== 개선된 확산 속도 모델 훈련 ===")
        
        X = self.prepare_features('speed')
        y = self.y_speed.astype(str)
        
        # 학습/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 피처 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 레이블 인코딩
        encoder = LabelEncoder()
        y_train_encoded = encoder.fit_transform(y_train)
        y_test_encoded = encoder.transform(y_test)
        
        # 하이퍼파라미터 튜닝
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10],
            'min_samples_split': [2, 5],
            'class_weight': ['balanced', None]
        }
        
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train_encoded)
        
        best_model = grid_search.best_estimator_
        
        # 예측 및 평가
        y_pred_encoded = best_model.predict(X_test_scaled)
        y_pred = encoder.inverse_transform(y_pred_encoded)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"정확도: {accuracy:.4f}")
        print(f"최적 파라미터: {grid_search.best_params_}")
        print("\\n분류 보고서:")
        print(classification_report(y_test, y_pred))
        
        # 모델, 스케일러, 인코더 저장
        self.models['speed'] = best_model
        self.scalers['speed'] = scaler
        self.encoders['speed'] = encoder
        self.feature_columns['speed'] = X.columns.tolist()
        
        return {
            'accuracy': accuracy,
            'best_params': grid_search.best_params_,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def save_models(self):
        """모델들을 파일로 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = '/Users/mmymacymac/Developer/Projects/WildFire_projects/wildfire/wildfire/ML'
        
        # 모델 저장
        for model_type, model in self.models.items():
            model_path = f'{base_path}/{model_type}_model_improved.joblib'
            joblib.dump(model, model_path)
            print(f"모델 저장: {model_path}")
        
        # 스케일러 저장
        for model_type, scaler in self.scalers.items():
            scaler_path = f'{base_path}/{model_type}_scaler_improved.joblib'
            joblib.dump(scaler, scaler_path)
            print(f"스케일러 저장: {scaler_path}")
        
        # 인코더 저장 (분류 모델용)
        for model_type, encoder in self.encoders.items():
            encoder_path = f'{base_path}/{model_type}_encoder_improved.joblib'
            joblib.dump(encoder, encoder_path)
            print(f"인코더 저장: {encoder_path}")
        
        # 피처 컬럼 정보 저장
        for model_type, columns in self.feature_columns.items():
            columns_path = f'{base_path}/{model_type}_columns_improved_{timestamp}.json'
            with open(columns_path, 'w') as f:
                json.dump(columns, f, indent=2)
            print(f"피처 컬럼 저장: {columns_path}")

def main():
    print("개선된 모델 훈련 시작...")
    
    data_path = '/Users/mmymacymac/Developer/Projects/WildFire_projects/wildfire/wildfire/ML/clean_training_dataset.csv'
    
    trainer = ImprovedModelTrainer(data_path)
    
    # 데이터 로드 및 전처리
    trainer.load_and_prepare_data()
    
    # 모델 훈련
    results = {}
    
    # 피해면적 모델 (개선됨)
    results['area'] = trainer.train_area_model_improved()
    
    # 속도 모델 (개선됨)
    results['speed'] = trainer.train_speed_model_improved()
    
    # 방향 모델 (바람 피처 제거, 지형 기반)
    results['direction'] = trainer.train_direction_model_improved()
    
    # 모델 저장
    trainer.save_models()
    
    # 결과 요약 저장
    results_path = '/Users/mmymacymac/Developer/Projects/WildFire_projects/wildfire/wildfire/ML/improved_model_training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\\n훈련 결과 저장: {results_path}")
    
    print("\\n=== 개선된 모델 훈련 완료 ===")
    print(f"피해면적 모델 ({results['area']['model_type']}) R²: {results['area']['r2']:.4f}")
    print(f"속도 모델 정확도: {results['speed']['accuracy']:.4f}")  
    print(f"방향 모델 정확도 (지형 기반): {results['direction']['accuracy']:.4f}")

if __name__ == "__main__":
    main()