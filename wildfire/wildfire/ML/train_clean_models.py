#!/usr/bin/env python3
"""
데이터 누수가 제거된 클린 데이터셋으로 모델 훈련
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import json
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class CleanModelTrainer:
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
                       ['t2m_', 'rh2m_', 'ws2m_', 'wd2m_', 'ws10m_', 'wd10m_', 'ps_', 'allsky_sfc_sw_dwn_'])]
        
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
        
        # 속도와 방향을 위한 타겟 변수 생성 (원본과 동일한 방식)
        # 속도: 피해면적 기준 범주화
        area_quantiles = self.y_area.quantile([0.33, 0.67])
        self.y_speed = pd.cut(self.y_area, 
                             bins=[0, area_quantiles[0.33], area_quantiles[0.67], self.y_area.max()],
                             labels=['slow', 'medium', 'fast'],
                             include_lowest=True)
        
        # 방향: 더미 변수 (실제로는 바람 방향 기반으로 해야 하지만 간단히 구현)
        if 'wd10m_0h' in self.X.columns:
            wind_direction = self.X['wd10m_0h'].fillna(0)
            direction_bins = [0, 90, 180, 270, 360]
            direction_labels = ['north', 'east', 'south', 'west']
            self.y_direction = pd.cut(wind_direction, bins=direction_bins, labels=direction_labels, include_lowest=True)
        else:
            # 임의로 방향 생성 (데모용)
            np.random.seed(42)
            self.y_direction = pd.Categorical(np.random.choice(['north', 'east', 'south', 'west'], size=len(self.df)))
        
        print(f"피처 수: {self.X.shape[1]}")
        print(f"피해면적 범위: {self.y_area.min():.3f} ~ {self.y_area.max():.3f}")
        print(f"속도 분포:\\n{self.y_speed.value_counts()}")
        print(f"방향 분포:\\n{self.y_direction.value_counts()}")
        
    def prepare_features(self, task='area'):
        """작업별 피처 준비"""
        # 카테고리 피처 인코딩
        categorical_features = ['is_spring', 'is_summer', 'is_autumn', 'is_winter']
        X_processed = self.X.copy()
        
        for cat_col in categorical_features:
            if cat_col in X_processed.columns:
                X_processed[cat_col] = X_processed[cat_col].astype(int)
        
        # 수치형 피처만 선택 (NaN이나 무한값 제거)
        X_numeric = X_processed.select_dtypes(include=[np.number])
        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
        X_numeric = X_numeric.fillna(X_numeric.mean())
        
        return X_numeric
        
    def train_area_model(self):
        """피해면적 예측 모델 훈련 (회귀)"""
        print("\\n=== 피해면적 모델 훈련 ===")
        
        X = self.prepare_features('area')
        y = self.y_area
        
        # 학습/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        # 피처 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 모델 훈련
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # 예측 및 평가
        y_pred = model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {np.sqrt(mse):.6f}")
        print(f"R²: {r2:.4f}")
        
        # 모델과 스케일러 저장
        self.models['area'] = model
        self.scalers['area'] = scaler
        self.feature_columns['area'] = X.columns.tolist()
        
        # 피처 중요도 저장
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2,
            'feature_importance': feature_importance.head(20).to_dict('records')
        }
    
    def train_speed_model(self):
        """확산 속도 예측 모델 훈련 (분류)"""
        print("\\n=== 확산 속도 모델 훈련 ===")
        
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
        
        # 모델 훈련
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train_encoded)
        
        # 예측 및 평가
        y_pred_encoded = model.predict(X_test_scaled)
        y_pred = encoder.inverse_transform(y_pred_encoded)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"정확도: {accuracy:.4f}")
        print("\\n분류 보고서:")
        print(classification_report(y_test, y_pred))
        
        # 모델, 스케일러, 인코더 저장
        self.models['speed'] = model
        self.scalers['speed'] = scaler
        self.encoders['speed'] = encoder
        self.feature_columns['speed'] = X.columns.tolist()
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def train_direction_model(self):
        """확산 방향 예측 모델 훈련 (분류)"""
        print("\\n=== 확산 방향 모델 훈련 ===")
        
        X = self.prepare_features('direction')
        y = self.y_direction.astype(str)
        
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
        
        # 모델 훈련
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train_encoded)
        
        # 예측 및 평가
        y_pred_encoded = model.predict(X_test_scaled)
        y_pred = encoder.inverse_transform(y_pred_encoded)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"정확도: {accuracy:.4f}")
        print("\\n분류 보고서:")
        print(classification_report(y_test, y_pred))
        
        # 모델, 스케일러, 인코더 저장
        self.models['direction'] = model
        self.scalers['direction'] = scaler
        self.encoders['direction'] = encoder
        self.feature_columns['direction'] = X.columns.tolist()
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def save_models(self):
        """모델들을 파일로 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = '/Users/mmymacymac/Developer/Projects/WildFire_projects/wildfire/wildfire/ML'
        
        # 모델 저장
        for model_type, model in self.models.items():
            model_path = f'{base_path}/{model_type}_model_clean_{timestamp}.joblib'
            joblib.dump(model, model_path)
            print(f"모델 저장: {model_path}")
        
        # 스케일러 저장
        for model_type, scaler in self.scalers.items():
            scaler_path = f'{base_path}/{model_type}_scaler_clean.joblib'
            joblib.dump(scaler, scaler_path)
            print(f"스케일러 저장: {scaler_path}")
        
        # 인코더 저장 (분류 모델용)
        for model_type, encoder in self.encoders.items():
            encoder_path = f'{base_path}/{model_type}_encoder_clean.joblib'
            joblib.dump(encoder, encoder_path)
            print(f"인코더 저장: {encoder_path}")
        
        # 피처 컬럼 정보 저장
        for model_type, columns in self.feature_columns.items():
            columns_path = f'{base_path}/{model_type}_columns_clean.json'
            with open(columns_path, 'w') as f:
                json.dump(columns, f, indent=2)
            print(f"피처 컬럼 저장: {columns_path}")

def main():
    print("클린 데이터셋으로 모델 훈련 시작...")
    
    data_path = '/Users/mmymacymac/Developer/Projects/WildFire_projects/wildfire/wildfire/ML/clean_training_dataset.csv'
    
    trainer = CleanModelTrainer(data_path)
    
    # 데이터 로드 및 전처리
    trainer.load_and_prepare_data()
    
    # 모델 훈련
    results = {}
    
    # 피해면적 모델
    results['area'] = trainer.train_area_model()
    
    # 속도 모델  
    results['speed'] = trainer.train_speed_model()
    
    # 방향 모델
    results['direction'] = trainer.train_direction_model()
    
    # 모델 저장
    trainer.save_models()
    
    # 결과 요약 저장
    results_path = '/Users/mmymacymac/Developer/Projects/WildFire_projects/wildfire/wildfire/ML/clean_model_training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\\n훈련 결과 저장: {results_path}")
    
    print("\\n=== 훈련 완료 ===")
    print(f"피해면적 모델 R²: {results['area']['r2']:.4f}")
    print(f"속도 모델 정확도: {results['speed']['accuracy']:.4f}")  
    print(f"방향 모델 정확도: {results['direction']['accuracy']:.4f}")

if __name__ == "__main__":
    main()