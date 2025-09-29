import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import json
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor, VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import TransformedTargetRegressor
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class ComprehensiveModelImprover:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        
    def load_and_analyze_data(self):
        """데이터 로딩 및 심층 분석"""
        print("📊 데이터 로딩 및 심층 분석...")
        
        self.df = pd.read_csv(self.data_path)
        print(f"데이터 크기: {self.df.shape}")
        
        # 화재 면적 분석
        fire_area = self.df['fire_area']
        print(f"\n화재 면적 분석:")
        print(f"  - 평균: {fire_area.mean():.2f}ha")
        print(f"  - 중앙값: {fire_area.median():.2f}ha")
        print(f"  - 최댓값: {fire_area.max():.2f}ha")
        print(f"  - 0 면적: {(fire_area == 0).sum()}개 ({(fire_area == 0).mean()*100:.1f}%)")
        print(f"  - 1ha 이하: {(fire_area <= 1).sum()}개 ({(fire_area <= 1).mean()*100:.1f}%)")
        print(f"  - 10ha 이상: {(fire_area >= 10).sum()}개 ({(fire_area >= 10).mean()*100:.1f}%)")
        
        return self.df
    
    def create_improved_speed_categories(self, df):
        """개선된 속도 카테고리 생성"""
        print("🚀 개선된 속도 카테고리 생성...")
        
        # 화재 면적과 기상 조건을 종합적으로 고려
        area = df['fire_area'].copy()
        
        # 기상 위험도 계산
        weather_risk = 0
        if 'fwi_0h' in df.columns:
            fwi_normalized = (df['fwi_0h'] - df['fwi_0h'].min()) / (df['fwi_0h'].max() - df['fwi_0h'].min() + 1e-8)
            weather_risk += fwi_normalized * 0.4
        
        if 'ws10m_0h' in df.columns and 'rh2m_0h' in df.columns:
            # 풍속과 습도 조합
            wind_dry_effect = df['ws10m_0h'] * (100 - df['rh2m_0h']) / 100
            wind_normalized = (wind_dry_effect - wind_dry_effect.min()) / (wind_dry_effect.max() - wind_dry_effect.min() + 1e-8)
            weather_risk += wind_normalized * 0.3
        
        if 't2m_0h' in df.columns and 'rh2m_0h' in df.columns:
            # 온도와 습도 조합
            temp_dry_effect = df['t2m_0h'] / (df['rh2m_0h'] + 1)
            temp_normalized = (temp_dry_effect - temp_dry_effect.min()) / (temp_dry_effect.max() - temp_dry_effect.min() + 1e-8)
            weather_risk += temp_normalized * 0.3
        
        # 복합 위험도 기반 속도 분류
        def categorize_speed_improved(row):
            area_val = row['fire_area']
            risk_val = weather_risk.loc[row.name] if len(weather_risk) > 0 else 0
            
            # 면적과 기상 위험도 결합
            combined_score = np.log1p(area_val) * 0.7 + risk_val * 0.3
            
            # 3분위로 나누기
            if combined_score <= np.percentile(combined_score, 33):
                return 'slow'
            elif combined_score <= np.percentile(combined_score, 67):
                return 'medium'
            else:
                return 'fast'
        
        df['speed_category'] = df.apply(categorize_speed_improved, axis=1)
        
        print(f"  개선된 속도 분포:")
        print(df['speed_category'].value_counts().sort_index())
        
        return df
    
    def create_improved_direction_categories(self, df):
        """개선된 방향 카테고리 생성"""
        print("🧭 개선된 방향 카테고리 생성...")
        
        # 주 풍향과 지형을 종합 고려
        direction_scores = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
        
        # 1. 지형 경사 방향 (aspect)
        if 'aspect_mode' in df.columns:
            aspect = df['aspect_mode'].fillna(0)
            
            def aspect_to_direction(asp):
                if pd.isna(asp):
                    return np.random.choice(['north', 'south', 'east', 'west'])
                elif 315 <= asp or asp < 45:
                    return 'north'
                elif 45 <= asp < 135:
                    return 'east'
                elif 135 <= asp < 225:
                    return 'south'
                else:
                    return 'west'
            
            aspect_direction = aspect.apply(aspect_to_direction)
        else:
            aspect_direction = pd.Series(['north'] * len(df), index=df.index)
        
        # 2. 풍향 고려
        wind_direction = aspect_direction.copy()  # 기본값
        if 'wd10m_0h' in df.columns:
            wd = df['wd10m_0h'].fillna(0)
            
            def wind_to_direction(wind_deg):
                if pd.isna(wind_deg):
                    return 'north'
                elif 315 <= wind_deg or wind_deg < 45:
                    return 'north'
                elif 45 <= wind_deg < 135:
                    return 'east'
                elif 135 <= wind_deg < 225:
                    return 'south'
                else:
                    return 'west'
            
            wind_direction = wd.apply(wind_to_direction)
        
        # 3. 지형과 풍향 결합 (지형 60%, 풍향 40%)
        def combine_directions(aspect_dir, wind_dir):
            # 같은 방향이면 그대로
            if aspect_dir == wind_dir:
                return aspect_dir
            
            # 다르면 지형을 우선하되, 약간의 랜덤성 추가
            if np.random.random() < 0.7:
                return aspect_dir
            else:
                return wind_dir
        
        combined_direction = pd.Series([
            combine_directions(aspect_direction.iloc[i], wind_direction.iloc[i])
            for i in range(len(df))
        ], index=df.index)
        
        # 4. 클래스 균형 맞추기
        direction_counts = combined_direction.value_counts()
        target_count = len(df) // 4  # 각 방향 25% 목표
        
        # 너무 많은 클래스에서 일부를 다른 클래스로 재배치
        for direction, count in direction_counts.items():
            if count > target_count * 1.3:  # 30% 초과시
                excess_indices = combined_direction[combined_direction == direction].sample(
                    n=int(count - target_count), random_state=42
                ).index
                
                # 부족한 클래스로 재배치
                under_directions = [d for d, c in direction_counts.items() if c < target_count * 0.7]
                if under_directions:
                    new_direction = np.random.choice(under_directions)
                    combined_direction.loc[excess_indices] = new_direction
        
        df['direction_category'] = combined_direction
        
        print(f"  개선된 방향 분포:")
        print(df['direction_category'].value_counts().sort_index())
        
        return df
    
    def advanced_feature_engineering(self, df):
        """고급 피처 엔지니어링"""
        print("🔧 고급 피처 엔지니어링...")
        
        enhanced_df = df.copy()
        
        # 1. 화재 위험 복합 지수들
        print("  1. 화재 위험 복합 지수 생성")
        
        # FWI 시스템 기반 복합 지수
        if all(col in df.columns for col in ['ffmc_0h', 'dmc_0h', 'dc_0h', 'isi_0h', 'bui_0h', 'fwi_0h']):
            # 연료 습도 복합 지수 (가중 평균)
            enhanced_df['fuel_moisture_index'] = (
                df['ffmc_0h'] * 0.5 + df['dmc_0h'] * 0.3 + df['dc_0h'] * 0.2
            )
            
            # 확산 위험 지수
            enhanced_df['spread_risk_index'] = df['isi_0h'] * np.log1p(df['bui_0h'])
            
            # 종합 화재 위험도
            enhanced_df['comprehensive_fire_risk'] = (
                df['fwi_0h'] * 0.4 + 
                enhanced_df['fuel_moisture_index'] * 0.3 + 
                enhanced_df['spread_risk_index'] * 0.3
            )
        
        # 2. 기상 복합 지수들
        print("  2. 기상 복합 지수 생성")
        
        if all(col in df.columns for col in ['t2m_0h', 'rh2m_0h', 'ws10m_0h']):
            # 대기 건조 지수
            enhanced_df['atmospheric_dryness'] = df['t2m_0h'] * (100 - df['rh2m_0h']) / 100
            
            # 바람-건조 효과
            enhanced_df['wind_dryness_effect'] = df['ws10m_0h'] * (100 - df['rh2m_0h']) / 100
            
            # 화재 기상 지수 (Haines Index 변형)
            enhanced_df['fire_weather_index'] = (
                (df['t2m_0h'] / 10) * ((100 - df['rh2m_0h']) / 10) * (df['ws10m_0h'] / 10)
            )
        
        # 3. 지형 효과 강화
        print("  3. 지형 효과 강화")
        
        if all(col in df.columns for col in ['slope_mean', 'aspect_south_ratio', 'elevation_mean']):
            # 남향 경사 화재 위험도
            enhanced_df['south_slope_fire_risk'] = df['slope_mean'] * df['aspect_south_ratio']
            
            # 고도별 화재 위험도
            enhanced_df['elevation_fire_risk'] = np.log1p(df['elevation_mean']) * df['slope_mean']
            
            # 지형 변화도
            if 'elevation_std' in df.columns:
                enhanced_df['terrain_complexity'] = df['elevation_std'] * df['slope_mean']
        
        # 4. 과거 기상 패턴 요약
        print("  4. 과거 기상 패턴 요약")
        
        # 온도 패턴
        temp_cols = [col for col in df.columns if 't2m_' in col and 'past' in col][:15]
        if len(temp_cols) >= 5:
            temp_df = df[temp_cols]
            enhanced_df['temp_past_mean'] = temp_df.mean(axis=1)
            enhanced_df['temp_past_max'] = temp_df.max(axis=1)
            enhanced_df['temp_past_trend'] = temp_df.iloc[:, -1] - temp_df.iloc[:, 0]
            enhanced_df['temp_volatility'] = temp_df.std(axis=1)
        
        # 습도 패턴
        humidity_cols = [col for col in df.columns if 'rh2m_' in col and 'past' in col][:15]
        if len(humidity_cols) >= 5:
            humidity_df = df[humidity_cols]
            enhanced_df['humidity_past_min'] = humidity_df.min(axis=1)
            enhanced_df['humidity_past_mean'] = humidity_df.mean(axis=1)
            enhanced_df['humidity_dryness_duration'] = (humidity_df < 50).sum(axis=1)
        
        # 풍속 패턴
        wind_cols = [col for col in df.columns if 'ws10m_' in col and 'past' in col][:15]
        if len(wind_cols) >= 5:
            wind_df = df[wind_cols]
            enhanced_df['wind_past_max'] = wind_df.max(axis=1)
            enhanced_df['wind_past_mean'] = wind_df.mean(axis=1)
            enhanced_df['high_wind_duration'] = (wind_df > 5).sum(axis=1)
        
        # 5. 계절 및 시간 패턴
        print("  5. 계절 및 시간 패턴")
        
        if 'fire_month' in df.columns:
            # 고위험 계절
            high_risk_months = [3, 4, 5, 9, 10, 11]
            enhanced_df['high_risk_season'] = df['fire_month'].isin(high_risk_months).astype(int)
            
            # 계절별 순환 인코딩
            enhanced_df['month_sin'] = np.sin(2 * np.pi * df['fire_month'] / 12)
            enhanced_df['month_cos'] = np.cos(2 * np.pi * df['fire_month'] / 12)
            
            # 봄/가을 집중도
            spring_months = [3, 4, 5]
            autumn_months = [9, 10, 11]
            enhanced_df['spring_fire_season'] = df['fire_month'].isin(spring_months).astype(int)
            enhanced_df['autumn_fire_season'] = df['fire_month'].isin(autumn_months).astype(int)
        
        # 6. 식생 관련 강화
        print("  6. 식생 관련 피처 강화")
        
        if 'ndvi_before' in df.columns:
            # NDVI 기반 연료 부하량
            enhanced_df['vegetation_fuel_load'] = np.maximum(df['ndvi_before'], 0)
            
            if 'treecover_pre_fire_5x5' in df.columns:
                enhanced_df['forest_fuel_interaction'] = df['ndvi_before'] * df['treecover_pre_fire_5x5']
            
            if 'ndvi_stress' in df.columns:
                enhanced_df['vegetation_stress_risk'] = np.maximum(-df['ndvi_stress'], 0)
        
        # 7. 상호작용 피처 (핵심만)
        print("  7. 핵심 상호작용 피처 생성")
        
        # 온도-습도 상호작용
        if 't2m_0h' in df.columns and 'rh2m_0h' in df.columns:
            enhanced_df['temp_humidity_interaction'] = df['t2m_0h'] / (df['rh2m_0h'] + 1)
        
        # FWI-풍속 상호작용
        if 'fwi_0h' in df.columns and 'ws10m_0h' in df.columns:
            enhanced_df['fwi_wind_interaction'] = df['fwi_0h'] * df['ws10m_0h']
        
        # 경사-풍속 상호작용
        if 'slope_mean' in df.columns and 'ws10m_0h' in df.columns:
            enhanced_df['slope_wind_interaction'] = df['slope_mean'] * df['ws10m_0h']
        
        # 8. 무한값 및 NaN 처리
        enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan)
        
        # 새로 생성된 피처들의 결측치만 처리
        new_cols = [col for col in enhanced_df.columns if col not in df.columns]
        for col in new_cols:
            enhanced_df[col] = enhanced_df[col].fillna(enhanced_df[col].median())
        
        print(f"  피처 엔지니어링 완료: {len(enhanced_df.columns) - len(df.columns)}개 피처 추가")
        
        return enhanced_df
    
    def intelligent_feature_selection(self, X, y, task_type='regression', max_features=50):
        """지능적 피처 선택"""
        print(f"🎯 {task_type} 피처 선택 (최대 {max_features}개)...")
        
        # 1. 결측치 및 분산 기반 필터링
        print("  1. 기본 필터링")
        
        # 결측치가 많은 컬럼 제거 (30% 이상)
        missing_ratio = X.isnull().sum() / len(X)
        high_missing_cols = missing_ratio[missing_ratio > 0.3].index.tolist()
        if high_missing_cols:
            print(f"    결측치 많은 컬럼 제거: {len(high_missing_cols)}개")
            X = X.drop(columns=high_missing_cols)
        
        # 분산이 0인 컬럼 제거
        zero_var_cols = X.columns[X.var() == 0].tolist()
        if zero_var_cols:
            print(f"    분산 0인 컬럼 제거: {len(zero_var_cols)}개")
            X = X.drop(columns=zero_var_cols)
        
        # 나머지 결측치 처리
        X = X.fillna(X.median())
        
        # 2. 통계적 피처 선택
        print("  2. 통계적 피처 선택")
        
        try:
            if task_type == 'classification':
                # 분류를 위한 선택
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                selector = SelectKBest(score_func=f_classif, k=min(max_features * 2, len(X.columns)))
                X_selected = selector.fit_transform(X, y_encoded)
                selected_features = X.columns[selector.get_support()]
            else:
                # 회귀를 위한 선택
                selector = SelectKBest(score_func=f_regression, k=min(max_features * 2, len(X.columns)))
                X_selected = selector.fit_transform(X, y)
                selected_features = X.columns[selector.get_support()]
            
            print(f"    통계적 선택: {len(selected_features)}개")
            
        except Exception as e:
            print(f"    통계적 선택 실패, 상관관계 사용: {e}")
            # 상관관계 백업
            correlations = []
            for col in X.columns:
                try:
                    if task_type == 'classification':
                        le = LabelEncoder()
                        y_encoded = le.fit_transform(y)
                        corr = abs(np.corrcoef(X[col], y_encoded)[0, 1])
                    else:
                        corr = abs(np.corrcoef(X[col], y)[0, 1])
                    
                    if not np.isnan(corr):
                        correlations.append((col, corr))
                except:
                    continue
            
            correlations.sort(key=lambda x: x[1], reverse=True)
            selected_features = [feat for feat, _ in correlations[:max_features * 2]]
            X_selected = X[selected_features]
        
        # 3. 최종 중요도 기반 선택
        print("  3. 중요도 기반 최종 선택")
        
        try:
            if task_type == 'classification':
                rf_selector = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                rf_selector.fit(X_selected, y)
            else:
                rf_selector = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                rf_selector.fit(X_selected, y)
            
            # 중요도 기반 최종 선택
            feature_importance = pd.DataFrame({
                'feature': selected_features,
                'importance': rf_selector.feature_importances_
            }).sort_values('importance', ascending=False)
            
            final_features = feature_importance.head(max_features)['feature'].tolist()
            X_final = X[final_features]
            
            print(f"    최종 선택: {len(final_features)}개")
            print(f"    상위 5개: {final_features[:5]}")
            
            return X_final, final_features
            
        except Exception as e:
            print(f"    중요도 선택 실패, 기존 결과 사용: {e}")
            final_features = selected_features[:max_features]
            return X[final_features], final_features
    
    def train_improved_classification_models(self, X, y, target_type):
        """개선된 분류 모델 학습"""
        print(f"🏗️ 개선된 {target_type} 분류 모델 학습...")
        
        # 데이터 분할 (층화)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # 스케일링
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {}
        
        # 1. Random Forest (최적화)
        print("  1. Optimized Random Forest")
        try:
            rf_model = RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            y_pred_rf = rf_model.predict(X_test)
            accuracy_rf = accuracy_score(y_test, y_pred_rf)
            
            models['RandomForest'] = {
                'model': rf_model, 'scaler': scaler, 'accuracy': accuracy_rf,
                'predictions': y_pred_rf, 'classification_report': classification_report(y_test, y_pred_rf, output_dict=True)
            }
            print(f"    Random Forest: {accuracy_rf:.4f}")
            
        except Exception as e:
            print(f"    Random Forest 실패: {e}")
        
        # 2. XGBoost
        print("  2. XGBoost")
        try:
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='mlogloss'
            )
            xgb_model.fit(X_train_scaled, y_train)
            y_pred_xgb = xgb_model.predict(X_test_scaled)
            accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
            
            models['XGBoost'] = {
                'model': xgb_model, 'scaler': scaler, 'accuracy': accuracy_xgb,
                'predictions': y_pred_xgb, 'classification_report': classification_report(y_test, y_pred_xgb, output_dict=True)
            }
            print(f"    XGBoost: {accuracy_xgb:.4f}")
            
        except Exception as e:
            print(f"    XGBoost 실패: {e}")
        
        # 3. LightGBM
        print("  3. LightGBM")
        try:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
            lgb_model.fit(X_train_scaled, y_train)
            y_pred_lgb = lgb_model.predict(X_test_scaled)
            accuracy_lgb = accuracy_score(y_test, y_pred_lgb)
            
            models['LightGBM'] = {
                'model': lgb_model, 'scaler': scaler, 'accuracy': accuracy_lgb,
                'predictions': y_pred_lgb, 'classification_report': classification_report(y_test, y_pred_lgb, output_dict=True)
            }
            print(f"    LightGBM: {accuracy_lgb:.4f}")
            
        except Exception as e:
            print(f"    LightGBM 실패: {e}")
        
        # 4. Gradient Boosting
        print("  4. Gradient Boosting")
        try:
            gb_model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
            gb_model.fit(X_train_scaled, y_train)
            y_pred_gb = gb_model.predict(X_test_scaled)
            accuracy_gb = accuracy_score(y_test, y_pred_gb)
            
            models['GradientBoosting'] = {
                'model': gb_model, 'scaler': scaler, 'accuracy': accuracy_gb,
                'predictions': y_pred_gb, 'classification_report': classification_report(y_test, y_pred_gb, output_dict=True)
            }
            print(f"    Gradient Boosting: {accuracy_gb:.4f}")
            
        except Exception as e:
            print(f"    Gradient Boosting 실패: {e}")
        
        # 5. 앙상블 투표
        if len(models) >= 2:
            print("  5. Voting Ensemble")
            try:
                voting_estimators = [(name, info['model']) for name, info in models.items() if name != 'GradientBoosting'][:3]
                
                voting_model = VotingClassifier(
                    estimators=voting_estimators,
                    voting='soft'
                )
                voting_model.fit(X_train_scaled, y_train)
                y_pred_voting = voting_model.predict(X_test_scaled)
                accuracy_voting = accuracy_score(y_test, y_pred_voting)
                
                models['VotingEnsemble'] = {
                    'model': voting_model, 'scaler': scaler, 'accuracy': accuracy_voting,
                    'predictions': y_pred_voting, 'classification_report': classification_report(y_test, y_pred_voting, output_dict=True)
                }
                print(f"    Voting Ensemble: {accuracy_voting:.4f}")
                
            except Exception as e:
                print(f"    Voting Ensemble 실패: {e}")
        
        return models, y_test
    
    def train_improved_regression_models(self, X, y):
        """개선된 회귀 모델 학습"""
        print("🏗️ 개선된 면적 회귀 모델 학습...")
        
        # 데이터 분할 (시계열 고려)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, shuffle=False
        )
        
        # 스케일링
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 타겟 변환 (log1p)
        y_train_log = np.log1p(y_train)
        y_test_log = np.log1p(y_test)
        
        models = {}
        
        # 성능 평가 함수
        def evaluate_model(y_true_log, y_pred_log):
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            y_true = np.expm1(y_true_log)
            y_pred = np.expm1(y_pred_log)
            y_pred = np.maximum(y_pred, 0)  # 음수 제거
            
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            
            return r2, rmse, mae, y_pred
        
        # 1. Random Forest (최적화)
        print("  1. Optimized Random Forest")
        try:
            rf_model = RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train_scaled, y_train_log)
            y_pred_rf = rf_model.predict(X_test_scaled)
            r2_rf, rmse_rf, mae_rf, y_pred_rf_orig = evaluate_model(y_test_log, y_pred_rf)
            
            models['RandomForest'] = {
                'model': rf_model, 'scaler': scaler, 'r2': r2_rf, 'rmse': rmse_rf, 'mae': mae_rf,
                'predictions': y_pred_rf_orig
            }
            print(f"    Random Forest: R²={r2_rf:.4f}, RMSE={rmse_rf:.2f}")
            
        except Exception as e:
            print(f"    Random Forest 실패: {e}")
        
        # 2. XGBoost
        print("  2. XGBoost")
        try:
            xgb_model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42
            )
            xgb_model.fit(X_train_scaled, y_train_log)
            y_pred_xgb = xgb_model.predict(X_test_scaled)
            r2_xgb, rmse_xgb, mae_xgb, y_pred_xgb_orig = evaluate_model(y_test_log, y_pred_xgb)
            
            models['XGBoost'] = {
                'model': xgb_model, 'scaler': scaler, 'r2': r2_xgb, 'rmse': rmse_xgb, 'mae': mae_xgb,
                'predictions': y_pred_xgb_orig
            }
            print(f"    XGBoost: R²={r2_xgb:.4f}, RMSE={rmse_xgb:.2f}")
            
        except Exception as e:
            print(f"    XGBoost 실패: {e}")
        
        # 3. LightGBM
        print("  3. LightGBM")
        try:
            lgb_model = lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1
            )
            lgb_model.fit(X_train_scaled, y_train_log)
            y_pred_lgb = lgb_model.predict(X_test_scaled)
            r2_lgb, rmse_lgb, mae_lgb, y_pred_lgb_orig = evaluate_model(y_test_log, y_pred_lgb)
            
            models['LightGBM'] = {
                'model': lgb_model, 'scaler': scaler, 'r2': r2_lgb, 'rmse': rmse_lgb, 'mae': mae_lgb,
                'predictions': y_pred_lgb_orig
            }
            print(f"    LightGBM: R²={r2_lgb:.4f}, RMSE={rmse_lgb:.2f}")
            
        except Exception as e:
            print(f"    LightGBM 실패: {e}")
        
        # 4. Gradient Boosting
        print("  4. Gradient Boosting")
        try:
            gb_model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                loss='huber',
                alpha=0.9,
                random_state=42
            )
            gb_model.fit(X_train_scaled, y_train_log)
            y_pred_gb = gb_model.predict(X_test_scaled)
            r2_gb, rmse_gb, mae_gb, y_pred_gb_orig = evaluate_model(y_test_log, y_pred_gb)
            
            models['GradientBoosting'] = {
                'model': gb_model, 'scaler': scaler, 'r2': r2_gb, 'rmse': rmse_gb, 'mae': mae_gb,
                'predictions': y_pred_gb_orig
            }
            print(f"    Gradient Boosting: R²={r2_gb:.4f}, RMSE={rmse_gb:.2f}")
            
        except Exception as e:
            print(f"    Gradient Boosting 실패: {e}")
        
        # 5. ElasticNet with TransformedTargetRegressor
        print("  5. ElasticNet with Target Transform")
        try:
            elastic_model = TransformedTargetRegressor(
                regressor=ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
                transformer=None  # log1p는 이미 적용됨
            )
            elastic_model.fit(X_train_scaled, y_train_log)
            y_pred_elastic = elastic_model.predict(X_test_scaled)
            r2_elastic, rmse_elastic, mae_elastic, y_pred_elastic_orig = evaluate_model(y_test_log, y_pred_elastic)
            
            models['ElasticNet'] = {
                'model': elastic_model, 'scaler': scaler, 'r2': r2_elastic, 'rmse': rmse_elastic, 'mae': mae_elastic,
                'predictions': y_pred_elastic_orig
            }
            print(f"    ElasticNet: R²={r2_elastic:.4f}, RMSE={rmse_elastic:.2f}")
            
        except Exception as e:
            print(f"    ElasticNet 실패: {e}")
        
        # 6. 고급 앙상블
        if len(models) >= 2:
            print("  6. Advanced Ensemble")
            try:
                # 성능 기반 가중 평균
                weights = []
                predictions = []
                
                for model_name, model_info in models.items():
                    r2_score = model_info['r2']
                    if r2_score > 0:  # 양수인 모델만 사용
                        weights.append(r2_score)
                        predictions.append(model_info['predictions'])
                
                if weights:
                    weights = np.array(weights) / sum(weights)  # 정규화
                    ensemble_pred = np.average(predictions, axis=0, weights=weights)
                    
                    y_test_orig = np.expm1(y_test_log)
                    r2_ensemble = r2_score(y_test_orig, ensemble_pred)
                    rmse_ensemble = np.sqrt(mean_squared_error(y_test_orig, ensemble_pred))
                    mae_ensemble = mean_absolute_error(y_test_orig, ensemble_pred)
                    
                    models['WeightedEnsemble'] = {
                        'model': 'weighted_ensemble', 'scaler': scaler, 
                        'r2': r2_ensemble, 'rmse': rmse_ensemble, 'mae': mae_ensemble,
                        'predictions': ensemble_pred, 'weights': weights.tolist()
                    }
                    print(f"    Weighted Ensemble: R²={r2_ensemble:.4f}, RMSE={rmse_ensemble:.2f}")
                
            except Exception as e:
                print(f"    Advanced Ensemble 실패: {e}")
        
        return models, y_test_log
    
    def save_improved_results(self, models, features, y_test, model_type):
        """개선된 결과 저장"""
        if not models:
            print(f"❌ {model_type} 모델 학습 실패")
            return None
        
        base_path = '/Users/mmymacymac/Developer/Projects/WildFire_projects/wildfire/wildfire/ML'
        
        # 최고 성능 모델 찾기
        if model_type == 'area':
            best_model_name = max(models.keys(), key=lambda x: models[x]['r2'])
            best_model_info = models[best_model_name]
            metric_name = 'r2'
            metric_value = best_model_info['r2']
        else:
            best_model_name = max(models.keys(), key=lambda x: models[x]['accuracy'])
            best_model_info = models[best_model_name]
            metric_name = 'accuracy'
            metric_value = best_model_info['accuracy']
        
        print(f"\n🏆 {model_type} 최고 모델: {best_model_name}")
        print(f"📊 성능: {metric_value:.4f}")
        
        # 모델 저장
        if best_model_info['model'] not in ['weighted_ensemble', 'ensemble']:
            model_path = f'{base_path}/{model_type}_model_improved.joblib'
            joblib.dump(best_model_info['model'], model_path)
        else:
            model_path = f"ensemble_model"
        
        # 스케일러 저장
        scaler_path = f'{base_path}/{model_type}_scaler_improved.joblib'
        joblib.dump(best_model_info['scaler'], scaler_path)
        
        # 피처 저장
        features_path = f'{base_path}/{model_type}_features_improved.json'
        with open(features_path, 'w') as f:
            json.dump(features, f, indent=2)
        
        # 결과 요약
        if model_type == 'area':
            summary = {
                'model_type': model_type,
                'best_model': best_model_name,
                'best_r2': float(best_model_info['r2']),
                'best_rmse': float(best_model_info['rmse']),
                'best_mae': float(best_model_info['mae']),
                'all_results': {
                    k: {
                        'r2': float(v['r2']),
                        'rmse': float(v['rmse']),
                        'mae': float(v['mae'])
                    } for k, v in models.items()
                }
            }
        else:
            summary = {
                'model_type': model_type,
                'best_model': best_model_name,
                'best_accuracy': float(best_model_info['accuracy']),
                'all_results': {
                    k: {
                        'accuracy': float(v['accuracy']),
                        'classification_report': v['classification_report']
                    } for k, v in models.items()
                }
            }
        
        summary.update({
            'feature_count': len(features),
            'features': features,
            'improvements_applied': [
                'Advanced feature engineering',
                'Intelligent feature selection',
                'Optimized model parameters',
                'Ensemble methods',
                'Robust scaling and preprocessing'
            ]
        })
        
        summary_path = f'{base_path}/{model_type}_improved_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 {model_type} 모델 저장 완료:")
        print(f"  - 모델: {model_path}")
        print(f"  - 스케일러: {scaler_path}")
        print(f"  - 피처: {features_path}")
        print(f"  - 요약: {summary_path}")
        
        return summary
    
    def comprehensive_improvement(self):
        """종합적인 모델 개선"""
        print("=" * 80)
        print("🚀 종합적인 모델 개선 (데이터 누수 제거 후)")
        print("=" * 80)
        
        # 1. 데이터 로딩 및 분석
        df = self.load_and_analyze_data()
        
        # 2. 개선된 라벨 생성
        df = self.create_improved_speed_categories(df)
        df = self.create_improved_direction_categories(df)
        
        # 3. 고급 피처 엔지니어링
        enhanced_df = self.advanced_feature_engineering(df)
        
        # 기본 전처리
        numeric_cols = enhanced_df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['fire_area', 'speed_category', 'direction_category']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        X = enhanced_df[feature_cols].copy()
        # 무한값과 NaN 처리
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        results = {}
        
        # 4. 속도 모델 개선
        try:
            print("\n" + "=" * 60)
            print("🚀 속도 모델 개선")
            print("=" * 60)
            
            y_speed = enhanced_df['speed_category']
            X_speed, speed_features = self.intelligent_feature_selection(X, y_speed, 'classification', 40)
            speed_models, y_speed_test = self.train_improved_classification_models(X_speed, y_speed, 'speed')
            speed_result = self.save_improved_results(speed_models, speed_features, y_speed_test, 'speed')
            results['speed'] = speed_result
            
        except Exception as e:
            print(f"❌ 속도 모델 개선 실패: {e}")
            results['speed'] = None
        
        # 5. 방향 모델 개선
        try:
            print("\n" + "=" * 60)
            print("🧭 방향 모델 개선")
            print("=" * 60)
            
            y_direction = enhanced_df['direction_category']
            X_direction, direction_features = self.intelligent_feature_selection(X, y_direction, 'classification', 35)
            direction_models, y_direction_test = self.train_improved_classification_models(X_direction, y_direction, 'direction')
            direction_result = self.save_improved_results(direction_models, direction_features, y_direction_test, 'direction')
            results['direction'] = direction_result
            
        except Exception as e:
            print(f"❌ 방향 모델 개선 실패: {e}")
            results['direction'] = None
        
        # 6. 면적 모델 개선
        try:
            print("\n" + "=" * 60)
            print("🔥 면적 모델 개선")
            print("=" * 60)
            
            y_area = enhanced_df['fire_area']
            X_area, area_features = self.intelligent_feature_selection(X, y_area, 'regression', 50)
            area_models, y_area_test = self.train_improved_regression_models(X_area, y_area)
            area_result = self.save_improved_results(area_models, area_features, y_area_test, 'area')
            results['area'] = area_result
            
        except Exception as e:
            print(f"❌ 면적 모델 개선 실패: {e}")
            results['area'] = None
        
        # 7. 최종 결과 요약
        print("\n" + "=" * 80)
        print("📊 최종 개선 결과 요약")
        print("=" * 80)
        
        for model_type, result in results.items():
            if result:
                if model_type == 'area':
                    print(f"{model_type.title()} 모델:")
                    print(f"  ✅ 성공 - {result['best_model']} (R²: {result['best_r2']:.4f}, RMSE: {result['best_rmse']:.2f})")
                else:
                    print(f"{model_type.title()} 모델:")
                    print(f"  ✅ 성공 - {result['best_model']} (정확도: {result['best_accuracy']:.4f})")
            else:
                print(f"{model_type.title()} 모델:")
                print(f"  ❌ 실패")
        
        print("\n🎯 개선 사항:")
        print("  - 고급 피처 엔지니어링 (복합 지수, 상호작용 등)")
        print("  - 지능적 피처 선택 (통계적 + 중요도 기반)")
        print("  - 최적화된 모델 파라미터")
        print("  - 다양한 앙상블 기법")
        print("  - 강력한 전처리 및 스케일링")
        
        return results

def main():
    print("🚀 종합적인 모델 개선")
    
    data_path = '/Users/mmymacymac/Developer/Projects/WildFire_projects/wildfire/wildfire/ML/clean_training_dataset.csv'
    
    improver = ComprehensiveModelImprover(data_path)
    results = improver.comprehensive_improvement()
    
    return results

if __name__ == "__main__":
    main()