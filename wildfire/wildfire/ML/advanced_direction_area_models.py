#!/usr/bin/env python3
"""
🔥 고급 방향 및 면적 모델 개선
- 딥러닝 및 고급 앙상블 기법 적용
- 시계열 특성 강화
- 스태킹 앙상블 및 메타 러닝
"""

import pandas as pd
import numpy as np
import warnings
import joblib
import json
from pathlib import Path
from datetime import datetime

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import RobustScaler, LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import classification_report, accuracy_score, r2_score, mean_squared_error, mean_absolute_error

# Advanced ML
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor

# Feature Engineering
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

class AdvancedDirectionAreaModels:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def load_and_analyze_data(self):
        """데이터 로딩 및 심층 분석"""
        print("🔥 고급 방향 및 면적 모델 개선")
        print("=" * 80)
        
        # 데이터 로딩
        df = pd.read_csv('/Users/mmymacymac/Developer/Projects/WildFire_projects/wildfire/wildfire/ML/clean_training_dataset.csv')
        print(f"📊 데이터 크기: {df.shape}")
        
        # 기본 전처리
        df = df.dropna(subset=['fire_area'])
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        print(f"📊 정리된 데이터 크기: {df.shape}")
        
        return df
    
    def create_advanced_categories(self, df):
        """고급 카테고리 생성 - 더 정교한 방법"""
        print("\n🎯 고급 카테고리 생성...")
        
        # 방향 카테고리 - 지형과 기상을 종합적으로 고려
        if all(col in df.columns for col in ['aspect_mode', 'wd2m_0h', 'elevation_mean']):
            # 지형 방향성 (0-360도)
            terrain_direction = df['aspect_mode'].fillna(0)
            
            # 풍향 (0-360도)  
            wind_direction = df['wd2m_0h'].fillna(0)
            
            # 고도 영향
            elevation_factor = df['elevation_mean'].fillna(df['elevation_mean'].median())
            elevation_normalized = (elevation_factor - elevation_factor.min()) / (elevation_factor.max() - elevation_factor.min())
            
            # 복합 방향 점수 계산
            def get_direction_complex(terrain_dir, wind_dir, elev_factor):
                # 지형과 풍향의 상호작용
                terrain_weight = 0.6 + elev_factor * 0.2  # 고도가 높을수록 지형 영향 증가
                wind_weight = 1 - terrain_weight
                
                # 방향별 위험도 (실제 산불 확산 패턴 기반)
                combined_direction = terrain_dir * terrain_weight + wind_dir * wind_weight
                combined_direction = combined_direction % 360
                
                if combined_direction < 45 or combined_direction >= 315:
                    return 'north'
                elif 45 <= combined_direction < 135:
                    return 'east' 
                elif 135 <= combined_direction < 225:
                    return 'south'
                else:
                    return 'west'
            
            df['direction_category'] = [get_direction_complex(t, w, e) for t, w, e in 
                                      zip(terrain_direction, wind_direction, elevation_normalized)]
        else:
            # 폴백 방법
            df['direction_category'] = 'north'
        
        print("  고급 방향 분포:")
        print(df['direction_category'].value_counts())
        
        return df
    
    def create_advanced_features(self, df):
        """고급 피처 엔지니어링"""
        print("\n🔧 고급 피처 엔지니어링...")
        
        enhanced_df = df.copy()
        
        # 1. 시계열 특성 강화
        print("  1. 시계열 특성 강화")
        
        # 기상 변수의 시간별 변화율
        weather_vars = ['t2m', 'rh2m', 'ws2m', 'ps']
        for var in weather_vars:
            time_cols = [col for col in df.columns if var in col and 'past' in col]
            if len(time_cols) >= 3:
                time_cols_sorted = sorted(time_cols, key=lambda x: int(x.split('h_past')[0].split('_')[-1]))[:10]
                
                # 변화율 계산
                for i in range(1, len(time_cols_sorted)):
                    if time_cols_sorted[i] in df.columns and time_cols_sorted[i-1] in df.columns:
                        enhanced_df[f'{var}_rate_{i}'] = (df[time_cols_sorted[i]] - df[time_cols_sorted[i-1]]) / (df[time_cols_sorted[i-1]] + 1e-8)
                
                # 트렌드 분석
                if len(time_cols_sorted) >= 5:
                    values = df[time_cols_sorted[:5]].values
                    enhanced_df[f'{var}_trend'] = np.array([np.polyfit(range(5), row, 1)[0] if not np.isnan(row).any() else 0 for row in values])
        
        # 2. 공간적 클러스터링
        print("  2. 공간적 클러스터링")
        
        if 'start_latitude' in df.columns and 'start_longitude' in df.columns:
            coords = df[['start_latitude', 'start_longitude']].fillna(df[['start_latitude', 'start_longitude']].median())
            
            # K-means 클러스터링으로 지역 구분
            kmeans = KMeans(n_clusters=8, random_state=42)
            enhanced_df['spatial_cluster'] = kmeans.fit_predict(coords)
            
            # 클러스터별 특성
            for cluster in range(8):
                mask = enhanced_df['spatial_cluster'] == cluster
                if mask.sum() > 0:
                    enhanced_df[f'cluster_{cluster}_risk'] = mask.astype(int)
        
        # 3. 고급 기상 복합 지수
        print("  3. 고급 기상 복합 지수")
        
        # Haines Index (대기 안정도)
        if 't2m_0h' in df.columns and 'rh2m_0h' in df.columns:
            temp = df['t2m_0h'].fillna(df['t2m_0h'].median())
            humidity = df['rh2m_0h'].fillna(df['rh2m_0h'].median())
            
            # 기온-습도 안정도 지수
            enhanced_df['haines_index'] = (temp - 273.15) / 10 + (100 - humidity) / 20
            
        # 4. 지형 복잡도 지수
        print("  4. 지형 복잡도 고급 분석")
        
        terrain_vars = ['elevation', 'slope', 'aspect']
        for var in terrain_vars:
            cols = [col for col in df.columns if var in col]
            if len(cols) >= 2:
                terrain_data = df[cols].fillna(df[cols].median())
                
                # 지형 변화 강도
                enhanced_df[f'{var}_variation'] = terrain_data.std(axis=1)
                enhanced_df[f'{var}_range'] = terrain_data.max(axis=1) - terrain_data.min(axis=1)
        
        # 5. 비선형 상호작용
        print("  5. 비선형 상호작용 피처")
        
        # 온도-바람 상호작용 (비선형)
        if 't2m_0h' in df.columns and 'ws2m_0h' in df.columns:
            temp = df['t2m_0h'].fillna(df['t2m_0h'].median())
            wind = df['ws2m_0h'].fillna(df['ws2m_0h'].median())
            
            enhanced_df['temp_wind_nonlinear'] = np.log1p(temp * wind**2)
            enhanced_df['temp_wind_ratio'] = temp / (wind + 1e-8)
        
        print(f"  고급 피처 엔지니어링 완료: {enhanced_df.shape[1] - df.shape[1]}개 피처 추가")
        
        return enhanced_df
    
    def advanced_feature_selection(self, X, y, task_type='classification', max_features=50):
        """고급 피처 선택 - 다단계 접근"""
        print(f"\n🎯 고급 피처 선택 (최대 {max_features}개)...")
        
        # 1. 기본 필터링
        print("  1. 기본 전처리")
        
        # 분산이 0인 컬럼 제거
        zero_var_cols = X.columns[X.var() == 0].tolist()
        X = X.drop(columns=zero_var_cols)
        print(f"    분산 0인 컬럼 제거: {len(zero_var_cols)}개")
        
        # 결측치가 너무 많은 컬럼 제거 (50% 이상)
        missing_threshold = 0.5
        missing_cols = X.columns[X.isnull().mean() > missing_threshold].tolist()
        X = X.drop(columns=missing_cols)
        print(f"    결측치 많은 컬럼 제거: {len(missing_cols)}개")
        
        # 무한값 처리
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # 2. 통계적 피처 선택
        print("  2. 통계적 피처 선택")
        
        if task_type == 'classification':
            selector = SelectKBest(score_func=f_classif, k=min(max_features * 2, X.shape[1]))
        else:
            selector = SelectKBest(score_func=f_regression, k=min(max_features * 2, X.shape[1]))
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        X = X[selected_features]
        print(f"    통계적 선택: {len(selected_features)}개")
        
        # 3. 모델 기반 중요도 선택
        print("  3. 모델 기반 중요도 선택")
        
        if task_type == 'classification':
            importance_model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            importance_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        importance_model.fit(X, y)
        importances = importance_model.feature_importances_
        
        # 중요도 기반 선택
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        final_features = feature_importance.head(max_features)['feature'].tolist()
        X_final = X[final_features]
        
        print(f"    최종 선택: {len(final_features)}개")
        print(f"    상위 5개: {final_features[:5]}")
        
        return X_final, final_features
    
    def train_advanced_direction_model(self, df):
        """고급 방향 모델 훈련 - 딥러닝 및 스태킹"""
        print("\n" + "=" * 60)
        print("🧭 고급 방향 모델 훈련")
        print("=" * 60)
        
        # 데이터 준비
        X = df.drop(['fire_area', 'direction_category'], axis=1, errors='ignore')
        y = df['direction_category']
        
        # 레이블 인코딩
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # 피처 선택
        X_selected, selected_features = self.advanced_feature_selection(X, y_encoded, 'classification', 45)
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
        )
        
        # 스케일링
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("🏗️ 고급 방향 모델 훈련...")
        
        models = {}
        
        # 1. 기본 모델들
        base_models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ),
            'ExtraTrees': ExtraTreesClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=150, max_depth=8, learning_rate=0.1,
                subsample=0.8, random_state=42
            )
        }
        
        # 2. 신경망 모델
        print("  1. 신경망 모델")
        try:
            mlp = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32), activation='relu',
                solver='adam', alpha=0.001, learning_rate='adaptive',
                max_iter=500, random_state=42
            )
            mlp.fit(X_train_scaled, y_train)
            mlp_score = mlp.score(X_test_scaled, y_test)
            models['MLP'] = {'model': mlp, 'accuracy': mlp_score}
            print(f"    MLP: {mlp_score:.4f}")
        except Exception as e:
            print(f"    MLP 실패: {str(e)}")
        
        # 3. 기본 모델 훈련
        for name, model in base_models.items():
            print(f"  2. {name}")
            try:
                model.fit(X_train_scaled, y_train)
                score = model.score(X_test_scaled, y_test)
                models[name] = {'model': model, 'accuracy': score}
                print(f"    {name}: {score:.4f}")
            except Exception as e:
                print(f"    {name} 실패: {str(e)}")
        
        # 4. 스태킹 앙상블
        print("  3. 스태킹 앙상블")
        try:
            # 베이스 모델들 선택 (상위 3개)
            best_models = sorted(models.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:3]
            
            if len(best_models) >= 2:
                base_estimators = [(name, info['model']) for name, info in best_models]
                
                stacking_model = StackingClassifier(
                    estimators=base_estimators,
                    final_estimator=LogisticRegression(random_state=42),
                    cv=5
                )
                
                stacking_model.fit(X_train_scaled, y_train)
                stacking_score = stacking_model.score(X_test_scaled, y_test)
                models['StackingEnsemble'] = {'model': stacking_model, 'accuracy': stacking_score}
                print(f"    Stacking: {stacking_score:.4f}")
        except Exception as e:
            print(f"    Stacking 실패: {str(e)}")
        
        # 최고 모델 선택
        if models:
            best_model_name = max(models.keys(), key=lambda k: models[k]['accuracy'])
            best_model = models[best_model_name]['model']
            best_accuracy = models[best_model_name]['accuracy']
            
            print(f"\n🏆 direction 최고 모델: {best_model_name}")
            print(f"📊 성능: {best_accuracy:.4f}")
            
            # 상세 평가
            y_pred = best_model.predict(X_test_scaled)
            report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
            
            # 모델 저장
            model_path = f'/Users/mmymacymac/Developer/Projects/WildFire_projects/wildfire/wildfire/ML/direction_model_advanced.joblib'
            scaler_path = f'/Users/mmymacymac/Developer/Projects/WildFire_projects/wildfire/wildfire/ML/direction_scaler_advanced.joblib'
            encoder_path = f'/Users/mmymacymac/Developer/Projects/WildFire_projects/wildfire/wildfire/ML/direction_encoder_advanced.joblib'
            features_path = f'/Users/mmymacymac/Developer/Projects/WildFire_projects/wildfire/wildfire/ML/direction_features_advanced.json'
            summary_path = f'/Users/mmymacymac/Developer/Projects/WildFire_projects/wildfire/wildfire/ML/direction_advanced_summary.json'
            
            joblib.dump(best_model, model_path)
            joblib.dump(scaler, scaler_path)
            joblib.dump(label_encoder, encoder_path)
            
            with open(features_path, 'w') as f:
                json.dump(selected_features, f, indent=2)
            
            # 요약 저장
            summary = {
                'model_type': 'direction_advanced',
                'best_model': best_model_name,
                'best_accuracy': best_accuracy,
                'all_results': {name: {'accuracy': info['accuracy']} for name, info in models.items()},
                'classification_report': report,
                'feature_count': len(selected_features),
                'features': selected_features[:10],  # 상위 10개만
                'improvements_applied': [
                    'Deep Learning (MLP)',
                    'Stacking Ensemble',
                    'Advanced Feature Engineering',
                    'Spatial Clustering',
                    'Time Series Features'
                ]
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n💾 direction 고급 모델 저장 완료:")
            print(f"  - 모델: {model_path}")
            print(f"  - 스케일러: {scaler_path}")
            print(f"  - 인코더: {encoder_path}")
            print(f"  - 피처: {features_path}")
            print(f"  - 요약: {summary_path}")
            
            return best_accuracy
        
        return None
    
    def train_advanced_area_model(self, df):
        """고급 면적 모델 훈련 - 시계열 및 고급 앙상블"""
        print("\n" + "=" * 60)
        print("🔥 고급 면적 모델 훈련")
        print("=" * 60)
        
        # 데이터 준비
        X = df.drop(['fire_area', 'direction_category'], axis=1, errors='ignore')
        y = df['fire_area']
        
        # 피처 선택
        X_selected, selected_features = self.advanced_feature_selection(X, y, 'regression', 60)
        
        # 시계열 분할 (시간 정보가 있다면)
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.25, random_state=42, shuffle=False
        )
        
        # 스케일링
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 타겟 변환
        y_train_log = np.log1p(y_train)
        y_test_log = np.log1p(y_test)
        
        print("🏗️ 고급 면적 회귀 모델 훈련...")
        
        models = {}
        
        # 성능 평가 함수
        def evaluate_model(y_true_log, y_pred_log):
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            y_true = np.expm1(y_true_log)
            y_pred = np.expm1(y_pred_log)
            y_pred = np.maximum(y_pred, 0)
            
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            
            return r2, rmse, mae
        
        # 1. 고급 회귀 모델들
        advanced_models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=300, max_depth=20, min_samples_split=3,
                min_samples_leaf=1, random_state=42, n_jobs=-1
            ),
            'ExtraTrees': ExtraTreesRegressor(
                n_estimators=300, max_depth=20, min_samples_split=3,
                min_samples_leaf=1, random_state=42, n_jobs=-1
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=300, max_depth=10, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=300, max_depth=10, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=200, max_depth=10, learning_rate=0.05,
                subsample=0.8, random_state=42
            )
        }
        
        # 2. 신경망 모델
        print("  1. 신경망 모델")
        try:
            mlp = MLPRegressor(
                hidden_layer_sizes=(256, 128, 64, 32), activation='relu',
                solver='adam', alpha=0.001, learning_rate='adaptive',
                max_iter=1000, random_state=42
            )
            mlp.fit(X_train_scaled, y_train_log)
            y_pred_mlp = mlp.predict(X_test_scaled)
            r2_mlp, rmse_mlp, mae_mlp = evaluate_model(y_test_log, y_pred_mlp)
            models['MLP'] = {'model': mlp, 'r2': r2_mlp, 'rmse': rmse_mlp, 'mae': mae_mlp}
            print(f"    MLP: R²={r2_mlp:.4f}, RMSE={rmse_mlp:.2f}")
        except Exception as e:
            print(f"    MLP 실패: {str(e)}")
        
        # 3. 기본 모델 훈련
        for name, model in advanced_models.items():
            print(f"  2. {name}")
            try:
                model.fit(X_train_scaled, y_train_log)
                y_pred = model.predict(X_test_scaled)
                r2, rmse, mae = evaluate_model(y_test_log, y_pred)
                models[name] = {'model': model, 'r2': r2, 'rmse': rmse, 'mae': mae}
                print(f"    {name}: R²={r2:.4f}, RMSE={rmse:.2f}")
            except Exception as e:
                print(f"    {name} 실패: {str(e)}")
        
        # 4. 스태킹 회귀
        print("  3. 스태킹 회귀")
        try:
            # 상위 모델들 선택
            best_models = sorted(models.items(), key=lambda x: x[1]['r2'], reverse=True)[:3]
            
            if len(best_models) >= 2:
                base_estimators = [(name, info['model']) for name, info in best_models]
                
                stacking_model = StackingRegressor(
                    estimators=base_estimators,
                    final_estimator=Ridge(alpha=1.0),
                    cv=TimeSeriesSplit(n_splits=5)
                )
                
                stacking_model.fit(X_train_scaled, y_train_log)
                y_pred_stack = stacking_model.predict(X_test_scaled)
                r2_stack, rmse_stack, mae_stack = evaluate_model(y_test_log, y_pred_stack)
                models['StackingRegressor'] = {'model': stacking_model, 'r2': r2_stack, 'rmse': rmse_stack, 'mae': mae_stack}
                print(f"    Stacking: R²={r2_stack:.4f}, RMSE={rmse_stack:.2f}")
        except Exception as e:
            print(f"    Stacking 실패: {str(e)}")
        
        # 5. 앙상블 보팅
        print("  4. 앙상블 보팅")
        try:
            if len(models) >= 3:
                # 상위 3개 모델로 보팅
                top_models = sorted(models.items(), key=lambda x: x[1]['r2'], reverse=True)[:3]
                voting_estimators = [(name, info['model']) for name, info in top_models]
                
                voting_model = VotingRegressor(estimators=voting_estimators)
                voting_model.fit(X_train_scaled, y_train_log)
                y_pred_vote = voting_model.predict(X_test_scaled)
                r2_vote, rmse_vote, mae_vote = evaluate_model(y_test_log, y_pred_vote)
                models['VotingRegressor'] = {'model': voting_model, 'r2': r2_vote, 'rmse': rmse_vote, 'mae': mae_vote}
                print(f"    Voting: R²={r2_vote:.4f}, RMSE={rmse_vote:.2f}")
        except Exception as e:
            print(f"    Voting 실패: {str(e)}")
        
        # 최고 모델 선택
        if models:
            best_model_name = max(models.keys(), key=lambda k: models[k]['r2'])
            best_model = models[best_model_name]['model']
            best_r2 = models[best_model_name]['r2']
            best_rmse = models[best_model_name]['rmse']
            
            print(f"\n🏆 area 최고 모델: {best_model_name}")
            print(f"📊 성능: R²={best_r2:.4f}, RMSE={best_rmse:.2f}")
            
            # 모델 저장
            model_path = f'/Users/mmymacymac/Developer/Projects/WildFire_projects/wildfire/wildfire/ML/area_model_advanced.joblib'
            scaler_path = f'/Users/mmymacymac/Developer/Projects/WildFire_projects/wildfire/wildfire/ML/area_scaler_advanced.joblib'
            features_path = f'/Users/mmymacymac/Developer/Projects/WildFire_projects/wildfire/wildfire/ML/area_features_advanced.json'
            summary_path = f'/Users/mmymacymac/Developer/Projects/WildFire_projects/wildfire/wildfire/ML/area_advanced_summary.json'
            
            joblib.dump(best_model, model_path)
            joblib.dump(scaler, scaler_path)
            
            with open(features_path, 'w') as f:
                json.dump(selected_features, f, indent=2)
            
            # 요약 저장
            summary = {
                'model_type': 'area_advanced',
                'best_model': best_model_name,
                'best_r2': best_r2,
                'best_rmse': best_rmse,
                'best_mae': models[best_model_name]['mae'],
                'all_results': {name: {k: v for k, v in info.items() if k != 'model'} for name, info in models.items()},
                'feature_count': len(selected_features),
                'features': selected_features[:15],  # 상위 15개만
                'improvements_applied': [
                    'Deep Learning (MLP)',
                    'Stacking Regression',
                    'Voting Ensemble',
                    'Time Series Cross-Validation',
                    'Advanced Feature Engineering',
                    'Log Transform with Robust Scaling'
                ]
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n💾 area 고급 모델 저장 완료:")
            print(f"  - 모델: {model_path}")
            print(f"  - 스케일러: {scaler_path}")
            print(f"  - 피처: {features_path}")
            print(f"  - 요약: {summary_path}")
            
            return best_r2
        
        return None
    
    def run_advanced_training(self):
        """고급 모델 훈련 실행"""
        print("🚀 고급 방향 및 면적 모델 개선 시작")
        print("=" * 80)
        
        # 데이터 로딩
        df = self.load_and_analyze_data()
        
        # 고급 카테고리 생성
        df = self.create_advanced_categories(df)
        
        # 고급 피처 엔지니어링
        df = self.create_advanced_features(df)
        
        # 방향 모델 훈련
        direction_score = self.train_advanced_direction_model(df)
        
        # 면적 모델 훈련
        area_score = self.train_advanced_area_model(df)
        
        # 최종 요약
        print("\n" + "=" * 80)
        print("📊 고급 모델 개선 결과 요약")
        print("=" * 80)
        
        if direction_score:
            print(f"Direction 모델:")
            print(f"  ✅ 성공 - 정확도: {direction_score:.4f}")
        else:
            print(f"Direction 모델:")
            print(f"  ❌ 실패")
        
        if area_score:
            print(f"Area 모델:")
            print(f"  ✅ 성공 - R²: {area_score:.4f}")
        else:
            print(f"Area 모델:")
            print(f"  ❌ 실패")
        
        print(f"\n🎯 고급 개선 사항:")
        print(f"  - 딥러닝 모델 (MLP) 적용")
        print(f"  - 스태킹 앙상블 및 보팅 앙상블")
        print(f"  - 시계열 특성 강화")
        print(f"  - 공간적 클러스터링")
        print(f"  - 고급 기상 복합 지수")
        print(f"  - 비선형 상호작용 피처")


if __name__ == "__main__":
    trainer = AdvancedDirectionAreaModels()
    trainer.run_advanced_training()