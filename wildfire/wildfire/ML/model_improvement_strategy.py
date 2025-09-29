import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class ModelImprovementStrategy:
    """모델 정확도 향상을 위한 종합 전략"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.improvement_strategies = {
            '피처_엔지니어링': self.feature_engineering_strategies,
            '데이터_전처리': self.data_preprocessing_strategies, 
            '모델_선택': self.model_selection_strategies,
            '하이퍼파라미터_튜닝': self.hyperparameter_tuning_strategies,
            '앙상블_기법': self.ensemble_strategies,
            '데이터_증강': self.data_augmentation_strategies
        }
    
    def load_data(self):
        """데이터 로드"""
        self.df = pd.read_csv(self.data_path)
        print(f"데이터 크기: {self.df.shape}")
        return self.df
    
    def feature_engineering_strategies(self):
        """피처 엔지니어링 전략"""
        strategies = {
            "1. 상호작용 피처 생성": [
                "온도 × 습도 = 열 스트레스 지수",
                "바람속도 × 건조도 = 확산 위험도",
                "경사 × 향 = 지형 위험도",
                "연료량 × 건조일수 = 연소 잠재력"
            ],
            
            "2. 시간적 피처": [
                "계절별 가중치 (봄/가을 화재 위험)",
                "월별 순환 피처 (sin/cos 변환)",
                "일별 변화율 (온도, 습도 변화량)",
                "과거 7일/14일/30일 평균 및 표준편차"
            ],
            
            "3. 지형 복합 지수": [
                "지형 거칠기 지수 = slope_std × elevation_std",
                "남향 경사면 비율 × 평균 경사도",
                "고도 변화율 = (max_elevation - min_elevation) / distance",
                "배수 효과 지수 (물 접근성)"
            ],
            
            "4. 기상 복합 지수": [
                "Enhanced FWI (기존 FWI 개선)",
                "Drought Index (장기 건조 지수)",
                "Wind Penetration Index (바람 침투도)",
                "Temperature-Humidity Index (열-습도 복합지수)"
            ],
            
            "5. 식생 위험 지수": [
                "연료 연속성 지수 = treecover × ndvi",
                "연료 건조도 = (1 - ndvi) × dry_days",
                "식생 밀도 변화율",
                "연료 모델 분류 (침엽수/활엽수/혼합)"
            ]
        }
        
        return strategies
    
    def create_interaction_features(self, df):
        """상호작용 피처 생성"""
        enhanced_df = df.copy()
        
        # 기상 상호작용
        if 't2m_0h' in df.columns and 'rh2m_0h' in df.columns:
            enhanced_df['heat_stress_index'] = df['t2m_0h'] * (100 - df['rh2m_0h']) / 100
        
        if 'ws10m_0h' in df.columns and 'rh2m_0h' in df.columns:
            enhanced_df['wind_dryness'] = df['ws10m_0h'] * (100 - df['rh2m_0h']) / 100
        
        # 지형 상호작용
        if 'slope_mean' in df.columns and 'aspect_south_ratio' in df.columns:
            enhanced_df['terrain_fire_risk'] = df['slope_mean'] * df['aspect_south_ratio']
        
        if 'elevation_std' in df.columns and 'slope_std' in df.columns:
            enhanced_df['terrain_roughness'] = df['elevation_std'] * df['slope_std']
        
        # 연료 상호작용
        if 'treecover_pre_fire_5x5' in df.columns and 'ndvi_before' in df.columns:
            enhanced_df['fuel_continuity'] = df['treecover_pre_fire_5x5'] * df['ndvi_before']
        
        if 'dry_days_30d_start' in df.columns and 'ndvi_before' in df.columns:
            enhanced_df['fuel_dryness'] = df['dry_days_30d_start'] * (1 - df['ndvi_before'])
        
        # 시간적 피처
        if 'fire_month' in df.columns:
            enhanced_df['month_sin'] = np.sin(2 * np.pi * df['fire_month'] / 12)
            enhanced_df['month_cos'] = np.cos(2 * np.pi * df['fire_month'] / 12)
        
        # FWI 개선
        base_fwi_cols = ['ffmc_0h', 'dmc_0h', 'dc_0h', 'isi_0h', 'bui_0h', 'fwi_0h']
        available_fwi = [col for col in base_fwi_cols if col in df.columns]
        
        if len(available_fwi) >= 3:
            enhanced_df['enhanced_fwi'] = 0
            for col in available_fwi:
                enhanced_df['enhanced_fwi'] += df[col] * np.random.uniform(0.8, 1.2)  # 가중치 적용
        
        print(f"상호작용 피처 생성 후: {len(enhanced_df.columns) - len(df.columns)}개 피처 추가")
        return enhanced_df
    
    def data_preprocessing_strategies(self):
        """데이터 전처리 전략"""
        strategies = {
            "1. 이상치 처리": [
                "IQR 기반 이상치 제거",
                "Robust Scaler 사용 (중앙값/IQR 기반)",
                "Winsorization (극값 제한)",
                "Log/Box-Cox 변환으로 정규화"
            ],
            
            "2. 결측치 고도화": [
                "KNN Imputation (유사 샘플 기반)",
                "Iterative Imputation (연쇄 회귀)",
                "지역별/계절별 그룹 평균",
                "시계열 보간법 (과거 패턴 활용)"
            ],
            
            "3. 피처 스케일링": [
                "RobustScaler (이상치에 강함)",
                "QuantileTransformer (균등분포 변환)",
                "PowerTransformer (정규분포 변환)",
                "피처별 맞춤 변환"
            ],
            
            "4. 차원 축소": [
                "PCA (주성분 분석)",
                "Feature Selection (상위 K개 선택)",
                "Recursive Feature Elimination",
                "L1 정규화를 통한 희소성 유도"
            ]
        }
        
        return strategies
    
    def model_selection_strategies(self):
        """모델 선택 전략"""
        strategies = {
            "1. 트리 기반 모델": {
                "RandomForest": "높은 안정성, 피처 중요도 제공",
                "GradientBoosting": "순차적 학습으로 높은 성능",
                "XGBoost": "최적화된 그래디언트 부스팅",
                "LightGBM": "빠른 속도와 메모리 효율성",
                "CatBoost": "범주형 변수 자동 처리"
            },
            
            "2. 선형 모델": {
                "ElasticNet": "L1+L2 정규화로 피처 선택",
                "Ridge": "L2 정규화로 과적합 방지",
                "Lasso": "L1 정규화로 희소 모델",
                "Bayesian Ridge": "불확실성 추정 가능"
            },
            
            "3. 신경망": {
                "MLP": "비선형 패턴 학습 가능",
                "TabNet": "테이블 데이터 특화 신경망",
                "Deep Forest": "깊은 랜덤 포레스트"
            },
            
            "4. 특수 모델": {
                "Support Vector Regression": "복잡한 경계면",
                "Gaussian Process": "불확실성 정량화",
                "Quantile Regression": "분위수 예측"
            }
        }
        
        return strategies
    
    def hyperparameter_tuning_strategies(self):
        """하이퍼파라미터 튜닝 전략"""
        strategies = {
            "1. 탐색 방법": [
                "GridSearchCV: 전체 조합 탐색",
                "RandomizedSearchCV: 랜덤 샘플링",
                "BayesianOptimization: 베이지안 최적화", 
                "Optuna: 진화적 최적화"
            ],
            
            "2. 교차 검증": [
                "StratifiedKFold: 층화 표본 추출",
                "TimeSeriesSplit: 시계열 데이터용",
                "GroupKFold: 그룹별 분할",
                "RepeatedKFold: 반복 교차 검증"
            ],
            
            "3. 평가 지표": [
                "회귀: R², RMSE, MAE, MAPE",
                "분류: Accuracy, F1, ROC-AUC, Precision/Recall",
                "비즈니스 지표: 실제 화재 대응 효과성"
            ]
        }
        
        return strategies
    
    def ensemble_strategies(self):
        """앙상블 전략"""
        strategies = {
            "1. 기본 앙상블": [
                "Voting: 여러 모델 결과 평균/투표",
                "Bagging: Bootstrap 샘플링 + 집계",
                "Boosting: 순차적 오류 개선",
                "Stacking: 메타 모델로 조합"
            ],
            
            "2. 고급 앙상블": [
                "Blending: holdout 데이터로 조합",
                "Multi-level Stacking: 다단계 스태킹",
                "Dynamic Ensemble: 입력에 따라 가중치 조정",
                "Diversity Ensemble: 다양성 극대화"
            ],
            
            "3. 모델 다양성": [
                "알고리즘 다양성: RF + GBM + Linear",
                "피처 다양성: 다른 피처 서브셋 사용",
                "데이터 다양성: 다른 샘플링 전략",
                "하이퍼파라미터 다양성: 다른 설정"
            ]
        }
        
        return strategies
    
    def data_augmentation_strategies(self):
        """데이터 증강 전략"""
        strategies = {
            "1. 합성 데이터 생성": [
                "SMOTE: 소수 클래스 오버샘플링",
                "ADASYN: 적응적 합성 샘플링",
                "GAN: 생성적 적대 신경망",
                "VAE: 변분 오토인코더"
            ],
            
            "2. 노이즈 추가": [
                "Gaussian Noise: 가우시안 잡음 추가",
                "Feature Dropout: 랜덤 피처 제거",
                "Mixup: 샘플 간 선형 조합",
                "CutMix: 피처 일부 교체"
            ],
            
            "3. 시뮬레이션 데이터": [
                "물리 기반 모델: 화재 확산 시뮬레이션",
                "Monte Carlo: 확률적 시나리오 생성",
                "Weather Generation: 기상 시나리오 생성",
                "Terrain Perturbation: 지형 변형"
            ]
        }
        
        return strategies
    
    def implement_quick_improvements(self, df, target_col='fire_area'):
        """즉시 적용 가능한 개선사항 구현"""
        print("빠른 개선사항 적용 중...")
        
        # 1. 상호작용 피처 생성
        enhanced_df = self.create_interaction_features(df)
        
        # 2. 타겟 변수 로그 변환
        X = enhanced_df.drop(columns=[target_col])
        y = enhanced_df[target_col]
        y_log = np.log1p(y)  # log(1+x) 변환
        
        # 3. 수치형 피처만 선택
        X_numeric = X.select_dtypes(include=[np.number])
        X_numeric = X_numeric.fillna(X_numeric.median())
        
        # 4. 피처 선택 (상위 50개)
        selector = SelectKBest(score_func=f_regression, k=min(50, X_numeric.shape[1]))
        X_selected = selector.fit_transform(X_numeric, y_log)
        selected_features = X_numeric.columns[selector.get_support()]
        
        print(f"선택된 피처 수: {len(selected_features)}")
        
        # 5. 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_log, test_size=0.2, random_state=42
        )
        
        # 6. 강건한 스케일링
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 7. 다양한 모델 테스트
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=8, random_state=42),
            'ExtraTrees': ExtraTreesRegressor(n_estimators=200, max_depth=15, random_state=42),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        }
        
        results = {}
        best_model = None
        best_score = float('-inf')
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred_log = model.predict(X_test_scaled)
            y_pred = np.expm1(y_pred_log)
            y_test_original = np.expm1(y_test)
            
            r2 = r2_score(y_test_original, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
            
            results[name] = {'r2': r2, 'rmse': rmse}
            
            if r2 > best_score:
                best_score = r2
                best_model = (name, model, scaler, selected_features)
        
        return results, best_model
    
    def print_comprehensive_strategy(self):
        """종합 개선 전략 출력"""
        print("\\n" + "="*60)
        print("🚀 모델 정확도 향상을 위한 종합 전략")
        print("="*60)
        
        for strategy_name, strategy_func in self.improvement_strategies.items():
            print(f"\\n📋 {strategy_name.replace('_', ' ').upper()}")
            print("-" * 40)
            
            strategies = strategy_func()
            
            if isinstance(strategies, dict):
                for category, items in strategies.items():
                    print(f"\\n  🔸 {category}")
                    if isinstance(items, list):
                        for item in items:
                            print(f"    • {item}")
                    elif isinstance(items, dict):
                        for key, value in items.items():
                            print(f"    • {key}: {value}")
                    else:
                        print(f"    • {items}")

def main():
    print("모델 개선 전략 가이드")
    
    # 전략 개요 출력
    data_path = '/Users/mmymacymac/Developer/Projects/WildFire_projects/wildfire/wildfire/ML/clean_training_dataset.csv'
    improver = ModelImprovementStrategy(data_path)
    improver.print_comprehensive_strategy()
    
    # 즉시 개선 사항 테스트
    print("\\n" + "="*60)
    print("🧪 즉시 적용 개선사항 테스트")
    print("="*60)
    
    df = improver.load_data()
    results, best_model = improver.implement_quick_improvements(df)
    
    print("\\n📊 모델 성능 비교:")
    for model_name, metrics in results.items():
        print(f"  {model_name:15s}: R² = {metrics['r2']:6.4f}, RMSE = {metrics['rmse']:8.2f}")
    
    print(f"\\n🏆 최고 성능 모델: {best_model[0]} (R² = {results[best_model[0]]['r2']:.4f})")
    
    print("\\n💡 추가 개선 권장사항:")
    print("  1. XGBoost/LightGBM 같은 고급 부스팅 모델 시도")
    print("  2. 베이지안 최적화를 통한 하이퍼파라미터 튜닝")
    print("  3. 앙상블 모델로 여러 모델 조합")
    print("  4. 도메인 지식 기반 피처 엔지니어링 강화")
    print("  5. 더 많은 데이터 수집 또는 합성 데이터 생성")

if __name__ == "__main__":
    main()