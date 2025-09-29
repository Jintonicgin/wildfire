#!/usr/bin/env python3
"""
Area v3 tuned 모델 테스트 - 94.5% R² 성능 검증
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def test_area_v3_model():
    """Area v3 tuned 모델 테스트"""
    print("🎯 Area v3 Tuned 모델 성능 검증")
    print("=" * 50)
    
    try:
        # 모델 파일들 로드
        print("📂 모델 파일 로드...")
        
        # 성능 정보 확인
        with open('area_model_performance.json', 'r') as f:
            performance = json.load(f)
        
        print(f"   기록된 성능: R² = {performance['r2_score']:.3f}")
        print(f"   RMSE: {performance['cv_rmse']:.4f}")
        
        # 모델과 스케일러 로드
        model = joblib.load('area_regressor_model_v3_tuned.joblib')
        scaler = joblib.load('area_model_scaler_v3_tuned.joblib') 
        
        # 피처 컬럼 정보 로드
        with open('area_model_columns_v3_tuned.json', 'r') as f:
            feature_columns = json.load(f)
        
        print(f"   모델 타입: {type(model)}")
        print(f"   피처 수: {len(feature_columns)}")
        
        # 데이터 로드
        print("\\n📊 테스트 데이터 준비...")
        df = pd.read_csv('final_merged_feature_engineered.csv', low_memory=False)
        fire_mask = (df['fire_area'] > 0) & df['fire_area'].notna()
        fire_df = df[fire_mask].copy()
        
        print(f"   화재 데이터: {fire_df.shape}")
        print(f"   면적 범위: {fire_df['fire_area'].min():.3f} ~ {fire_df['fire_area'].max():.3f} ha")
        
        # 피처 매칭 확인
        available_features = [col for col in feature_columns if col in fire_df.columns]
        missing_features = [col for col in feature_columns if col not in fire_df.columns]
        
        print(f"\\n🎯 피처 매칭:")
        print(f"   사용 가능: {len(available_features)}/{len(feature_columns)}")
        if missing_features:
            print(f"   누락된 피처: {len(missing_features)}개")
            print(f"   누락 예시: {missing_features[:5]}")
        
        if len(available_features) < len(feature_columns) * 0.8:
            print("   ⚠️ 너무 많은 피처가 누락되었습니다.")
            return
        
        # 데이터 준비
        X = fire_df[available_features].copy()
        y = fire_df['fire_area'].copy()
        
        # 결측치 처리
        for col in X.columns:
            if X[col].isna().sum() > 0:
                X[col] = X[col].fillna(X[col].median())
        
        # 무한값 처리
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        print(f"\\n📊 전처리 완료:")
        print(f"   X shape: {X.shape}")
        print(f"   y 통계: 평균 {y.mean():.2f}, 중앙값 {y.median():.2f}")
        
        # 테스트 분할 (동일한 random_state 사용)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\\n🔄 모델 예측...")
        print(f"   훈련셋: {X_train.shape}")
        print(f"   테스트셋: {X_test.shape}")
        
        # 스케일링
        X_test_scaled = scaler.transform(X_test)
        
        # 예측
        y_pred = model.predict(X_test_scaled)
        
        # 음수 값 처리
        y_pred = np.maximum(0, y_pred)
        
        # 성능 평가
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"\\n🏆 실제 성능 결과:")
        print(f"   R²: {r2:.4f} ({r2:.1%})")
        print(f"   RMSE: {rmse:.3f} ha")
        print(f"   MAE: {mae:.3f} ha")
        print(f"   실제 평균: {y_test.mean():.2f} ha")
        print(f"   예측 평균: {y_pred.mean():.2f} ha")
        
        # 성능 분석
        if r2 > 0.8:
            print("\\n🎉 우수한 성능! 실용적으로 활용 가능합니다.")
        elif r2 > 0.5:
            print("\\n📊 좋은 성능! 참고용으로 활용 가능합니다.")
        elif r2 > 0.2:
            print("\\n⚠️ 제한적 성능. 추가 개선이 필요합니다.")
        else:
            print("\\n❌ 성능 부족. 모델에 문제가 있을 수 있습니다.")
        
        # 예측 vs 실제 분포 비교
        print(f"\\n📈 예측 품질 분석:")
        print(f"   실제값 범위: {y_test.min():.3f} ~ {y_test.max():.3f} ha")
        print(f"   예측값 범위: {y_pred.min():.3f} ~ {y_pred.max():.3f} ha")
        
        # 큰 화재에 대한 예측 성능
        large_fire_mask = y_test > y_test.quantile(0.9)
        if large_fire_mask.sum() > 0:
            large_r2 = r2_score(y_test[large_fire_mask], y_pred[large_fire_mask])
            print(f"   대형 화재(상위 10%) R²: {large_r2:.4f}")
        
        # 작은 화재에 대한 예측 성능  
        small_fire_mask = y_test <= y_test.quantile(0.5)
        if small_fire_mask.sum() > 0:
            small_r2 = r2_score(y_test[small_fire_mask], y_pred[small_fire_mask])
            print(f"   소형 화재(하위 50%) R²: {small_r2:.4f}")
        
        # 피처 중요도 확인 (가능하면)
        if hasattr(model, 'feature_importances_'):
            print(f"\\n🎯 상위 피처 중요도:")
            importances = model.feature_importances_
            feature_imp = list(zip(available_features, importances))
            feature_imp.sort(key=lambda x: x[1], reverse=True)
            
            for i, (feat, imp) in enumerate(feature_imp[:10]):
                print(f"   {i+1:2d}. {feat:25}: {imp:.4f}")
        
        return r2, rmse, mae
        
    except FileNotFoundError as e:
        print(f"❌ 파일을 찾을 수 없습니다: {e}")
        return None, None, None
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    result = test_area_v3_model()
    if result[0] is not None:
        r2, rmse, mae = result
        print(f"\\n✅ 테스트 완료: R² = {r2:.1%}")
    else:
        print(f"\\n❌ 테스트 실패")