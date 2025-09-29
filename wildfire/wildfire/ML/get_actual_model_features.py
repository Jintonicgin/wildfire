#!/usr/bin/env python3
"""
실제 모델들이 기대하는 피처 확인
"""

import joblib
import json

def get_actual_model_features():
    """모델들이 실제로 기대하는 피처들 확인"""
    print("🔍 실제 모델 피처 확인")
    print("=" * 40)
    
    # Speed/Direction 모델의 실제 피처들
    speed_data = joblib.load("improved_speed_model_v2.joblib")
    direction_data = joblib.load("improved_direction_model_v2.joblib")
    
    speed_features = speed_data.get('features', [])
    direction_features = direction_data.get('features', [])
    
    print(f"Speed 모델 피처 ({len(speed_features)}개):")
    for i, feat in enumerate(speed_features):
        print(f"  {i+1:2d}. {feat}")
    
    print(f"\nDirection 모델 피처 ({len(direction_features)}개):")
    for i, feat in enumerate(direction_features):
        print(f"  {i+1:2d}. {feat}")
    
    # Advanced area model의 피처들
    area_data = joblib.load("advanced_area_model_final.joblib")
    area_features = area_data.get('features', [])
    
    print(f"\nAdvanced Area 모델 피처 ({len(area_features)}개):")
    for i, feat in enumerate(area_features):
        print(f"  {i+1:2d}. {feat}")
    
    # 각 모델의 피처를 JSON으로 저장
    with open('actual_speed_features.json', 'w') as f:
        json.dump(speed_features, f, indent=2)
    
    with open('actual_direction_features.json', 'w') as f:
        json.dump(direction_features, f, indent=2)
        
    with open('actual_area_features.json', 'w') as f:
        json.dump(area_features, f, indent=2)
    
    print(f"\n💾 실제 피처 목록들이 JSON 파일로 저장됨:")
    print(f"   - actual_speed_features.json")
    print(f"   - actual_direction_features.json") 
    print(f"   - actual_area_features.json")
    
    return speed_features, direction_features, area_features

def check_fallback_models():
    """Fallback 모델들의 피처 확인"""
    print(f"\n🔄 Fallback 모델들 확인")
    print("-" * 30)
    
    fallback_files = [
        ("area_regressor_model_v4.joblib", "area_model_columns_v4.json"),
        ("speed_classifier_model_v4.joblib", "speed_model_columns_v4.json"),
        ("direction_classifier_model_v4.joblib", "direction_model_columns_v4.json")
    ]
    
    for model_file, columns_file in fallback_files:
        try:
            with open(columns_file, 'r') as f:
                features = json.load(f)
            print(f"{columns_file}: {len(features)} features")
            print(f"   샘플: {features[:3]}...")
        except:
            print(f"{columns_file}: 파일 없음")

if __name__ == "__main__":
    speed_features, direction_features, area_features = get_actual_model_features()
    check_fallback_models()
    
    print(f"\n📋 요약:")
    print(f"   Speed: {len(speed_features)} features")
    print(f"   Direction: {len(direction_features)} features") 
    print(f"   Area: {len(area_features)} features")
    
    print(f"\n💡 다음 단계: predict_from_feature.py를 실제 피처 목록에 맞게 수정해야 합니다.")