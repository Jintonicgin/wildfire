import sys
import json
import joblib
import numpy as np
import pandas as pd
import warnings
import os
import traceback
import glob
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

# 로컬 모듈 임포트
try:
    from wildfire.ML.DB_data.oracle_db import OracleDB
    from wildfire.ML.fetch_all_weather import fetch_all_weather_features
    from wildfire.ML.fwi_calc import fwi_calc
except ImportError as e:
    print(f"[초기화 오류] 필수 모듈 임포트에 실패했습니다: {e}")
    from wildfire.ML.DB_data.oracle_db import OracleDB
    from wildfire.ML.fetch_all_weather import fetch_all_weather_features
    from wildfire.ML.fwi_calc import fwi_calc

warnings.filterwarnings("ignore")

class ModernizedPredictionSystem:
    """현대화된 화재 예측 시스템"""
    
    def __init__(self, model_base_path: str = None):
        self.model_base_path = model_base_path or os.path.dirname(os.path.abspath(__file__))
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = {}
        self.column_mappings = {}
        
        # 모델 타입별 설정
        self.model_configs = {
            'area': {
                'target_type': 'regression',
                'output_transform': lambda x: np.expm1(x)  # 로그 변환 되돌리기
            },
            'speed': {
                'target_type': 'classification',
                'categories': ['slow', 'medium', 'fast']
            },
            'direction': {
                'target_type': 'classification', 
                'categories': ['north', 'east', 'south', 'west']
            }
        }
        
        # 초기화
        self.load_latest_models()
        self.initialize_column_mappings()
    
    def find_latest_model_files(self) -> Dict[str, Dict[str, str]]:
        """최신 모델 파일들을 자동으로 찾습니다."""
        model_files = {}
        
        for model_type in ['area', 'speed', 'direction']:
            model_files[model_type] = {}
            
            # 모델 파일 찾기 (improved > clean > v3_tuned 순으로 우선순위)
            patterns = [
                f"{model_type}_model_improved.joblib",
                f"{model_type}_model_clean*.joblib", 
                f"{model_type}_*_model_v3_tuned.joblib",
                f"{model_type}_regressor_model_v3_tuned.joblib" if model_type == 'area' else f"{model_type}_classifier_model_v2_tuned_cw.joblib"
            ]
            
            for pattern in patterns:
                files = glob.glob(os.path.join(self.model_base_path, pattern))
                if files:
                    model_files[model_type]['model'] = max(files, key=os.path.getmtime)
                    break
            
            # 스케일러 파일 찾기
            scaler_patterns = [
                f"{model_type}_scaler_improved.joblib",
                f"{model_type}_scaler_clean*.joblib",
                f"{model_type}_*_scaler_v3_tuned.joblib",
                f"{model_type}_model_scaler_v3_tuned.joblib" if model_type == 'area' else f"{model_type}_scaler_v2_tuned_cw.joblib"
            ]
            
            for pattern in scaler_patterns:
                files = glob.glob(os.path.join(self.model_base_path, pattern))
                if files:
                    model_files[model_type]['scaler'] = max(files, key=os.path.getmtime)
                    break
            
            # 인코더 파일 찾기 (분류 모델만)
            if model_type in ['speed', 'direction']:
                encoder_patterns = [
                    f"{model_type}_encoder_improved.joblib",
                    f"{model_type}_encoder_clean*.joblib"
                ]
                
                for pattern in encoder_patterns:
                    files = glob.glob(os.path.join(self.model_base_path, pattern))
                    if files:
                        model_files[model_type]['encoder'] = max(files, key=os.path.getmtime)
                        break
            
            # 컬럼 파일 찾기
            column_patterns = [
                f"{model_type}_columns_improved*.json",
                f"{model_type}_columns_clean*.json",
                f"{model_type}_model_columns_v3_tuned.json" if model_type == 'area' else f"{model_type}_model_columns_v2_tuned_cw.json"
            ]
            
            for pattern in column_patterns:
                files = glob.glob(os.path.join(self.model_base_path, pattern))
                if files:
                    model_files[model_type]['columns'] = max(files, key=os.path.getmtime)
                    break
        
        return model_files
    
    def load_latest_models(self):
        """최신 모델들을 자동으로 로드합니다."""
        print("🔍 최신 모델 파일들을 검색 중...")
        
        model_files = self.find_latest_model_files()
        
        for model_type, files in model_files.items():
            try:
                print(f"\\n📁 {model_type.upper()} 모델 로딩:")
                
                # 모델 로드
                if 'model' in files:
                    self.models[model_type] = joblib.load(files['model'])
                    print(f"  ✅ 모델: {os.path.basename(files['model'])}")
                else:
                    print(f"  ❌ 모델 파일을 찾을 수 없습니다.")
                    continue
                
                # 스케일러 로드
                if 'scaler' in files:
                    self.scalers[model_type] = joblib.load(files['scaler'])
                    print(f"  ✅ 스케일러: {os.path.basename(files['scaler'])}")
                
                # 인코더 로드 (분류 모델만)
                if model_type in ['speed', 'direction'] and 'encoder' in files:
                    self.encoders[model_type] = joblib.load(files['encoder'])
                    print(f"  ✅ 인코더: {os.path.basename(files['encoder'])}")
                
                # 컬럼 정보 로드
                if 'columns' in files:
                    with open(files['columns'], 'r') as f:
                        self.feature_columns[model_type] = json.load(f)
                    print(f"  ✅ 피처 컬럼 ({len(self.feature_columns[model_type])}개): {os.path.basename(files['columns'])}")
                
            except Exception as e:
                print(f"  ❌ {model_type} 모델 로딩 실패: {e}")
        
        loaded_models = list(self.models.keys())
        print(f"\\n🎉 로딩 완료된 모델: {loaded_models}")
        
        if not loaded_models:
            raise RuntimeError("로딩된 모델이 없습니다!")
    
    def initialize_column_mappings(self):
        """동적 컬럼 매핑 초기화"""
        # 기본 매핑 규칙들
        base_mappings = {
            # 시간 관련
            'startmonth': 'fire_month',
            'startday': 'startday', 
            'startyear': 'startyear',
            
            # 계절 정보
            'is_spring': 'is_spring',
            'is_summer': 'is_summer',
            'is_autumn': 'is_autumn', 
            'is_winter': 'is_winter',
            
            # 지형 정보
            'elevation_mean': 'elevation_mean',
            'elevation_std': 'elevation_std',
            'elevation_min': 'elevation_min',
            'elevation_max': 'elevation_max',
            'slope_mean': 'slope_mean',
            'slope_std': 'slope_std',
            'slope_min': 'slope_min',
            'slope_max': 'slope_max',
            'aspect_mode': 'aspect_mode',
            'aspect_std': 'aspect_std',
            'aspect_north_ratio': 'aspect_north_ratio',
            'aspect_south_ratio': 'aspect_south_ratio',
            
            # 식생 정보
            'ndvi_before': 'ndvi_before',
            'treecover_pre_fire_5x5': 'treecover_pre_fire_5x5',
            
            # 위치 정보
            'start_latitude': 'start_latitude',
            'start_longitude': 'start_longitude'
        }
        
        # 각 모델별 동적 매핑 생성
        for model_type, columns in self.feature_columns.items():
            self.column_mappings[model_type] = base_mappings.copy()
            
            # 모델별 특정 컬럼들 추가
            for col in columns:
                if col not in self.column_mappings[model_type]:
                    # 자동 매핑 로직
                    if '_0h' in col or '_past' in col:
                        # 기상 데이터 매핑
                        self.column_mappings[model_type][col] = col
                    elif 'fwi' in col.lower() or 'ffmc' in col.lower() or 'dmc' in col.lower():
                        # FWI 관련 매핑
                        self.column_mappings[model_type][col] = col
                    elif any(keyword in col for keyword in ['dry_days', 'total_precip', 'consecutive']):
                        # 강수량 관련 매핑
                        self.column_mappings[model_type][col] = col
                    else:
                        # 기본 매핑
                        self.column_mappings[model_type][col] = col
    
    def collect_features_safely(self, lat: float, lon: float, region_name: str = "") -> Dict[str, Any]:
        """안전한 피처 수집 (에러 처리 강화)"""
        try:
            print(f"🌍 지역 데이터 수집: {region_name} ({lat:.4f}, {lon:.4f})")
            
            # 1. 기상 데이터 수집 시도
            try:
                timestamp = datetime.now()
                weather_features = fetch_all_weather_features(lat, lon, timestamp)
                
                if weather_features.get('success', False):
                    print("✅ 기상 데이터 수집 성공")
                else:
                    print("⚠️ 기상 데이터 수집 부분 실패 - 기본값 사용")
                    weather_features = self.get_default_weather_features(lat, lon)
                    
            except Exception as e:
                print(f"❌ 기상 데이터 수집 실패: {e}")
                weather_features = self.get_default_weather_features(lat, lon)
            
            # 2. 지형/식생 데이터 DB에서 수집 시도
            try:
                db = OracleDB()
                db_features = db.get_features_by_region(region_name)
                if db_features:
                    print("✅ DB 지형 데이터 수집 성공")
                    weather_features.update(db_features)
                else:
                    print("⚠️ DB 데이터 없음 - 기본 지형값 사용")
                    default_terrain = self.get_default_terrain_features(lat, lon)
                    weather_features.update(default_terrain)
                    
            except Exception as e:
                print(f"❌ DB 데이터 수집 실패: {e}")
                default_terrain = self.get_default_terrain_features(lat, lon)
                weather_features.update(default_terrain)
            
            # 3. 추가 계산 피처들
            self.add_computed_features(weather_features)
            
            return weather_features
            
        except Exception as e:
            print(f"❌ 전체 피처 수집 실패: {e}")
            return self.get_emergency_features(lat, lon)
    
    def get_default_weather_features(self, lat: float, lon: float) -> Dict[str, Any]:
        """기본 기상 피처 (API 실패시)"""
        current_time = datetime.now()
        month = current_time.month
        
        # 계절별 기본값
        seasonal_defaults = {
            'spring': {'temp': 15.0, 'humidity': 55.0, 'wind': 4.0},
            'summer': {'temp': 25.0, 'humidity': 65.0, 'wind': 3.0},
            'autumn': {'temp': 12.0, 'humidity': 50.0, 'wind': 5.0},
            'winter': {'temp': 2.0, 'humidity': 70.0, 'wind': 4.0}
        }
        
        if month in [3, 4, 5]:
            season_data = seasonal_defaults['spring']
        elif month in [6, 7, 8]:
            season_data = seasonal_defaults['summer'] 
        elif month in [9, 10, 11]:
            season_data = seasonal_defaults['autumn']
        else:
            season_data = seasonal_defaults['winter']
        
        return {
            # 기본 기상값
            't2m_0h': season_data['temp'],
            'rh2m_0h': season_data['humidity'],
            'ws10m_0h': season_data['wind'],
            'wd10m_0h': 180.0,
            'ps_0h': 1013.25,
            'prectotcorr_0h': 0.0,
            'allsky_sfc_sw_dwn_0h': 200.0,
            
            # 시간 정보
            'fire_month': month,
            'startday': current_time.day,
            'startmonth': month,
            'startyear': current_time.year,
            
            # 계절 정보
            'is_spring': 1 if month in [3, 4, 5] else 0,
            'is_summer': 1 if month in [6, 7, 8] else 0,
            'is_autumn': 1 if month in [9, 10, 11] else 0,
            'is_winter': 1 if month in [12, 1, 2] else 0,
            
            # 위치 정보
            'start_latitude': lat,
            'start_longitude': lon,
            
            # 기본 FWI 값
            'ffmc_0h': 85.0,
            'dmc_0h': 30.0,
            'dc_0h': 200.0,
            'isi_0h': 5.0,
            'bui_0h': 35.0,
            'fwi_0h': 15.0,
            
            # 건조 관련
            'dry_days_7d_start': 3,
            'dry_days_30d_start': 10,
            'consecutive_dry_days_start': 2,
            
            'success': False  # API 실패 표시
        }
    
    def get_default_terrain_features(self, lat: float, lon: float) -> Dict[str, Any]:
        """기본 지형 피처"""
        # 강원도 평균적인 지형 특성
        return {
            'elevation_mean': 600.0,
            'elevation_std': 200.0,
            'elevation_min': 200.0,
            'elevation_max': 1000.0,
            'slope_mean': 20.0,
            'slope_std': 8.0,
            'slope_min': 0.0,
            'slope_max': 45.0,
            'aspect_mode': 180.0,  # 남향
            'aspect_std': 90.0,
            'aspect_north_ratio': 0.25,
            'aspect_south_ratio': 0.30,
            'ndvi_before': 0.65,
            'treecover_pre_fire_5x5': 75.0
        }
    
    def get_emergency_features(self, lat: float, lon: float) -> Dict[str, Any]:
        """비상시 최소 피처 (모든 수집 실패시)"""
        weather = self.get_default_weather_features(lat, lon)
        terrain = self.get_default_terrain_features(lat, lon)
        weather.update(terrain)
        weather['emergency_mode'] = True
        return weather
    
    def add_computed_features(self, features: Dict[str, Any]):
        """계산된 피처들 추가"""
        try:
            # 온도와 습도가 있으면 열 스트레스 지수 계산
            if 't2m_0h' in features and 'rh2m_0h' in features:
                temp = features['t2m_0h']
                humidity = features['rh2m_0h']
                features['heat_stress_index'] = temp * (100 - humidity) / 100
            
            # 바람과 건조도 조합
            if 'ws10m_0h' in features and 'rh2m_0h' in features:
                wind = features['ws10m_0h']
                humidity = features['rh2m_0h']
                features['wind_dryness'] = wind * (100 - humidity) / 100
            
            # 지형 위험도
            if all(k in features for k in ['slope_mean', 'aspect_south_ratio']):
                features['terrain_fire_risk'] = features['slope_mean'] * features['aspect_south_ratio']
            
            # 월별 순환 특성
            if 'fire_month' in features:
                month = features['fire_month']
                features['month_sin'] = np.sin(2 * np.pi * month / 12)
                features['month_cos'] = np.cos(2 * np.pi * month / 12)
                
        except Exception as e:
            print(f"⚠️ 계산 피처 생성 중 오류: {e}")
    
    def prepare_model_features(self, features: Dict[str, Any], model_type: str) -> Optional[np.ndarray]:
        """모델별 피처 배열 준비"""
        try:
            if model_type not in self.feature_columns:
                print(f"❌ {model_type} 모델의 컬럼 정보가 없습니다.")
                return None
            
            required_columns = self.feature_columns[model_type]
            column_mapping = self.column_mappings.get(model_type, {})
            
            feature_values = []
            missing_features = []
            
            for col in required_columns:
                # 매핑된 컬럼명 찾기
                mapped_col = column_mapping.get(col, col)
                
                if mapped_col in features:
                    value = features[mapped_col]
                elif col in features:
                    value = features[col]
                else:
                    value = 0.0  # 기본값
                    missing_features.append(col)
                
                # 타입 확인 및 변환
                try:
                    value = float(value) if value is not None else 0.0
                except (ValueError, TypeError):
                    value = 0.0
                
                feature_values.append(value)
            
            if missing_features:
                print(f"⚠️ {model_type} 모델에서 누락된 피처 {len(missing_features)}개: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
            
            # numpy 배열로 변환
            X = np.array(feature_values).reshape(1, -1)
            
            # 스케일링 적용
            if model_type in self.scalers:
                X_scaled = self.scalers[model_type].transform(X)
                return X_scaled
            else:
                print(f"⚠️ {model_type} 모델의 스케일러가 없습니다.")
                return X
                
        except Exception as e:
            print(f"❌ {model_type} 모델 피처 준비 실패: {e}")
            return None
    
    def predict_single_model(self, features: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        """단일 모델 예측"""
        try:
            if model_type not in self.models:
                return {'error': f'{model_type} 모델이 로드되지 않았습니다.'}
            
            # 피처 준비
            X = self.prepare_model_features(features, model_type)
            if X is None:
                return {'error': f'{model_type} 모델 피처 준비 실패'}
            
            # 예측 수행
            model = self.models[model_type]
            prediction = model.predict(X)[0]
            
            # 결과 처리
            result = {'raw_prediction': float(prediction)}
            
            if self.model_configs[model_type]['target_type'] == 'regression':
                # 회귀 모델 (피해면적)
                if 'output_transform' in self.model_configs[model_type]:
                    transformed_value = self.model_configs[model_type]['output_transform'](prediction)
                    result['predicted_value'] = float(transformed_value)
                else:
                    result['predicted_value'] = float(prediction)
                result['unit'] = 'hectares' if model_type == 'area' else 'unknown'
                
            else:
                # 분류 모델 (속도, 방향)
                if model_type in self.encoders:
                    try:
                        encoded_prediction = int(prediction)
                        decoded_prediction = self.encoders[model_type].inverse_transform([encoded_prediction])[0]
                        result['predicted_category'] = decoded_prediction
                        result['confidence'] = 'medium'  # 실제로는 predict_proba 사용해야 함
                    except:
                        result['predicted_category'] = prediction
                        result['confidence'] = 'low'
                else:
                    result['predicted_category'] = prediction
                    result['confidence'] = 'low'
            
            result['success'] = True
            return result
            
        except Exception as e:
            return {'error': f'{model_type} 예측 실패: {str(e)}', 'success': False}
    
    def predict_all(self, lat: float, lon: float, region_name: str = "") -> Dict[str, Any]:
        """전체 예측 수행"""
        try:
            print(f"\\n🔥 화재 위험도 예측 시작: {region_name}")
            
            # 1. 피처 수집
            features = self.collect_features_safely(lat, lon, region_name)
            
            # 2. 각 모델별 예측
            predictions = {}
            
            for model_type in ['area', 'speed', 'direction']:
                if model_type in self.models:
                    print(f"\\n🎯 {model_type.upper()} 예측 중...")
                    pred_result = self.predict_single_model(features, model_type)
                    predictions[model_type] = pred_result
                    
                    if pred_result.get('success', False):
                        if model_type == 'area':
                            print(f"  ✅ 예상 피해면적: {pred_result.get('predicted_value', 0):.2f} hectares")
                        else:
                            print(f"  ✅ 예측 결과: {pred_result.get('predicted_category', 'unknown')}")
                    else:
                        print(f"  ❌ 예측 실패: {pred_result.get('error', 'unknown')}")
            
            # 3. 종합 위험도 계산
            overall_risk = self.calculate_overall_risk(predictions, features)
            
            # 4. 최종 결과 구성
            result = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'location': {
                    'latitude': lat,
                    'longitude': lon,
                    'region': region_name
                },
                'predictions': predictions,
                'overall_risk': overall_risk,
                'data_sources': {
                    'weather_api': features.get('success', False),
                    'database': not features.get('emergency_mode', False),
                    'emergency_mode': features.get('emergency_mode', False)
                }
            }
            
            print(f"\\n🏆 종합 위험도: {overall_risk.get('level', 'unknown')} ({overall_risk.get('score', 0):.1f}%)")
            
            return result
            
        except Exception as e:
            error_msg = f"전체 예측 시스템 오류: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            }
    
    def calculate_overall_risk(self, predictions: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
        """종합 위험도 계산"""
        try:
            risk_score = 0
            max_score = 100
            
            # 피해면적 기반 점수 (40점)
            if 'area' in predictions and predictions['area'].get('success'):
                area = predictions['area'].get('predicted_value', 0)
                if area > 100:
                    risk_score += 40
                elif area > 10:
                    risk_score += 30
                elif area > 1:
                    risk_score += 20
                else:
                    risk_score += 10
            
            # 속도 기반 점수 (30점)
            if 'speed' in predictions and predictions['speed'].get('success'):
                speed = predictions['speed'].get('predicted_category', 'slow')
                if speed == 'fast':
                    risk_score += 30
                elif speed == 'medium':
                    risk_score += 20
                else:
                    risk_score += 10
            
            # 기상 조건 기반 점수 (30점)
            weather_score = 0
            temp = features.get('t2m_0h', 15)
            humidity = features.get('rh2m_0h', 60)
            wind = features.get('ws10m_0h', 3)
            
            # 온도 위험도
            if temp > 30:
                weather_score += 10
            elif temp > 25:
                weather_score += 7
            elif temp > 20:
                weather_score += 5
            
            # 습도 위험도 (낮을수록 위험)
            if humidity < 30:
                weather_score += 10
            elif humidity < 50:
                weather_score += 7
            elif humidity < 70:
                weather_score += 5
            
            # 바람 위험도
            if wind > 10:
                weather_score += 10
            elif wind > 5:
                weather_score += 5
            
            risk_score += weather_score
            
            # 위험도 등급
            risk_percentage = min(100, risk_score)
            
            if risk_percentage >= 80:
                level = "매우 높음"
                color = "#FF0000"
            elif risk_percentage >= 60:
                level = "높음"
                color = "#FF6600"
            elif risk_percentage >= 40:
                level = "보통"
                color = "#FFCC00"
            elif risk_percentage >= 20:
                level = "낮음"
                color = "#66CC00"
            else:
                level = "매우 낮음"
                color = "#00CC00"
            
            return {
                'level': level,
                'score': risk_percentage,
                'color': color,
                'recommendation': self.get_recommendation(level),
                'components': {
                    'area_score': min(40, risk_score),
                    'speed_score': weather_score,
                    'weather_score': weather_score
                }
            }
            
        except Exception as e:
            return {
                'level': '계산 오류',
                'score': 0,
                'color': '#CCCCCC',
                'error': str(e)
            }
    
    def get_recommendation(self, risk_level: str) -> str:
        """위험도별 권고사항"""
        recommendations = {
            "매우 높음": "🚨 즉시 대피 준비, 화재 발생시 신속 대응 필요",
            "높음": "⚠️ 화재 예방 조치 강화, 감시 체계 가동",
            "보통": "📋 일반적인 화재 예방 수칙 준수", 
            "낮음": "✅ 정상적인 활동 가능, 기본 주의사항 유지",
            "매우 낮음": "🟢 화재 위험 거의 없음"
        }
        return recommendations.get(risk_level, "권고사항 없음")

# 전역 인스턴스
_prediction_system = None

def get_prediction_system() -> ModernizedPredictionSystem:
    """싱글톤 예측 시스템 인스턴스 반환"""
    global _prediction_system
    if _prediction_system is None:
        _prediction_system = ModernizedPredictionSystem()
    return _prediction_system

def predict_fire_risk(lat: float, lon: float, region_name: str = "") -> Dict[str, Any]:
    """편의 함수: 화재 위험도 예측"""
    system = get_prediction_system()
    return system.predict_all(lat, lon, region_name)

if __name__ == "__main__":
    # 테스트 실행
    print("🔥 현대화된 화재 예측 시스템 테스트")
    
    # 강릉시 테스트
    result = predict_fire_risk(37.7519, 128.8761, "강릉시")
    
    if result['success']:
        print("\\n✅ 예측 성공!")
        print(f"종합 위험도: {result['overall_risk']['level']}")
        for pred_type, pred_data in result['predictions'].items():
            if pred_data.get('success'):
                if pred_type == 'area':
                    print(f"{pred_type}: {pred_data.get('predicted_value', 0):.2f} hectares")
                else:
                    print(f"{pred_type}: {pred_data.get('predicted_category', 'unknown')}")
    else:
        print(f"❌ 예측 실패: {result.get('error', 'unknown')}")