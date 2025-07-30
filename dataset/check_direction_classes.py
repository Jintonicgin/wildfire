import joblib
import sys

ENCODER_PATH = '/Users/mmymacymac/Developer/Projects/eclipse/WildFire/dataset/direction_label_encoder.joblib'

try:
    encoder = joblib.load(ENCODER_PATH)
    if hasattr(encoder, 'classes_'):
        print(encoder.classes_)
    else:
        print("인코더에 classes_ 속성이 없습니다.")
except FileNotFoundError:
    print(f"❌ 오류: {ENCODER_PATH} 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"❌ 인코더 로드 중 오류 발생: {e}")
