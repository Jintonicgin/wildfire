import os
import subprocess
import sys

def run_training_script(script_name):
    """주어진 파이썬 스크립트를 실행합니다."""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
    if not os.path.exists(script_path):
        print(f"❌ Error: 스크립트 파일을 찾을 수 없습니다: {script_path}")
        return False
    
    print("-" * 70)
    print(f"🚀 '{script_name}' 스크립트 실행을 시작합니다...")
    print("-" * 70)
    
    try:
        # sys.executable은 현재 실행 중인 파이썬 인터프리터를 가리킵니다.
        process = subprocess.run(
            [sys.executable, script_path],
            check=True, # 오류 발생 시 예외를 발생시킴
            text=True,  # 출력을 텍스트로 디코딩
            capture_output=False # 실시간 출력을 위해 False로 설정
        )
        print(f"\n✅ '{script_name}' 스크립트 실행이 성공적으로 완료되었습니다.")
        return True
    except FileNotFoundError:
        print(f"❌ Error: 파이썬 실행 파일을 찾을 수 없습니다. ('{sys.executable}')")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: '{script_name}' 실행 중 오류가 발생했습니다.")
        print(f"Return code: {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return False

def main():
    """모든 모델 학습 스크립트를 순차적으로 실행합니다."""
    print("======= 전체 모델 재학습을 시작합니다 ======")
    
    # 1. 면적 모델 재학습
    if not run_training_script("train_v3_model.py"):
        print("\n면적 모델 학습에 실패하여 전체 프로세스를 중단합니다.")
        return

    # 2. 속도/방향 모델 재학습
    if not run_training_script("train_v2.py"):
        print("\n속도/방향 모델 학습에 실패하여 전체 프로세스를 중단합니다.")
        return
        
    print("\n🎉🎉🎉 모든 모델 재학습이 성공적으로 완료되었습니다! 🎉🎉🎉")

if __name__ == "__main__":
    main()
