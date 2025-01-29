import logging
from logging.handlers import RotatingFileHandler
from anomaly_detector import AccidentSoundAnomalyDetector, AudioConfig, ModelConfig, AudioMonitor
import sys
import traceback
import os

# 로깅 설정
handler = RotatingFileHandler("debug.log", maxBytes=10*1024*1024, backupCount=5)
logging.basicConfig(
    handlers=[handler],
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    try:
        print("1. 프로그램 시작")
        
        # 설정 초기화
        print("2. 설정 초기화 중...")
        audio_config = AudioConfig()
        model_config = ModelConfig()
        print("3. 설정 초기화 완료")
        
        # 감지기 초기화
        print("4. 감지기 초기화 중...")
        try:
            detector = AccidentSoundAnomalyDetector(audio_config, model_config)
        except Exception as e:
            print(f"감지기 초기화 실패: {e}")
            logging.error(f"감지기 초기화 실패: {e}")
            sys.exit(1)
        print("5. 감지기 초기화 완료")
        
        # 데이터셋 경로 확인
        dataset_path = "yamnet/accident_detection/dataset/accident"
        print(f"6. 데이터셋 경로 확인: {dataset_path}")
        if not os.path.exists(dataset_path):
            print(f"오류: 데이터셋 경로를 찾을 수 없습니다. 경로: {dataset_path}")
            logging.error(f"데이터셋 경로를 찾을 수 없습니다. 경로: {dataset_path}")
            sys.exit(1)
        print(f"   경로 존재 여부: {os.path.exists(dataset_path)}")
        files = os.listdir(dataset_path)
        print(f"   파일 목록: {files}")
        
        # 모델 학습
        print("7. 모델 학습 시작...")
        detector.train(dataset_path)
        print("8. 모델 학습 완료")
        
        # 모니터링 시작 전 모델 준비 상태 확인
        if not detector.is_model_ready:
            print("오류: 모델이 준비되지 않았습니다. 먼저 학습하거나 로드해주세요.")
            logging.error("모델이 준비되지 않았습니다. 먼저 학습하거나 로드해주세요.")
            sys.exit(1)
        
        # 모니터링 시작
        print("9. 모니터링 시작 중...")
        monitor = AudioMonitor(detector, audio_config)
        print("10. 모니터링 시작")
        try:
            monitor.start()
        except KeyboardInterrupt:
            print("\n모니터링을 종료합니다.")
            logging.info("모니터링을 종료합니다.")
        finally:
            monitor.stop()
        
    except Exception as e:
        print("\n*** 오류 발생 ***")
        print(f"오류 유형: {type(e).__name__}")
        print(f"오류 내용: {str(e)}")
        print("\n상세 오류 정보:")
        traceback.print_exc()
        
        logging.error("상세 오류 정보:")
        logging.error(traceback.format_exc())
        
        sys.exit(1)

if __name__ == "__main__":
    main()