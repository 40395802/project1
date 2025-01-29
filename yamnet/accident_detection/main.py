import logging
from anomaly_detector import AccidentSoundAnomalyDetector, AudioConfig, ModelConfig, AudioMonitor
import sys
import traceback

logging.basicConfig(
    filename="debug.log",
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
        detector = AccidentSoundAnomalyDetector(audio_config, model_config)
        print("5. 감지기 초기화 완료")
        
        # 데이터셋 경로 확인
        import os
        dataset_path = "yamnet/accident_detection/dataset/accident"
        print(f"6. 데이터셋 경로 확인: {dataset_path}")
        print(f"   경로 존재 여부: {os.path.exists(dataset_path)}")
        if os.path.exists(dataset_path):
            files = os.listdir(dataset_path)
            print(f"   파일 목록: {files}")
        
        # 모델 학습
        print("7. 모델 학습 시작...")
        detector.train(dataset_path)
        print("8. 모델 학습 완료")
        
        # 모니터링 시작
        print("9. 모니터링 시작 중...")
        monitor = AudioMonitor(detector, audio_config)
        print("10. 모니터링 시작")
        monitor.start()
        
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