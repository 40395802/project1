import logging
from anomaly_detector import AccidentSoundAnomalyDetector, start_monitoring

# 로깅 설정
logging.basicConfig(filename="debug.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    try:
        logging.info("감지기 초기화 시작")
        # 감지기 초기화
        detector = AccidentSoundAnomalyDetector()

        logging.info("모델 학습 시작")
        # 모델 학습 (처음 한 번만 실행)
        detector.train("yamnet\accident_detection")

        logging.info("모니터링 시작")
        # 모니터링 시작
        start_monitoring(detector)

    except Exception as e:
        logging.error(f"오류 발생: {e}")
        print(f"오류 발생: {e}")
        raise  # 예외를 다시 발생시켜, 상위 코드에서 처리할 수 있도록 함

if __name__ == "__main__":
    main()
