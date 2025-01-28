import logging
from anomaly_detector import AccidentSoundAnomalyDetector, start_monitoring

# 디버깅
logging.basicConfig(filename="debug.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    try:
        logging.info("감지기 초기화 시작")
        # 감지기 초기화
        detector = AccidentSoundAnomalyDetector()

        logging.info("모델 학습 시작")
        # 모델 학습
        detector.train("yamnet/accident_detection/dataset/accident")

        logging.info("모니터링 시작")
        # 모니터링 시작
        start_monitoring(detector)
        
    # 디버깅
    except Exception as e:
        logging.error(f"오류 발생: {e}")
        print(f"오류 발생: {e}")
        raise  

if __name__ == "__main__":
    main()
