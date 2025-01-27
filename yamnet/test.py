import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sounddevice as sd
import librosa
import os
from sklearn.ensemble import IsolationForest
import joblib
from datetime import datetime

class AccidentSoundAnomalyDetector:
    def __init__(self):
        # YAMNet 모델 로드
        self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        
        # Isolation Forest 모델 (이상 감지용)
        self.anomaly_detector = IsolationForest(
            contamination=0.1,  # 이상치 비율 예상값
            random_state=42,
            n_estimators=100
        )
        
        self.YAMNET_SAMPLE_RATE = 16000
        self.threshold = None  # 자동으로 설정될 임계값
        
    def preprocess_audio(self, audio_path):
        """오디오 파일을 YAMNet 임베딩으로 변환"""
        audio_data, sr = librosa.load(audio_path, sr=self.YAMNET_SAMPLE_RATE)
        
        # YAMNet 임베딩 추출
        scores, embeddings, spectrogram = self.yamnet_model(audio_data)
        
        # 임베딩의 평균을 계산하여 단일 벡터로 변환
        embedding_mean = tf.reduce_mean(embeddings, axis=0)
        return embedding_mean.numpy()
    
    def train(self, accident_dir):
        """사고음 데이터만으로 모델 학습"""
        print("사고음 데이터 로딩 중...")
        X = []  # 임베딩 특징
        
        # 사고음 파일 처리
        for filename in os.listdir(accident_dir):
            if filename.endswith(('.wav', '.mp3')):
                audio_path = os.path.join(accident_dir, filename)
                embedding = self.preprocess_audio(audio_path)
                X.append(embedding)
        
        X = np.array(X)
        print(f"총 {len(X)}개의 사고음 데이터로 학습")
        
        # Isolation Forest 학습
        print("이상 감지 모델 학습 중...")
        self.anomaly_detector.fit(X)
        
        # 임계값 설정을 위한 점수 계산
        scores = self.anomaly_detector.score_samples(X)
        self.threshold = np.percentile(scores, 5)  # 하위 5% 기준으로 임계값 설정
        
        # 모델 저장
        model_path = 'accident_anomaly_detector.joblib'
        joblib.dump({'model': self.anomaly_detector, 'threshold': self.threshold}, model_path)
        print(f"모델 저장됨: {model_path}")
    
    def load_model(self, model_path='accident_anomaly_detector.joblib'):
        """저장된 모델 로드"""
        saved_data = joblib.load(model_path)
        self.anomaly_detector = saved_data['model']
        self.threshold = saved_data['threshold']
    
    def predict_realtime(self, audio_data, input_sample_rate):
        """실시간 오디오 데이터에서 사고음 감지"""
        # 오디오 데이터 전처리
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        if input_sample_rate != self.YAMNET_SAMPLE_RATE:
            audio_data = librosa.resample(
                audio_data, 
                orig_sr=input_sample_rate, 
                target_sr=self.YAMNET_SAMPLE_RATE
            )
        
        # YAMNet 임베딩 추출
        scores, embeddings, spectrogram = self.yamnet_model(audio_data)
        embedding_mean = tf.reduce_mean(embeddings, axis=0).numpy()
        
        # 이상 감지 점수 계산
        anomaly_score = self.anomaly_detector.score_samples([embedding_mean])[0]
        
        # 임계값과 비교하여 사고음 여부 판단
        is_accident = anomaly_score < self.threshold
        
        # 신뢰도 점수 계산 (0~1 범위로 정규화)
        confidence = 1 - (anomaly_score - self.anomaly_detector.offset_) / (np.abs(self.threshold - self.anomaly_detector.offset_))
        confidence = np.clip(confidence, 0, 1)
        
        return is_accident, confidence

def start_monitoring(detector):
    """실시간 모니터링 시작"""
    SAMPLE_RATE = 48000
    DURATION = 1
    
    def audio_callback(indata, frames, time, status, detector):
        if status:
            print(status)
        
        audio_data = indata.copy()
        is_accident, confidence = detector.predict_realtime(audio_data, SAMPLE_RATE)
        
        # 현재 시간
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # 상태 출력
        if is_accident:
            print(f"\r🚨 [{current_time}] 사고음 감지! (신뢰도: {confidence:.1%})", end="")
        else:
            print(f"\r✅ [{current_time}] 정상 (신뢰도: {confidence:.1%})", end="")
    
    with sd.InputStream(
        callback=lambda *args: audio_callback(*args, detector),
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=int(SAMPLE_RATE * DURATION)
    ):
        print("사고음 감지 모니터링 시작... (Ctrl+C로 종료)")
        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("\n모니터링을 종료합니다.")

# 사용 예시:
"""
# 감지기 초기화
detector = AccidentSoundAnomalyDetector()

# 학습 (처음 한 번만)
detector.train("./dataset/accident/")

# 또는 저장된 모델 로드
detector.load_model()

# 모니터링 시작
start_monitoring(detector)
"""