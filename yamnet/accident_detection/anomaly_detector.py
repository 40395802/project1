import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sounddevice as sd
import librosa
import os
from sklearn.ensemble import IsolationForest
import joblib
from datetime import datetime
import matplotlib.pyplot as plt


class AccidentSoundAnomalyDetector:
    def __init__(self):
        # YAMNet 모델 로드
        self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        
        # Isolation Forest 모델 (사고 감지용)
        self.anomaly_detector = IsolationForest(
            contamination=0.1,  
            random_state=42,
            n_estimators=100
        )
        
        self.YAMNET_SAMPLE_RATE = 16000
        self.threshold = None  # 임계값 (학습 시 자동 설정)
        
    def preprocess_audio(self, audio_path):
        """오디오 파일을 YAMNet 임베딩으로 변환"""
        audio_data, sr = librosa.load(audio_path, sr=self.YAMNET_SAMPLE_RATE)
        
        # YAMNet 임베딩 추출
        _, embeddings, _ = self.yamnet_model(audio_data)
        
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
        _, embeddings, _ = self.yamnet_model(audio_data)
        embedding_mean = tf.reduce_mean(embeddings, axis=0).numpy()
        
        # 이상 감지 점수 계산
        anomaly_score = self.anomaly_detector.score_samples([embedding_mean])[0]
        
        # anomaly_score 기본값 설정
        if np.isnan(anomaly_score):
            anomaly_score = 0  # 기본값 설정

        # 임계값과 비교하여 사고음 여부 판단
        is_accident = anomaly_score < self.threshold
        
        # 신뢰도 계산 
        if self.threshold is not None:
            confidence = (self.threshold - anomaly_score) / np.abs(self.threshold)
        else:
            confidence = 0
        
        confidence = np.clip(confidence, 0, 1)
        
        # 유사도 계산 (Isolation Forest에서 제공하는 score를 활용)
        similarity = 1 - np.abs(anomaly_score / self.threshold)
        similarity = np.clip(similarity, 0, 1)
        
        # log-mel spectrogram, waveform 
        mel_spectrogram = self.compute_log_mel_spectrogram(audio_data, self.YAMNET_SAMPLE_RATE)
        waveform = audio_data  
        
        return is_accident, confidence, similarity, mel_spectrogram, waveform

    def compute_log_mel_spectrogram(self, audio_data, sample_rate):
        """log-mel spectrogram """
       
        D = librosa.stft(audio_data)
        
        mel_spectrogram = librosa.feature.melspectrogram(S=np.abs(D), sr=sample_rate, n_mels=128)
        
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        
        return log_mel_spectrogram


def plot_spectrogram_and_waveform(mel_spectrogram, waveform, sample_rate):
    """시각화하는 함수"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # waveform
    ax1.plot(np.linspace(0, len(waveform) / sample_rate, len(waveform)), waveform)
    ax1.set_title("Waveform (Time Domain)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)
    
    # mel spectrogram
    img = ax2.imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='inferno', extent=[0, mel_spectrogram.shape[-1] / sample_rate, 0, sample_rate / 2])
    ax2.set_title("Log-Mel Spectrogram")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")
    fig.colorbar(img, ax=ax2, format="%+2.0f dB")  
    ax2.grid(True)
    
    plt.tight_layout()  
    plt.show()


def start_monitoring(detector):
    """실시간 모니터링 시작"""
    SAMPLE_RATE = 48000
    DURATION = 1
    
    def audio_callback(indata, frames, time, status, detector):
        if status:
            print(status)
        
        audio_data = indata.copy()
        is_accident, confidence, similarity, mel_spectrogram, waveform = detector.predict_realtime(audio_data, SAMPLE_RATE)
        
        # 현재 시간
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # 상태 출력
        if is_accident:
            print(f"\r🚨 [{current_time}] 사고음 감지! (신뢰도: {confidence:.1%}, 유사도: {similarity:.1%})", end="")
            
            # 사고음 감지 시 시각화 출력
            plot_spectrogram_and_waveform(mel_spectrogram, waveform, SAMPLE_RATE)
        else:
            print(f"\r✅ [{current_time}] 정상 (신뢰도: {confidence:.1%}, 유사도: {similarity:.1%})", end="")
    
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
