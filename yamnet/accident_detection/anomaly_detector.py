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
from dataclasses import dataclass
from typing import Tuple, Optional
import logging
import yaml
import queue
from threading import Thread


@dataclass
class AudioConfig:
    """오디오 설정 클래스"""
    sample_rate: int = 48000
    duration: int = 1
    channels: int = 1
    yamnet_sample_rate: int = 16000

@dataclass
class ModelConfig:
    """모델 설정 클래스"""
    contamination: float = 0.1
    n_estimators: int = 100
    threshold_percentile: int = 5
    model_path: str = 'accident_anomaly_detector.joblib'

class AudioBuffer:
    """스레드 안전한 오디오 버퍼"""
    def __init__(self, maxsize=10):
        self.buffer = queue.Queue(maxsize=maxsize)
    
    def put(self, data):
        try:
            self.buffer.put_nowait(data)
        except queue.Full:
            self.buffer.get()  # 가장 오래된 데이터 제거
            self.buffer.put(data)
    
    def get(self):
        return self.buffer.get() if not self.buffer.empty() else None

class AccidentSoundAnomalyDetector:
    def __init__(self, audio_config: AudioConfig = None, model_config: ModelConfig = None):
        """
        향상된 초기화 함수
        """
        self.audio_config = audio_config or AudioConfig()
        self.model_config = model_config or ModelConfig()
        self.audio_buffer = AudioBuffer()
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        try:
            self.logger.info("YAMNet 모델 로딩 중...")
            self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
            
            self.anomaly_detector = IsolationForest(
                contamination=self.model_config.contamination,
                random_state=42,
                n_estimators=self.model_config.n_estimators
            )
            
            self.threshold = None
            
        except Exception as e:
            self.logger.error(f"모델 초기화 실패: {e}")
            raise

    def preprocess_audio(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """
        오디오 전처리 함수
        """
        try:
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            if sr != self.audio_config.yamnet_sample_rate:
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=sr,
                    target_sr=self.audio_config.yamnet_sample_rate
                )
            
            _, embeddings, _ = self.yamnet_model(audio_data)
            return tf.reduce_mean(embeddings, axis=0).numpy()
            
        except Exception as e:
            self.logger.error(f"오디오 전처리 실패: {e}")
            raise

    @property
    def is_model_ready(self) -> bool:
        """모델이 사용 가능한 상태인지 확인"""
        return self.threshold is not None and self.anomaly_detector is not None

    def train(self, accident_dir: str) -> None:
        """
        향상된 모델 학습 함수
        """
        try:
            self.logger.info("사고음 데이터 로딩 중...")
            X = []
            
            for filename in os.listdir(accident_dir):
                if filename.endswith(('.wav', '.mp3')):
                    audio_path = os.path.join(accident_dir, filename)
                    audio_data, sr = librosa.load(audio_path, sr=self.audio_config.yamnet_sample_rate)
                    embedding = self.preprocess_audio(audio_data, sr)
                    X.append(embedding)
            
            X = np.array(X)
            self.logger.info(f"총 {len(X)}개의 사고음 데이터로 학습")
            
            self.anomaly_detector.fit(X)
            scores = self.anomaly_detector.score_samples(X)
            self.threshold = np.percentile(scores, self.model_config.threshold_percentile)
            
            self._save_model()
            
        except Exception as e:
            self.logger.error(f"모델 학습 실패: {e}")
            raise

    def _save_model(self) -> None:
        """모델 저장"""
        try:
            model_data = {
                'model': self.anomaly_detector,
                'threshold': self.threshold,
                'config': {
                    'audio': self.audio_config.__dict__,
                    'model': self.model_config.__dict__
                }
            }
            joblib.dump(model_data, self.model_config.model_path)
            self.logger.info(f"모델 저장됨: {self.model_config.model_path}")
        except Exception as e:
            self.logger.error(f"모델 저장 실패: {e}")
            raise

    def load_model(self) -> None:
        """모델 로드"""
        try:
            saved_data = joblib.load(self.model_config.model_path)
            self.anomaly_detector = saved_data['model']
            self.threshold = saved_data['threshold']
            
            # 설정 복원
            if 'config' in saved_data:
                self.audio_config = AudioConfig(**saved_data['config']['audio'])
                self.model_config = ModelConfig(**saved_data['config']['model'])
                
            self.logger.info("모델 로드 완료")
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {e}")
            raise

    def predict_realtime(self, audio_data: np.ndarray, input_sample_rate: int) -> Tuple[bool, float, float, np.ndarray, np.ndarray]:
        """
        실시간 예측 함수
        """
        try:
            if not self.is_model_ready:
                raise RuntimeError("모델이 준비되지 않았습니다. 먼저 학습하거나 로드해주세요.")
            
            embedding = self.preprocess_audio(audio_data, input_sample_rate)
            
            # 이상 감지 점수 계산
            anomaly_score = self.anomaly_detector.score_samples([embedding])[0]
            if np.isnan(anomaly_score):
                anomaly_score = 0
            
            is_accident = anomaly_score < self.threshold
            confidence = np.clip((self.threshold - anomaly_score) / np.abs(self.threshold), 0, 1)
            similarity = np.clip(1 - np.abs(anomaly_score / self.threshold), 0, 1)
            
            mel_spectrogram = self.compute_log_mel_spectrogram(audio_data, input_sample_rate)
            
            return is_accident, confidence, similarity, mel_spectrogram, audio_data
            
        except Exception as e:
            self.logger.error(f"실시간 예측 실패: {e}")
            raise

    def compute_log_mel_spectrogram(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """mel 스펙트로그램 계산"""
        try:
            D = librosa.stft(audio_data)
            mel_spectrogram = librosa.feature.melspectrogram(
                S=np.abs(D), 
                sr=sample_rate, 
                n_mels=128
            )
            return librosa.power_to_db(mel_spectrogram)
        except Exception as e:
            self.logger.error(f"스펙트로그램 계산 실패: {e}")
            raise

def plot_spectrogram_and_waveform(mel_spectrogram: np.ndarray, waveform: np.ndarray, sample_rate: int) -> None:
    """시각화 함수"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        ax1.plot(np.linspace(0, len(waveform) / sample_rate, len(waveform)), waveform)
        ax1.set_title("파형 (시간 도메인)")
        ax1.set_xlabel("시간 (초)")
        ax1.set_ylabel("진폭")
        ax1.grid(True)
        
        img = ax2.imshow(
            mel_spectrogram, 
            aspect='auto', 
            origin='lower', 
            cmap='inferno',
            extent=[0, mel_spectrogram.shape[-1] / sample_rate, 0, sample_rate / 2]
        )
        ax2.set_title("로그-멜 스펙트로그램")
        ax2.set_xlabel("시간 (초)")
        ax2.set_ylabel("주파수 (Hz)")
        fig.colorbar(img, ax=ax2, format="%+2.0f dB")
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        logging.error(f"시각화 실패: {e}")
        raise

class AudioMonitor:
    """오디오 모니터링 클래스"""
    def __init__(self, detector: AccidentSoundAnomalyDetector, config: AudioConfig):
        self.detector = detector
        self.config = config
        self.is_running = False
        self.logger = logging.getLogger(__name__)
    
    def audio_callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        """오디오 콜백 함수"""
        if status:
            self.logger.warning(f"오디오 스트림 상태: {status}")
        
        try:
            audio_data = indata.copy()
            self.detector.audio_buffer.put({
                'data': audio_data,
                'timestamp': datetime.now()
            })
            
        except Exception as e:
            self.logger.error(f"오디오 콜백 처리 실패: {e}")
    
    def process_audio(self) -> None:
        """오디오 처리 스레드"""
        while self.is_running:
            try:
                audio_packet = self.detector.audio_buffer.get()
                if audio_packet is None:
                    continue
                
                audio_data = audio_packet['data']
                timestamp = audio_packet['timestamp']
                
                results = self.detector.predict_realtime(
                    audio_data, 
                    self.config.sample_rate
                )
                
                is_accident, confidence, similarity, mel_spectrogram, waveform = results
                
                current_time = timestamp.strftime("%H:%M:%S")
                if is_accident:
                    print(f"\r🚨 [{current_time}] 사고음 감지! (신뢰도: {confidence:.1%}, 유사도: {similarity:.1%})", end="")
                    plot_spectrogram_and_waveform(mel_spectrogram, waveform, self.config.sample_rate)
                else:
                    print(f"\r✅ [{current_time}] 정상 (신뢰도: {confidence:.1%}, 유사도: {similarity:.1%})", end="")
                    
            except Exception as e:
                self.logger.error(f"오디오 처리 실패: {e}")
    
    def start(self) -> None:
        """모니터링 시작"""
        try:
            self.is_running = True
            
            # 오디오 처리 스레드 시작
            process_thread = Thread(target=self.process_audio)
            process_thread.start()
            
            with sd.InputStream(
                callback=self.audio_callback,
                channels=self.config.channels,
                samplerate=self.config.sample_rate,
                blocksize=int(self.config.sample_rate * self.config.duration)
            ):
                print("사고음 감지 모니터링 시작... (Ctrl+C로 종료)")
                try:
                    while self.is_running:
                        sd.sleep(1000)
                except KeyboardInterrupt:
                    self.stop()
                    
        except Exception as e:
            self.logger.error(f"모니터링 시작 실패: {e}")
            self.stop()
            raise
    
