import tensorflow as tf           # TensorFlow 라이브러리 (머신러닝 및 딥러닝)
import tensorflow_hub as hub      # TensorFlow Hub 라이브러리 (사전 학습된 모델 사용)
import numpy as np                # NumPy 라이브러리 (수치 계산을 위한 배열 처리)
import sounddevice as sd          # SoundDevice 라이브러리 (마이크 입력 오디오 녹음 및 재생)
import librosa                    # Librosa 라이브러리 (오디오 분석 및 처리)
import csv                        # CSV 파일 처리 csv 모듈 
import io                         # 입출력 처리 io 모듈 
import matplotlib.pyplot as plt   # Matplotlib 라이브러리  (데이터 시각화)


# 모델 불러오기
model = hub.load('https://tfhub.dev/google/yamnet/1')

# CSV에서 클래스 이름 추출 함수
def class_names_from_csv(class_map_csv_text):
    """점수 벡터에 해당하는 클래스 이름을 반환하는 함수"""
    class_map_csv = io.StringIO(class_map_csv_text)
    class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
    return class_names[1:]  # CSV 헤더 건너뛰기

# 클래스 이름 로드
class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(tf.io.read_file(class_map_path).numpy().decode('utf-8'))

# 오디오 샘플링 설정
YAMNET_SAMPLE_RATE = 16000  # YAMNet 샘플링 레이트
DURATION = 1  # 분석 주기

# 오디오 데이터 전처리 함수
def preprocess_audio(audio_data, input_sample_rate):
    """입력 데이터를 YAMNet에 맞게 전처리"""
    # 스테레오 입력을 모노로 변환 
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)  # 다중 채널 -> 단일 채널 변환
    
    # 샘플레이트 변환 
    if input_sample_rate != YAMNET_SAMPLE_RATE:
        audio_data = librosa.resample(audio_data, orig_sr=input_sample_rate, target_sr=YAMNET_SAMPLE_RATE)
    
    return np.array(audio_data, dtype=np.float32)

# 오디오 분석 함수
def analyze_audio(audio_data, input_sample_rate):
    """YAMNet 모델로 오디오 데이터를 분석"""
    # 오디오 전처리
    waveform = preprocess_audio(audio_data, input_sample_rate)
    
    # 모델 실행
    scores, embeddings, log_mel_spectrogram = model(waveform)
    
    # 평균 점수를 기반으로 상위 5개 클래스 선택
    mean_scores = scores.numpy().mean(axis=0)
    top_indices = mean_scores.argsort()[-5:][::-1]
    top_classes = [(class_names[i], mean_scores[i]) for i in top_indices]
    
    print("상위 감지된 소리:")
    for name, score in top_classes:
        print(f" - {name}: {score:.3f}")
    
    # 파형 시각화
    plt.figure(figsize=(14, 6))
    
    # 1. 원본 파형
    plt.subplot(3, 1, 1)
    plt.plot(waveform)
    plt.title("Waveform")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    
    # 2. 스펙트로그램 시각화
    plt.subplot(3, 1, 2)
    plt.imshow(log_mel_spectrogram.numpy().T, aspect='auto', origin='lower', 
               extent=[0, len(waveform) / YAMNET_SAMPLE_RATE, 0, log_mel_spectrogram.shape[1]])
    plt.title("Spectrogram (Log-Mel)")
    plt.xlabel("Time (s)")
    plt.ylabel("Mel Frequency")
    plt.colorbar(label="Log Magnitude")
    
    # 3. 상위 클래스 점수 시각화
    plt.subplot(3, 1, 3)
    labels, scores = zip(*top_classes)
    plt.barh(labels, scores, color='skyblue')
    plt.title("Top 5 Detected Classes")
    plt.xlabel("Confidence Score")
    plt.gca().invert_yaxis()  # 상위 클래스가 위에 오도록 순서를 뒤집음
    
    plt.tight_layout()
    plt.show()
    return top_classes

# 마이크 입력 콜백 함수
def audio_callback(indata, frames, time, status):
    """마이크에서 입력된 데이터를 실시간으로 처리"""
    if status:
        print(status)
    audio_data = indata.copy()  # 입력 데이터 복사
    analyze_audio(audio_data, SAMPLE_RATE)

# 스트리밍 시작
SAMPLE_RATE = 48000  # 마이크 기본 샘플레이트 
def start_streaming():
    """실시간 스트리밍 시작"""
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=int(SAMPLE_RATE * DURATION)):
        print("실시간 소리 분석 시작... (Ctrl+C로 종료)")
        try:
            while True:
                pass  # 계속 실행
        except KeyboardInterrupt:
            print("종료합니다.")

# 실행
start_streaming()
