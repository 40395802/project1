import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import csv
import io
import matplotlib.pyplot as plt

# 모델 불러오기
model = hub.load('https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1')

# 오디오 파일 로딩 함수
def load_audio(file_path, sr=16000):
    waveform, _ = librosa.load(file_path, sr=sr)  # 오디오 파일 로드 및 샘플링
    return waveform

# 샘플 오디오 파일 로드
file_path = "yamnet\\test.wav"  # 실제 파일 경로로 변경
waveform = load_audio(file_path)

# 파형 길이를 16000의 배수로 맞추기 
if len(waveform) % 16000 != 0:
    # 잘라내기
    waveform = np.pad(waveform, (0, 16000 - len(waveform) % 16000), 'constant')

# 모델 실행 및 예측값 받기
scores, embeddings, log_mel_spectrogram = model(waveform)

# 출력 형태 확인
scores.shape.assert_is_compatible_with([None, 521])
embeddings.shape.assert_is_compatible_with([None, 1024])
log_mel_spectrogram.shape.assert_is_compatible_with([None, 64])

# CSV에서 클래스 이름 추출
def class_names_from_csv(class_map_csv_text):
    """점수 벡터에 해당하는 클래스 이름을 반환하는 함수"""
    class_map_csv = io.StringIO(class_map_csv_text)
    class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
    class_names = class_names[1:]  
    return class_names

# 클래스 이름 로드
class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(tf.io.read_file(class_map_path).numpy().decode('utf-8'))

# 평균 점수가 가장 높은 클래스를 예측하여 출력
predicted_class = class_names[scores.numpy().mean(axis=0).argmax()]
print(predicted_class)  # 예측된 클래스 출력

# 시각화
plt.figure(figsize=(10, 4))
plt.plot(waveform)
plt.title('waveform')
plt.xlabel('sample')
plt.ylabel('amplitude')
plt.show()
