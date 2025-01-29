import unittest
import numpy as np
# import librosa
import soundfile as sf
import os
from anomaly_detector import AccidentSoundAnomalyDetector, AudioConfig, ModelConfig

class TestAccidentSoundDetector(unittest.TestCase):
    def setUp(self):
        self.audio_config = AudioConfig()
        self.model_config = ModelConfig()
        self.detector = AccidentSoundAnomalyDetector(self.audio_config, self.model_config)
        
        # 테스트용 데이터 경로
        self.test_accident_dir = "yamnet/accident_detection/test_sample/accident"
        self.test_normal_dir = "yamnet/accident_detection/test_sample/normal"
        
        # 테스트 데이터 생성
        self._create_test_data()
        
    def _create_test_data(self):
        """테스트용 오디오 데이터 생성"""
        os.makedirs(self.test_accident_dir, exist_ok=True)
        os.makedirs(self.test_normal_dir, exist_ok=True)
        
        # 테스트용 사고음 생성 (급격한 변화가 있는 소리)
        duration = 1.0
        sr = self.audio_config.sample_rate
        t = np.linspace(0, duration, int(sr * duration))
        
        # 사고음 시뮬레이션 (급격한 주파수 변화)
        accident_sound = np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 10)
        sf.write(
            os.path.join(self.test_accident_dir, "test_accident1.wav"),
            accident_sound,
            sr
)
        
        # 일반 소리 시뮬레이션 (일정한 주파수)
        normal_sound = np.sin(2 * np.pi * 440 * t)
        sf.write(
            os.path.join(self.test_normal_dir, "test_normal1.wav"),
            normal_sound,
            sr
)

    def test_model_training(self):
        """모델 학습 테스트"""
        print("\n테스트 1: 모델 학습")
        try:
            self.detector.train(self.test_accident_dir)
            self.assertTrue(self.detector.is_model_ready)
            print("✅ 모델 학습 성공")
        except Exception as e:
            self.fail(f"모델 학습 실패: {str(e)}")

    def test_accident_detection(self):
        """사고음 감지 테스트"""
        print("\n테스트 2: 사고음 감지")
        self.detector.train(self.test_accident_dir)
        
        # 사고음 테스트
        audio_path = os.path.join(self.test_accident_dir, "test_accident1.wav")
        audio_data, sr = sf.load(audio_path, sr=self.audio_config.sample_rate)
        is_accident, confidence, _, _, _ = self.detector.predict_realtime(audio_data, sr)
        
        print(f"사고음 감지 결과: {'감지됨' if is_accident else '감지안됨'} (신뢰도: {confidence:.2%})")
        self.assertTrue(is_accident)
        
        # 일반 소리 테스트
        audio_path = os.path.join(self.test_normal_dir, "test_normal1.wav")
        audio_data, sr = sf.load(audio_path, sr=self.audio_config.sample_rate)
        is_accident, confidence, _, _, _ = self.detector.predict_realtime(audio_data, sr)
        
        print(f"일반음 감지 결과: {'감지됨' if is_accident else '감지안됨'} (신뢰도: {confidence:.2%})")
        self.assertFalse(is_accident)

    def test_real_audio_files(self):
        """실제 오디오 파일 테스트"""
        print("\n테스트 3: 실제 오디오 파일")
        if os.path.exists("yamnet/accident_detection/dataset/accident"):
            self.detector.train("yamnet/accident_detection/dataset/accident")
            
            # 데이터셋의 첫 번째 파일로 테스트
            accident_files = os.listdir("yamnet/accident_detection/dataset/accident")
            if accident_files:
                test_file = os.path.join("yamnet/accident_detection/dataset/accident", accident_files[0])
                audio_data, sr = sf.load(test_file, sr=self.audio_config.sample_rate)
                is_accident, confidence, _, _, _ = self.detector.predict_realtime(audio_data, sr)
                print(f"실제 사고음 파일 테스트 결과: {'감지됨' if is_accident else '감지안됨'} (신뢰도: {confidence:.2%})")
        else:
            print("⚠️ 실제 데이터셋을 찾을 수 없습니다")

    def tearDown(self):
        """테스트 데이터 정리"""
        import shutil
        if os.path.exists(self.test_accident_dir):
            shutil.rmtree(self.test_accident_dir)
        if os.path.exists(self.test_normal_dir):
            shutil.rmtree(self.test_normal_dir)

if __name__ == '__main__':
    unittest.main(verbosity=2)
    
    
    #실행이 안됌 수정 필요