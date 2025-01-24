import numpy as np
import soundfile as sf
from scipy.io import wavfile

def convert_to_mono(data):
    if len(data.shape) == 2:
        data = np.mean(data, axis=1)
    return data

def convert_sample_rate(input_file, output_file, target_sample_rate):
    # 오디오 파일 읽기
    data, sample_rate = sf.read(input_file)
    
    # 모노로 변환
    data = convert_to_mono(data)
    
    # 샘플레이트 변환
    if sample_rate != target_sample_rate:
        number_of_samples = round(len(data) * float(target_sample_rate) / sample_rate)
        data = np.interp(np.linspace(0, len(data), number_of_samples), np.arange(len(data)), data)
    
    # 변환된 오디오 파일 저장
    sf.write(output_file, data, target_sample_rate)
    print(f"Converted {input_file} to mono and from {sample_rate}Hz to {target_sample_rate}Hz, saved to {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert audio file to mono and change sample rate.")
    parser.add_argument("input_file", type=str, help="Path to the input audio file.")
    parser.add_argument("output_file", type=str, help="Path to save the converted audio file.")
    parser.add_argument("target_sample_rate", type=int, help="Target sample rate for the converted audio file.")

    args = parser.parse_args()

    convert_sample_rate(args.input_file, args.output_file, args.target_sample_rate)