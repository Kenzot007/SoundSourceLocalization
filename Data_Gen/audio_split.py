from pydub import AudioSegment
import os
import math

def split_audio_to_1s_segments(input_file, output_dir='G:\GitHub\SoundSourceLocalization\Data_Gen'):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载音频（pydub会根据后缀识别格式）
    audio = AudioSegment.from_file(input_file)

    duration_ms = len(audio)  # 音频时长（毫秒）
    segment_length = 1000     # 每段 1 秒（1000 毫秒）

    total_segments = math.ceil(duration_ms / segment_length)

    print(f"音频总时长: {duration_ms / 1000:.2f} 秒，共分割为 {total_segments} 段。")

    for i in range(total_segments):
        start = i * segment_length
        end = min((i + 1) * segment_length, duration_ms)
        segment = audio[start:end]

        segment_path = os.path.join(output_dir, f"Noise_{i + 1}.wav")
        segment.export(segment_path, format="wav")
        print(f"保存: {segment_path}")

    print("音频分割完成。")

# 示例调用
if __name__ == "__main__":
    input_audio_path = "G:/GitHub/SoundSourceLocalization/Dataset/ESC-50-master/audio/1-137-A-32.wav"  # 替换为你的音频路径
    split_audio_to_1s_segments(input_audio_path)
