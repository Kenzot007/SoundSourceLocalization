import numpy as np
import soundfile as sf
import h5py
import argparse
from scipy.signal import fftconvolve


def read_sofa(sofa_file):
    with h5py.File(sofa_file, 'r') as f:
        hrtf_data = f['Data.IR'][:]
        sampling_rate = f['Data.SamplingRate'][0]
        source_positions = f['SourcePosition'][:]
    return hrtf_data, sampling_rate, source_positions


def interpolate_hrtf(hrtf_data, source_positions, location):
    distances = np.linalg.norm(source_positions[:, :2] - location[:2], axis=1)
    nearest_index = np.argmin(distances)
    return hrtf_data[nearest_index]


def binauralize(audio_data, fs, location, sofa_file):
    hrtf_data, sofa_fs, source_positions = read_sofa(sofa_file)

    assert fs == sofa_fs, "Sampling rate mismatch."

    hrtf = interpolate_hrtf(hrtf_data, source_positions, np.array(location))
    output_audio = np.array([fftconvolve(audio_data, hrtf_channel) for hrtf_channel in hrtf]).T

    return output_audio


def mix_audio(audio1, audio2):
    """Mix two audios and normalization"""
    min_length = min(len(audio1), len(audio2))
    mixed_audio = audio1[:min_length] + audio2[:min_length]
    mixed_audio /= np.max(np.abs(mixed_audio))
    return mixed_audio


def parse_location(location_str):
    try:
        azimuth, elevation = map(int, location_str.split(","))
        return [azimuth, elevation]
    except ValueError:
        raise argparse.ArgumentTypeError("input format should like '270,0'")


def main():
    parser = argparse.ArgumentParser(description="Mix two audios and use HRTF")

    # 添加命令行参数
    parser.add_argument("--file1", type=str, required=True)
    parser.add_argument("--file2", type=str, required=True)
    parser.add_argument("--location1", type=parse_location, required=True)
    parser.add_argument("--location2", type=parse_location, required=True)
    parser.add_argument("--sofa", type=str, required=True, help="SOFA 文件路径 (HRTF 数据)")

    args = parser.parse_args()

    audio1, fs1 = sf.read(args.file1)
    audio2, fs2 = sf.read(args.file2)

    assert fs1 == fs2, "sampling rate mismatch."

    mixed_audio = mix_audio(audio1, audio2)

    binaural_audio1 = binauralize(audio1, fs1, args.location1, sofa_file=args.sofa)
    binaural_audio2 = binauralize(audio2, fs2, args.location2, sofa_file=args.sofa)

    final_binaural_audio = mix_audio(binaural_audio1, binaural_audio2)

    output_file = f"{args.file1}_{args.location1}_{args.file2}_{args.location2}.wav"
    sf.write(output_file, final_binaural_audio, fs1)

    print(f"Generated {output_file}")


if __name__ == "__main__":
    main()
