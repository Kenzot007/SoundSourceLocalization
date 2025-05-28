import os
import numpy as np
import soundfile as sf
import random
import h5py
import csv
from scipy.signal import fftconvolve, resample_poly

# ---------- 配置路径 ----------
main_audio_dir = r'G:\GitHub\SoundSourceLocalization\Data_Gen\main_audio'
noise_dir = r'G:\GitHub\SoundSourceLocalization\Data_Gen\noise'
sofa_dir = r'G:\GitHub\SoundSourceLocalization\cipic_sofa_all'
output_dir = r'G:\GitHub\SoundSourceLocalization\Dataset'
os.makedirs(output_dir, exist_ok=True)

noise_classes = [d for d in os.listdir(noise_dir) if os.path.isdir(os.path.join(noise_dir, d))]

# ---------- 工具函数 ----------
def resample_to_44100(signal, sr):
    if sr != 44100:
        gcd = np.gcd(sr, 44100)
        signal = resample_poly(signal, 44100 // gcd, sr // gcd)
    return signal

def pad_to_1s(signal, target_len=44100):
    return signal[:target_len] if len(signal) >= target_len else np.concatenate([signal, np.zeros(target_len - len(signal))])

def normalize(signal):
    return signal / (np.max(np.abs(signal)) + 1e-9)

def load_hrtf_from_sofa(sofa_path):
    with h5py.File(sofa_path, 'r') as f:
        hrir = f['Data.IR'][:]          # [N, 2, T]
        positions = f['SourcePosition'][:, 0]  # azimuth only
    return hrir, positions

def interpolate_hrir(hrir_data, az_list, target_az):
    idx = np.argmin(np.abs(az_list - target_az))
    hrir = hrir_data[idx]
    return hrir[0], hrir[1]  # left, right

def binaural_convolve(sig, hrir_l, hrir_r):
    left = fftconvolve(sig, hrir_l, mode='full')[:len(sig)]
    right = fftconvolve(sig, hrir_r, mode='full')[:len(sig)]
    return np.stack([left, right], axis=-1)

def compute_scaling_factor(target, noise, snr_db):
    target_power = np.mean(target ** 2)
    noise_power = np.mean(noise ** 2)
    desired_noise_power = target_power / (10 ** (snr_db / 10))
    return np.sqrt(desired_noise_power / (noise_power + 1e-9))

# ---------- 主循环 ----------
main_files = [f for f in os.listdir(main_audio_dir) if f.endswith('.wav')]
noise_files = [f for f in os.listdir(noise_dir) if f.endswith('.wav')]
sofa_files = [f for f in os.listdir(sofa_dir) if f.endswith('.sofa')]
angles = list(range(0, 360, 5))  # 72个方向

for file in main_files:
    signal_path = os.path.join(main_audio_dir, file)
    signal, sr = sf.read(signal_path)
    signal = resample_to_44100(signal, sr)
    signal = pad_to_1s(signal)
    signal = normalize(signal)

    for az in angles:
        sofa_path = os.path.join(sofa_dir, random.choice(sofa_files))
        hrir_data, az_list = load_hrtf_from_sofa(sofa_path)
        hrir_l, hrir_r = interpolate_hrir(hrir_data, az_list, az)
        sig_bin = binaural_convolve(signal, hrir_l, hrir_r)

        noise_binaural = np.zeros_like(sig_bin)
        selected_class = random.choice(noise_classes)
        class_dir = os.path.join(noise_dir, selected_class)
        class_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
        selected_noises = random.sample(class_files, 2)
        noise_azs = random.sample(angles, 2)
        snrs = []

        for i in range(2):
            n_path = os.path.join(class_dir, selected_noises[i])
            noise, sr_n = sf.read(n_path)
            noise = resample_to_44100(noise, sr_n)
            noise = pad_to_1s(noise)
            noise = normalize(noise)
            hrir_l_n, hrir_r_n = interpolate_hrir(hrir_data, az_list, noise_azs[i])
            noise_bin = binaural_convolve(noise, hrir_l_n, hrir_r_n)

            snr_i = random.uniform(15, 40)
            snrs.append(round(snr_i, 2))
            alpha = compute_scaling_factor(sig_bin, noise_bin, snr_db=snr_i)
            noise_binaural += alpha * noise_bin / np.sqrt(2)

        out = normalize(sig_bin + noise_binaural)
        out_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_azi{az}.wav")
        sf.write(out_path, out, 44100)
        print(f"✅ Saved: {out_path}")

        import csv

        log_path = os.path.join(output_dir, "metadata.csv")
        first_time = not os.path.exists(log_path)

        with open(log_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if first_time:
                writer.writerow([
                    "output_filename", "main_audio", "main_azimuth", "sofa_file",
                    "noise_class", "noise_1", "noise_1_azimuth", "snr_1",
                    "noise_2", "noise_2_azimuth", "snr_2"
                ])

            # 在每次生成 out_path 后添加：
            writer.writerow([
                os.path.basename(out_path),
                file,
                az,
                os.path.basename(sofa_path),
                selected_class,
                selected_noises[0],
                noise_azs[0],
                snrs[0],
                selected_noises[1],
                noise_azs[1],
                snrs[1]
            ])
