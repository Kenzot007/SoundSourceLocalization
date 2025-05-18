# Complete pipeline: unify sampling rate, pad to 1s, spatialize signal, add spatialized noise

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import fftconvolve
import h5py
import os
import random

# ---------------------- Signal utilities ----------------------
def resample_to_44100(signal, sr):
    if sr != 44100:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=44100)
    return signal

def pad_to_1s(signal, target_len=44100):
    if len(signal) >= target_len:
        return signal[:target_len]
    else:
        pad_len = target_len - len(signal)
        return np.concatenate([signal, np.zeros(pad_len)])

def normalize(sig):
    return sig / (np.max(np.abs(sig)) + 1e-9)

# ---------------------- HRTF loading and interpolation ----------------------
def load_hrtf_from_sofa(sofa_path):
    with h5py.File(sofa_path, 'r') as f:
        hrir = f['Data.IR'][:]            # shape: [N, 2, T]
        positions = f['SourcePosition'][:, 0]  # only azimuth (in degrees)
    return hrir, positions

def interpolate_hrir(hrir_data, az_list, target_az):
    idx = np.argmin(np.abs(az_list - target_az))
    hrir = hrir_data[idx]
    return hrir[0], hrir[1]  # left, right

# ---------------------- SNR control ----------------------
def compute_scaling_factor(target, noise, snr_db):
    target_power = np.mean(target ** 2)
    noise_power = np.mean(noise ** 2)
    desired_noise_power = target_power / (10 ** (snr_db / 10))
    return np.sqrt(desired_noise_power / (noise_power + 1e-9))

# ---------------------- Binaural convolution ----------------------
def binaural_hrtf_convolve(signal, hrir_l, hrir_r):
    left = fftconvolve(signal, hrir_l, mode='full')[:len(signal)]
    right = fftconvolve(signal, hrir_r, mode='full')[:len(signal)]
    return np.stack([left, right], axis=-1)

# ---------------------- Main spatialization + mixing ----------------------
def spatial_mix(signal_path, bg_paths, sofa_file, target_azimuth, bg_azimuths=None, snr_db=10):
    # 1. Load and preprocess main signal
    signal, sr = sf.read(signal_path)
    signal = resample_to_44100(signal, sr)
    signal = pad_to_1s(signal)
    signal = normalize(signal)

    # 2. Load HRIR
    hrir_data, az_list = load_hrtf_from_sofa(sofa_file)
    hrir_L, hrir_R = interpolate_hrir(hrir_data, az_list, target_azimuth)

    # 3. Spatialize main signal
    sig_binaural = binaural_hrtf_convolve(signal, hrir_L, hrir_R)

    # 4. Background noise
    bg_binaural_total = np.zeros_like(sig_binaural)
    N = len(bg_paths)
    for i, bg_path in enumerate(bg_paths):
        bg, bg_sr = sf.read(bg_path)
        bg = resample_to_44100(bg, bg_sr)
        bg = pad_to_1s(bg)
        bg = normalize(bg)

        az = bg_azimuths[i] if bg_azimuths else random.uniform(0, 360)
        hrir_L_bg, hrir_R_bg = interpolate_hrir(hrir_data, az_list, az)
        bg_binaural = binaural_hrtf_convolve(bg, hrir_L_bg, hrir_R_bg)

        alpha = compute_scaling_factor(sig_binaural, bg_binaural, snr_db)
        bg_binaural_total += (alpha / np.sqrt(N)) * bg_binaural

    # 5. Mix and normalize
    out = sig_binaural + bg_binaural_total
    out = normalize(out)
    return out, 44100

# ---------------------- Example usage ----------------------
if __name__ == '__main__':
    signal_path = 'Data_Gen/1.wav'
    bg_paths = ['G:\GitHub\SoundSourceLocalization\Data_Gen\office\Office_1.wav', 'G:\GitHub\SoundSourceLocalization\Data_Gen\office\Office_2.wav']
    sofa_file = 'hrtf/subject_003.sofa'
    target_azimuth = 90
    bg_azimuths = [30, 210]  # or None for random

    output, sr = spatial_mix(signal_path, bg_paths, sofa_file, target_azimuth, bg_azimuths, snr_db=10)
    sf.write('output_mixed.wav', output, sr)
