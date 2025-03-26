import numpy as np
import soundfile as sf
import gammatone.filters as gt_filters
from binaural_features import GetCues, GetCues_clean
from visualization import *
from auditory_model import lowpass_filter
import scipy.io
import matplotlib.pyplot as plt

# Load audio file
def load_audio(file_path):
    data, fs = sf.read(file_path)
    print(f"Loaded {file_path} with {len(data)} samples.")
    return data, fs

def main():
    """
    parameters for gammatone filter:
        center frequency range: 80Hz to 5000Hz
        num_freqs: Number of filter channels
        order of gammatone filter: 4
    """
    # Load signal and visualize
    # signal, fs = load_audio('audio/sin/sin_270°.wav')
    # signal, fs = load_audio('audio1/bin_prox_dir/split1_ov1_2.flac')
    signal, fs = load_audio('stereo_5000Hz_left_440Hz_right.wav')
    #signal, fs = load_audio('audio/impulse/impulse_270°.wav')
    #signal, fs = load_audio('audio/sin_440Hz.wav_[270, 0]_sin_5000Hz.wav_[90, 0].wav')
    #signal, fs = load_audio('noisy_market1.wav')
    left_signal = signal[:, 0]
    right_signal = signal[:, 1]
    t = np.arange(0, len(left_signal)) / fs
    visualize_time(left_signal, right_signal, t, duration=0.05)
    visualize_frequency(left_signal, right_signal, fs)

    # filtered_signal = lowpass_filter(signal, fs, cutoff=3000, order=10)
    # filtered_left_signal = filtered_signal[:, 0]
    # filtered_right_signal = filtered_signal[:, 1]
    # visualize_time(filtered_left_signal, filtered_right_signal, t, duration=0.05)
    # visualize_frequency(filtered_left_signal, filtered_right_signal, fs)

    #signal = librosa.resample(signal.T, orig_sr=fs, target_sr=16000).T
    #fs_resampled = fs
    #visual_binary(fs_resampled, signal[:, 0], signal[:, 1])

    # num_freqs = 32
    # min_freq = 80
    # max_freq = 5000
    # cfs = erb_space(min_freq, max_freq, num_freqs)
    filter_type = 'Gammatone'
    cfs = gt_filters.centre_freqs(cutoff=80, fs=fs, num_freqs=32)

    spatial_cues, ccf = GetCues_clean(cfs=cfs,
                                     filter_type=filter_type,
                                     frame_len=int(fs * 0.023),
                                     frame_shift=int(fs * 0.023 * 0.5), # 50% overlap
                                     signal=signal, fs=fs,
                                     max_delay=int(fs * 0.001),
                                     ihc_type='Roman')

    # visualize ccf
    # visualize_ccf(left_signal, right_signal, fs)

    # visualize itd and ild
    # choose a certain frequency channel from ccfs.
    freq_channel_index = 10
    itd_data = spatial_cues[freq_channel_index, :, 0]   # ITD
    ild_data = spatial_cues[freq_channel_index, :, 1]   # ILD
    frames = np.arange(itd_data.shape[0])
    visualize_binary_cues(itd_data, ild_data, frames)
    visualize_spatial_cues(spatial_cues, nonlinearity="log")

    #scipy.io.savemat("spatial_cues.mat", {'spatial_cues': spatial_cues})
    #print("Spatial cues saved")

if __name__ == "__main__":
    main()
