import numpy as np
import soundfile as sf
import gammatone.filters as gt_filters
from binaural_features import GetCues_clean
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
        center frequency range: 80Hz to 19200Hz
        num_freqs: Number of filter channels
        order of gammatone filter: 4
    """
    # Load signal and visualize
    signal, fs = load_audio('/Users/mousei/PycharmProjects/Final Project/SoundSourceLocalization/audio/sin_440Hz.wav_[270, 0]_sin_5000Hz.wav_[90, 0].wav')
    left_signal = signal[:, 0]
    right_signal = signal[:, 1]
    t = np.arange(0, len(left_signal)) / fs
    visualize_time(left_signal, right_signal, t, duration=0.05)
    visualize_frequency(left_signal, right_signal, fs)

    filter_type = 'Gammatone'
    cfs = gt_filters.centre_freqs(cutoff=80, fs=fs, num_freqs=32)

    frame_len = int(fs * 0.032)
    spatial_cues, gamma_all, ics_all = GetCues_clean(cfs=cfs,
                                     filter_type=filter_type,
                                     frame_len=frame_len,
                                     tau=10,    # 10ms tau
                                     frame_shift=int(frame_len * 0.5), # 50% overlap
                                     signal=signal, fs=fs,
                                     max_delay=int(fs * 0.001),
                                     ihc_type='Christof')


    # visualize itd and ild
    # choose a certain frequency channel from ccfs.
    freq_channel_index = 20
    itd_data = spatial_cues[freq_channel_index, :, 0]   # ITD
    ild_data = spatial_cues[freq_channel_index, :, 1]   # ILD
    ic_data = ics_all[freq_channel_index, :]
    frames = np.arange(itd_data.shape[0])

    visualize_binary_cues(itd_data, ild_data, frames)
    visualize_spatial_cues(spatial_cues, nonlinearity="log")
    #
    # plt.figure()
    # plt.plot(frames, ic_data)
    # plt.title(f"IC over time (channel {freq_channel_index})")
    # plt.xlabel("Frame index")
    # plt.ylabel("IC (max cross-correlation)")
    # plt.ylim(0.9, 1.03)
    # plt.grid()
    # plt.tight_layout()
    # plt.show()

    plt.figure(figsize=(10, 4))
    plt.imshow(ics_all, aspect="auto", origin="lower", cmap="magma")
    plt.title("IC Cochleagram", fontsize=16)
    plt.xlabel("Frame index", fontsize=14)
    plt.ylabel("Filter # (frequency channel)", fontsize=14)
    plt.colorbar(label="IC value")
    plt.tight_layout()
    plt.show()

    # example: visualize gamma at a frame
    frame_idx = 20
    gamma_curve = gamma_all[freq_channel_index, frame_idx, :]
    lags = np.arange(-int(fs * 0.001), int(fs * 0.001) + 1) * 1000 / fs  # in ms

    plt.figure()
    plt.plot(lags, gamma_curve)
    plt.title(f"Normalized Cross-Correlation γ(n, m) @ frame {frame_idx}")
    plt.xlabel("Delay (ms)")
    plt.ylabel("γ")
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
