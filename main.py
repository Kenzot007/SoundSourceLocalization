import numpy as np
import soundfile as sf
import gammatone.filters as gt_filters
from binaural_features import GetCues_clean
from visualization import *
from auditory_model import lowpass_filter
import scipy.io

import matplotlib
#matplotlib.use('Qt5Agg')
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

    signal, fs = load_audio('/Users/mousei/PycharmProjects/Final Project/SoundSourceLocalization/Data_Gen/output_mixed_snr=0dB.wav')
    filter_type = 'Gammatone'
    cfs = gt_filters.centre_freqs(cutoff=80, fs=fs, num_freqs=32)

    frame_len = int(fs * 0.02)

    cues = GetCues_clean(cfs=cfs,
                         filter_type=filter_type,
                         frame_len=frame_len,
                         frame_shift=int(frame_len * 0.5), # 50% overlap
                         signal=signal, fs=fs,
                         max_delay=1.0,
                         ihc_type='Christof')

    itd_all = cues["itd"]  # (F, N)
    ild_all = cues["ild"]
    ic_all = cues["ic"]
    power_all = cues["power"]

    print("ITD shape:", itd_all.shape)
    print("ILD shape:", ild_all.shape)
    print("IC shape:", ic_all.shape)

    visualize_itd_ild_ic(itd_all, ild_all, ic_all)

if __name__ == "__main__":
    main()
