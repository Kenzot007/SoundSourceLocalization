import numpy as np
import scipy.signal as sn
from visualization import *
from apgf import APGF
import gammatone.filters as gt_filters

def get_erb(cf):
    return 24.7 * (4.37 * cf / 1000 + 1.0)

def lowpass_filter(signal, fs, cutoff=3000, order=10):
    """
    Apply a Butterworth low-pass filter to a binaural signal.

    Args:
        signal: Input binaural signal (num_samples, 2)
        fs: Sampling rate (Hz)
        cutoff: Cutoff frequency for low-pass filtering (Hz)
        order: Filter order

    Returns:
        Filtered binaural signal (num_samples, 2)
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalized cutoff frequency
    sos = sn.butter(order, normal_cutoff, btype='low', output='sos')

    # Ensure input is stereo
    if signal.ndim == 1:
        raise ValueError("Expected a stereo (2-channel) signal but got mono.")

    # Apply filter to both left and right channels
    filtered_signal = np.zeros_like(signal)
    for i in range(signal.shape[1]):
        filtered_signal[:, i] = sn.sosfilt(sos, signal[:, i])

    return filtered_signal

def erb_space(low_freq, high_freq, num_freqs):
    """
    Compute ERB-scaled center frequencies
    """
    ear_q = 9.26449  # Equivalent Rectangular Bandwidth (ERB) parameters
    min_bw = 24.7
    erb_points = np.linspace(
        low_freq / (ear_q + min_bw),
        high_freq / (ear_q + min_bw),
        num_freqs
    )
    return (ear_q + min_bw) * erb_points

def Butter_spectrogram(signal, fs, cfs):
    # signal_std = signal
    signal_length = signal.shape[0]

    filter_num = cfs.shape[0]
    audiogram = np.zeros((filter_num, signal_length, 2))
    cutoff = np.zeros(2)
    for filter_index in range(filter_num):
        erb = get_erb(cfs[filter_index])
        cutoff[0] = cfs[filter_index] - erb/2
        cutoff[1] = cfs[filter_index] + erb/2
        sos = sn.butter(Wn=cutoff/fs*2, btype='bandpass', N=6, output='sos')
        audiogram[filter_index, :, :] = sn.sosfiltfilt(x=signal, sos=sos, axis=0)

    return audiogram

def GF_spectrogram(signal, fs, cfs):
    coefs = gt_filters.make_erb_filters(fs=fs, centre_freqs=cfs)

    # Visualize Gammatone frequency response
    # plot_gammatone_frequency_response(fs=fs, cfs=cfs, coefs=coefs)
    left_audiogram = gt_filters.erb_filterbank(signal[:, 0], coefs)
    right_audiogram = gt_filters.erb_filterbank(signal[:, 1], coefs)

    # Visualize Gammatone frequency response(Cochleagram)
    # plot_left_audiogram_spectrogram(left_audiogram, fs)
    # plot_right_audiogram_spectrogram(right_audiogram, fs)
    audiogram = np.concatenate((left_audiogram[:, :, np.newaxis],
                                       right_audiogram[:, :, np.newaxis]), axis=2)
    return audiogram


# APFG Filter
def APGF_spectrogram(signal, fs, cfs):
    num_channels = len(cfs)
    apgf = APGF(sample_rate=fs, low_frequency=cfs[0], high_frequency=cfs[-1], channel_count=num_channels)

    left_audiogram = apgf.process(signal[:, 0])
    right_audiogram = apgf.process(signal[:, 1])

    lr_audiogram = np.stack((left_audiogram, right_audiogram), axis=2)
    return lr_audiogram

def Haircell_model(signal, fs, model_type='Lindemann'):
    if model_type == 'Lindemann':
        signal_rectified = np.maximum(signal, 0)  # Half-wave rectification
        b, a = sn.butter(1, 800.0 / (fs / 2.0))  # low-pass filter with 800Hz cutoff frequency
        if signal_rectified.shape[0] <= max(6, len(b) * 3): # check signal length
            print(f"Skipping short signal ({signal_rectified.shape[0]} samples).")
            return np.zeros_like(signal_rectified)
        envelope = sn.filtfilt(b, a, signal_rectified, axis=1)

    elif model_type == 'Breebaart':
        signal_rectified = np.maximum(signal, 0)
        b, a = sn.butter(5, 2000.0 / (fs / 2.0))
        envelope = sn.filtfilt(b, a, signal_rectified, axis=1)

    elif model_type == 'Roman':
        signal_rectified = np.maximum(signal, 0)
        envelope = np.sqrt(signal_rectified)

    elif model_type == 'rectify':
        envelope = np.maximum(signal, 0)

    elif model_type == 'Christof':
        signal_compressed = np.power(np.abs(signal), 0.23)
        signal_rectified = np.maximum(signal_compressed, 0)
        sos = sn.butter(4, 425/(fs/2.0), btype='low', output='sos')
        envelope = sn.sosfiltfilt(sos, signal_rectified, axis=1)

    else:
        raise ValueError("Unsupported haircell model type.")

    return envelope

def Audiotory_peripheral(signal, fs, cfs, filter_type, model_type=None):
    if len(cfs.shape) == 0:
        cfs = np.reshape(cfs,newshape=[1])
    freq_chann_num = cfs.shape[0]

    if filter_type == 'Gammatone':
        audiogram = GF_spectrogram(signal, fs, cfs)
    elif filter_type == 'Butterworth':
        audiogram = Butter_spectrogram(signal, fs, cfs)
    elif filter_type == 'APGF':
        audiogram = APGF_spectrogram(signal, fs, cfs)
    else:
        raise ValueError("Unsupported filter type.")

    if audiogram.shape[0] != freq_chann_num:
        raise Exception('wrong shape of audiogram')

    if model_type == None:
        env = audiogram
    else:
        #env = Haircell_model(audiogram, fs, model_type)
        env = audiogram

    return [audiogram, env]