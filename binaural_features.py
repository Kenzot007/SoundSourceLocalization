import numpy as np
from scipy.signal import correlate
from auditory_model import Audiotory_peripheral, Haircell_model

def Efficient_ccf(left_signal, right_signal):
    """calculate cross-correlation function in frequency domain, which is more efficient than the direct calculation"""
    if left_signal.shape[0] != right_signal.shape[0]:
        raise Exception('length mismatch')
    signal_len = left_signal.shape[0]
    window = np.hanning(signal_len)
    left_signal = left_signal * window
    right_signal = right_signal * window

    left_fft = np.fft.fft(left_signal, 2 * signal_len - 1) # zero-padding
    right_fft = np.fft.fft(right_signal, 2 * signal_len - 1)
    ccf_unshift = np.real(np.fft.ifft(left_fft * np.conjugate(right_fft)))
    ccf = np.fft.fftshift(ccf_unshift)

    return ccf

def calculate_itd(signal, fs, max_delay=None, inter_method='exponential', alpha=0.9):
    """
        estimate ITD based on interaural corss-correlation function
        itd = chann0_delay - chann1_delay
        x_detrend[:, 0]: left ear signal, x_trend[:, 1]: right ear signal
        corr(i) = sum(x0[t]*x1[t-i])
            | >0 chann1 lead
        itd |
            | <0 chann0 lead

        Args:
            signal: Input binaural audio signal, shape = (num_samples, 2)
            max_delay: maximum value of ITD, which equals sampling rate * maximum delay(s), default value: 1ms
            inter_method: method of ccf interpolation, "None"(default),"parabolic","exponential".
            alpha: smoothness factor
        """
    signal_len = signal.shape[0]
    signal_detrend = signal - np.mean(signal, axis=0)   # x_detrend = x - np.mean(x, axis=0)

    if max_delay is None:
        max_delay = int(1e-3 * fs)  # 1ms delay (typical human ITD max)

    x1 = signal[:, 0]
    x2 = signal[:, 1]
    N = len(x1)
    lags = np.arange(-max_delay, max_delay + 1)
    gamma = np.zeros(len(lags))

    a11 = np.zeros(len(lags))
    a12 = np.zeros(len(lags))
    a22 = np.zeros(len(lags))

    for i, m in enumerate(lags):
        for n in range(N):
            i1 = n-m if m < 0 else n
            i2 = n+m if m > 0 else n

            if i1 < 0 or i1 >= N or i2 < 0 or i2 >= N:
                continue

            x1n = x1[i1]
            x2n = x2[i2]

            a11[i] = alpha * (x1n ** 2) + (1 - alpha) * a11[i]
            a12[i] = alpha * (x1n * x2n) + (1 - alpha) * a12[i]
            a22[i] = alpha * (x2n ** 2) + (1 - alpha) * a22[i]

        gamma[i] = a12[i] / (np.sqrt(a11[i] * a22[i]) + 1e-12)  # Y(n,m)

    max_idx = np.argmax(gamma)
    IC = gamma[max_idx]
    ITD = lags[max_idx] / fs * 1e3

    # if False:
    #     # frequency domain
    #     ccf_full = Efficient_ccf(signal_detrend[:, 0], signal_detrend[:, 1])
    #     ccf = ccf_full[signal_len-1-max_delay : signal_len+max_delay]
    # else:
    #     # time domain
    #     ccf_full = np.correlate(signal_detrend[:, 0], signal_detrend[:, 1], mode='full')
    #     ccf = ccf_full[signal_len - 1 - max_delay:signal_len + max_delay]
    #
    # energy_left = np.sum(signal_detrend[:, 0] ** 2)
    # energy_right = np.sum(signal_detrend[:, 1] ** 2)
    # ccf_std = ccf / np.sqrt(energy_left * energy_right) + 1e-10
    # max_pos = np.argmax(ccf)  # Optimal delay in signal alignment between the left and right ears
    #
    # # exponential interpolation
    # delta = 0
    # if inter_method == 'exponential':
    #     if max_pos > 0 and max_pos < max_delay * 2 - 2:
    #         if np.min(ccf[max_pos - 1:max_pos + 2]) > 0:
    #             delta = (np.log10(ccf[max_pos + 1]) - np.log10(ccf[max_pos - 1])) / \
    #                     (4 * np.log10(ccf[max_pos]) -
    #                     2 * np.log10(ccf[max_pos - 1]) -
    #                     2 * np.log10(ccf[max_pos + 1]))
    # elif inter_method == 'parabolic':
    #     if max_pos > 0 and max_pos < max_delay * 2 - 2:
    #         delta = (ccf[max_pos - 1] - ccf[max_pos + 1]) / (
    #                     2 * (ccf[max_pos + 1] - 2 * ccf[max_pos] + ccf[max_pos - 1]))
    #
    # ITD = float((max_pos - max_delay - 1 + delta)) / fs * 1e3
    # IC = np.max(ccf_std)
    # return [ITD, ccf_std, IC]  # ITD(ms)
    return ITD, IC, gamma

def calculate_ild(signal):
    """ ILD
        To be consistent with ITD:
        ild = 10log10(chann0_energy/chann1_energy)
            |>0 chann0 lead(left ear)
        ild |
            |<0 chann1 lead(right ear)
    """

    rms_left = np.sqrt(np.mean(np.power(signal[:, 0], 2))) + np.finfo(float).eps
    rms_right = np.sqrt(np.mean(np.power(signal[:, 1], 2))) + np.finfo(float).eps

    # calculate ILD
    ild = 20 * np.log10(rms_left / rms_right)

    return ild

def GetCues_clean(signal, fs, frame_len, filter_type, cfs, frame_shift=None, max_delay=None, ihc_type=None):
    """calculate binaural localization cues, [itds,ilds,ccfs]

    Args:
        signal: target signal
        fs: sample frequency
        frame_len:
        filter_type: "Gammatone" / "Butterworth"
        cfs: center frequencies of filters
        frame_shift: default frame_len/2
        max_daly: the maximum delay(sample), default frame_len-1
        ihc_type: different model used in previous work
            Lindemann: half-wave rectify + low-pass filter(1th Butterworth, cutoff-frequency 800Hz)
            Breebart: half-wave rectify + low-pass filter(5th, cutoff-frequency 770Hz)
            More to added....
    Returns:
        [[itds,ilds],ccfs]
        spatial_cues: shape = (freq_chann_num, frame_num, 2) [ITD, ILD]
        ccf_std_all: shape = (freq_chann_num, frame_num, max_delay*2 + 1)
    """

    signal_len = signal.shape[0]
    rms_all = np.sqrt(np.mean(signal ** 2))

    if frame_len > 2*signal_len:
        frame_len = signal_len // 2
        frame_shift = int(frame_len / 2)
        max_delay = frame_len - 1
        print('frame_len too big, automatically shrunk its value')

    # filtered signal
    #freq_chann_num = cfs.shape[0]
    freq_chann_num = len(cfs)
    _, signal_env = Audiotory_peripheral(signal, fs, cfs, filter_type, ihc_type)

    #frame_num = int((signal_len - frame_len - frame_len) / frame_shift + 1)
    frame_num = (signal_len - frame_len) // frame_shift

    # initialize matrix for ITD and ILD
    spatial_cues = np.zeros((freq_chann_num, frame_num, 2), dtype=np.float32)  # [itd_frame,ild_frame]
    #ccf_std_all = np.zeros((freq_chann_num, frame_num, max_delay * 2 + 1), dtype=np.float32)
    gamma_all = np.zeros((freq_chann_num, frame_num, max_delay*2 + 1), dtype=np.float32)
    ics_all = np.zeros((freq_chann_num, frame_num), dtype=np.float32)

    alpha = 0.9

    frame_rms_list = []
    for frame_i in range(frame_num):
        frame_start_pos = frame_i * frame_shift + frame_len
        frame_end_pos = frame_start_pos + frame_len
        for freq_chann_i in range(freq_chann_num):
            tar_chann_frame = signal_env[freq_chann_i, frame_start_pos:frame_end_pos, :]
            # calculate left and right energy
            left_energy = np.sqrt(np.mean(tar_chann_frame[:, 0] ** 2))
            right_energy = np.sqrt(np.mean(tar_chann_frame[:, 1] ** 2))
            energy_avg = (left_energy + right_energy) / 2
            frame_rms_list.append(energy_avg)
    energy_threshold = np.percentile(frame_rms_list, 5)

    for frame_i in range(frame_num):
        frame_start_pos = frame_i * frame_shift + frame_len
        frame_end_pos = frame_start_pos + frame_len

        for freq_chann_i in range(freq_chann_num):
            tar_chann_frame = signal_env[freq_chann_i, frame_start_pos:frame_end_pos, :]

            left_energy = np.sqrt(np.mean(tar_chann_frame[:, 0] ** 2))
            right_energy = np.sqrt(np.mean(tar_chann_frame[:, 1] ** 2))
            energy_avg = (left_energy + right_energy) / 2
            if energy_avg < energy_threshold:
                continue    # skip this frame

            # calculate ITD and ILD
            itd_frame, ic_frame, gamma = calculate_itd(tar_chann_frame, fs, max_delay=max_delay)
            ild_frame = calculate_ild(tar_chann_frame)

            if ild_frame == np.inf:
                print(freq_chann_i, frame_i)
                raise Exception('invalid ild')

            # EMA
            if frame_i > 0:
                spatial_cues[freq_chann_i, frame_i, 0] = alpha * itd_frame + (1-alpha) * spatial_cues[freq_chann_i, frame_i - 1, 0]
                spatial_cues[freq_chann_i, frame_i, 1] = alpha * ild_frame + (1-alpha) * spatial_cues[freq_chann_i, frame_i - 1, 1]
            else:
                spatial_cues[freq_chann_i, frame_i, 0] = itd_frame
                spatial_cues[freq_chann_i, frame_i, 1] = ild_frame

            # cross-correlation function
            ics_all[freq_chann_i, frame_i] = ic_frame
            gamma_all[freq_chann_i, frame_i, :] = gamma
    return [spatial_cues, gamma_all, ics_all]



def GetCues(tar, interfer, fs, frame_len, filter_type, cfs, frame_shift=None, max_delay=None, ihc_type=None):
    """calculate binaural localization cues, [itds,ilds,ccfs]

    Args:
        tar: target signal
        fs: sample frequency
        frame_len:
        filter_type: "Gammatone" / "Butterworth"
        cfs: center frequencies of filters
        frame_shift: default frame_len/2
        max_daly: the maximum delay(sample), default frame_len-1
        ihc_type: different model used in previous work
            Lindemann: half-wave rectify + low-pass filter(1th Butterworth, cutoff-frequency 800Hz)
            Breebart: half-wave rectify + low-pass filter(5th, cutoff-frequency 770Hz)
            More to added....
    Returns:
        [[itds,ilds],ccfs]
        spatial_cues: shape = (freq_chann_num, frame_num, 2) [ITD, ILD]
        ccf_std_all: shape = (freq_chann_num, frame_num, max_delay*2 + 1)
    """

    wav_len = tar.shape[0]
    if wav_len != interfer.shape[0]:
        raise Exception('length do match')

    if frame_shift == None:  # default overlap: frame_len/2
        frame_shift = int(frame_len / 2)

    freq_chann_num = cfs.shape[0]

    tar_audiogram, _ = Audiotory_peripheral(tar, fs, cfs, filter_type, ihc_type)
    interfer_audiogram = Audiotory_peripheral(interfer, fs, cfs, filter_type, ihc_type)

    mix_audiogram = tar_audiogram + interfer_audiogram

    if ihc_type == None:
        mix_env = mix_audiogram
    else:
        print('ihc model: %s' % ihc_type)
        mix_env = Haircell_model(mix_audiogram, fs, ihc_type)

    frame_num = int((wav_len - frame_len - frame_len) / frame_shift + 1)
    if max_delay == None:
        max_delay = frame_len - 1

    spatial_cues = np.zeros((freq_chann_num, frame_num, 2), dtype=np.float32)  # [itd_frame,ild_frame]
    ccf_std_all = np.zeros((freq_chann_num, frame_num, max_delay * 2 + 1), dtype=np.float32)
    SNR_all = np.zeros((freq_chann_num, frame_num), dtype=np.float32)

    for freq_chann_i in range(freq_chann_num):
        itds_chann = []
        ilds_chann = []
        ccfs_chann = []
        # print(cfs[freq_chann_i])
        for frame_i in range(frame_num):
            frame_start_pos = frame_i * frame_shift + frame_len
            frame_end_pos = frame_start_pos + frame_len

            mix_env_chann_fram = mix_env[freq_chann_i, frame_start_pos:frame_end_pos]

            tar_chann_frame = tar_audiogram[freq_chann_i, frame_start_pos:frame_end_pos]
            interfer_chann_frame = interfer_audiogram[freq_chann_i, frame_start_pos:frame_end_pos]
            SNR_chan_frame = 10 * np.log10(np.sum(tar_chann_frame ** 2) / np.sum(interfer_chann_frame ** 2))

            # bianural cues
            itd_frame, ccf_std_frame = calculate_itd(mix_env_chann_fram, fs, max_delay=max_delay)
            ild_frame = calculate_ild(mix_env_chann_fram)
            # check whether ILD is valid
            if ild_frame == np.inf:
                print(freq_chann_i, frame_i)
                raise Exception('invalid ild')

            spatial_cues[freq_chann_i, frame_i, 0] = itd_frame
            spatial_cues[freq_chann_i, frame_i, 1] = ild_frame
            ccf_std_all[freq_chann_i, frame_i, :] = ccf_std_frame
            SNR_all[freq_chann_i, frame_i] = SNR_chan_frame

    return [spatial_cues, ccf_std_all, SNR_all]