import numpy as np
from scipy.signal import correlate
import soundfile as sf
from auditory_model import Audiotory_peripheral, Haircell_model

def Efficient_ccf(left_signal, right_signal):
    """calculate cross-correlation function in frequency domain, which is more efficient than the direct calculation"""
    if left_signal.shape[0] != right_signal.shape[0]:
        raise Exception('length mismatch')
    wav_len = left_signal.shape[0]
    wf = np.hanning(wav_len)
    left_signal *= wf
    right_signal *= wf

    left_signal = np.fft.fft(left_signal, 2 * wav_len - 1)# zero-padding
    right_signal = np.fft.fft(right_signal, 2 * wav_len - 1)
    ccf_unshift = np.real(np.fft.ifft(np.multiply(left_signal, np.conjugate(right_signal))))
    ccf = np.concatenate([ccf_unshift[wav_len:], ccf_unshift[:wav_len]], axis=0)

    return ccf

def calculate_itd(signal, fs, max_delay=None, inter_method='parabolic'):
    """
        estimate ITD based on interaural corss-correlation function
        itd = chann0_delay - chann1_delay
        x_detrend[:, 0]: left ear signal, x_trend[:, 1]: right ear signal
        corr(i) = sum(x0[t]*x1[t-i])
            | >0 chann1 lead
        itd |
            | <0 chann1 lead
        input:
            max_delay: maximum value of ITD, which equals sampling rate * maximum delay(s), default value: 1ms
             inter_method: method of ccf interpolation, "None"(default),"parabolic","exponential".
        """
    signal_len = signal.shape[0]
    signal_detrend = signal   # x_detrend = x - np.mean(x, axis=0)

    if max_delay == None:
        max_delay = int(1e-3*fs)    # default delay=1ms
    if False:
        # time domain
        ccf_full = np.correlate(signal_detrend[:, 0], signal_detrend[:, 1], 'full')
        ccf = ccf_full[wav_len-1-max_delay : wav_len+max_delay]
    else:
        # frequency domain
        ccf_full = Efficient_ccf(signal_detrend[:, 0], signal_detrend[:, 1])
        ccf = ccf_full[signal_len-1-max_delay : signal_len+max_delay]

        ccf_std = ccf / (np.sqrt(np.sum(signal_detrend[:, 0] ** 2) * np.sum(signal_detrend[:, 1] ** 2)))
        max_pos = np.argmax(ccf)  # Optimal delay in signal alignment between the left and right ears

        # exponential interpolation
        delta = 0
        if inter_method == 'exponential':
            if max_pos > 0 and max_pos < max_delay * 2 - 2:
                if np.min(ccf[max_pos - 1:max_pos + 2]) > 0:
                    delta = (np.log(ccf[max_pos + 1]) - np.log(ccf[max_pos - 1])) / \
                            (4 * np.log(ccf[max_pos]) -
                             2 * np.log(ccf[max_pos - 1]) -
                             2 * np.log(ccf[max_pos + 1]))
        elif inter_method == 'parabolic':
            if max_pos > 0 and max_pos < max_delay * 2 - 2:
                delta = (ccf[max_pos - 1] - ccf[max_pos + 1]) / (
                            2 * (ccf[max_pos + 1] - 2 * ccf[max_pos] + ccf[max_pos - 1]))

        ITD = float((max_pos - max_delay - 1 + delta)) / fs * 1e3
        return [ITD, ccf_std]  # ITD(ms)

def calculate_ild(signal):
    """ ILD
        To be consistent with ITD:
        ild = 10log10(chann1_energy/chann0_energy)
            |>0 chann1 lead
        ild |
            |<0 chann0 lead
    """
    if np.sum(signal[:, 0]**2) == 0:
        print('left channel no signal')
        return np.inf

    if np.sum(signal[:, 1]**2) == 0:
        print('right channel no signal')

    return 10*np.log10(np.sum(signal[:,1]**2)/np.sum(signal[:,0]**2)+1e-10)


# 无噪声环境下提取cues
def GetCues_clean(tar, fs, frame_len, filter_type, cfs, frame_shift=None, max_delay=None, ihc_type=None):
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
    """

    wav_len = tar.shape[0]

    if frame_shift == None:  # default overlap: frame_len/2
        frame_shift = int(frame_len / 2)

    freq_chann_num = cfs.shape[0]

    _, tar_env = Audiotory_peripheral(tar, fs, cfs, filter_type, ihc_type)

    frame_num = int((wav_len - frame_len - frame_len) / frame_shift + 1)
    spatial_cues = np.zeros((freq_chann_num, frame_num, 2), dtype=np.float32)  # [itd_frame,ild_frame]

    if max_delay == None:
        max_delay = frame_len - 1

    ccf_std_all = np.zeros((freq_chann_num, frame_num, max_delay * 2 + 1), dtype=np.float32)

    for freq_chann_i in range(freq_chann_num):
        itds_chann = []
        ilds_chann = []
        ccfs_chann = []
        # print(cfs[freq_chann_i])
        for frame_i in range(frame_num):
            frame_start_pos = frame_i * frame_shift + frame_len
            frame_end_pos = frame_start_pos + frame_len

            tar_chann_frame = tar_env[freq_chann_i, frame_start_pos:frame_end_pos, :]

            ##################
            #             pdb.set_trace()
            # raw_input('pause: ')
            ##################

            # bianural cues

            itd_frame, ccf_std_frame = calculate_itd(tar_chann_frame, fs, max_delay=max_delay)
            ild_frame = calculate_ild(tar_chann_frame)
            # check whether ILD is valid
            if ild_frame == np.inf:
                print(freq_chann_i, frame_i)
                raise Exception('invalid ild')

            spatial_cues[freq_chann_i, frame_i, 0] = itd_frame
            spatial_cues[freq_chann_i, frame_i, 1] = ild_frame
            ccf_std_all[freq_chann_i, frame_i, :] = ccf_std_frame

    return [spatial_cues, ccf_std_all]



# 噪声环境下提取cues
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