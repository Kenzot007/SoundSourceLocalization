import numpy as np
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

def calculate_ic(signal, fs, frame_shift, max_delay=None, tau=10):
    """Compute only IC and cross-correlation function (gamma) without ITD estimation."""
    if signal.ndim != 2 or signal.shape[1] != 2:
        raise ValueError("`signal` must be (N,2) stereo frame")

    if max_delay is None:
        max_delay = int(1e-3 * fs)  # ±1ms default

    alpha = np.exp(-frame_shift / (fs * (tau / 1000)))
    xL = np.ascontiguousarray(signal[:, 0], dtype=float)    # signal in left channel
    xR = np.ascontiguousarray(signal[:, 1], dtype=float)    # signal in right channel
    N = xL.size     # frame length
    D = max_delay

    xR_pad = np.pad(xR, (D, D)) # e.g. change xr from [1] to [0,1,0] if D=1, Accessing arbitrary deferred state
    lags = np.arange(-D, D + 1) # -max_delay < lags < max_delay+1
    K = lags.size   # 2D+1

    cross = np.zeros(K)
    eL = np.full(K, 1e-40)
    eR = np.full(K, 1e-40)

    for n in range(D,N-D):  # Each sampling point of the current frame
        sL = xL[n]
        idx_R = lags + n + D
        sR_vec = xR_pad[idx_R]

        cross = (1 - alpha) * cross + alpha * (sL * sR_vec)
        eL = (1 - alpha) * eL + alpha * (sL ** 2)
        eR = (1 - alpha) * eR + alpha * (sR_vec ** 2)

    denom = np.sqrt(eL * eR) + 1e-20
    gamma = cross / denom

    ic = float(np.max(gamma))
    return ic, gamma

def calculate_itd(gamma, fs, max_delay):
    """Given precomputed gamma, estimate ITD."""
    maxitd=1.0
    maxlag = int(round(maxitd * fs / 1000.0))
    lags = np.arange(-max_delay, max_delay + 1)
    best_idx = int(np.argmax(gamma))
    itd = (lags[best_idx] / fs) * 1e3  # convert to ms
    itd_val = (best_idx - maxlag) * 1000.0 / fs
    return itd_val

def calculate_ild(signal):
    """Compute ILD [dB] for a stereo frame."""
    if signal.ndim != 2 or signal.shape[1] != 2:
        raise ValueError("`signal` must be (N,2) stereo frame")
    eps = np.finfo(float).eps
    rms_left = np.sqrt(np.mean(signal[:, 0] ** 2)) + eps
    rms_right = np.sqrt(np.mean(signal[:, 1] ** 2)) + eps
    return 20.0 * np.log10(rms_left / rms_right)

def GetCues_clean(signal, fs, frame_len, filter_type, cfs, tau, frame_shift=None,
                   max_delay=None, ihc_type=None, c0=0.98):
    """Compute ITD, ILD, IC cues per critical‑band frame (vectorised ITD)."""
    from auditory_model import Audiotory_peripheral  # keep existing front‑end

    if frame_shift is None:
        frame_shift = frame_len // 2
    frame_shift = int(frame_shift)
    if max_delay is None:
        max_delay = int(1e-3 * fs)

    # Auditory periphery → envelope (freq_ch, N, 2)
    bm, _ = Audiotory_peripheral(signal, fs, cfs, filter_type, ihc_type)
    N = signal.shape[0]
    frame_num = (N - frame_len) // frame_shift
    F = len(cfs)

    spatial_cues = np.zeros((F, frame_num, 2), dtype=np.float32)
    gamma_all = np.zeros((F, frame_num, 2 * max_delay + 1), dtype=np.float32)
    ics_all = np.zeros((F, frame_num), dtype=np.float32)

    # Energy threshold (5th percentile of RMS)
    frame_rms = []
    for fi in range(frame_num):
        s = fi * frame_shift
        e = s + frame_len
        frame_rms.extend(np.sqrt(np.mean(bm[:, s:e, :] ** 2, axis=2)).ravel())
    thr = np.percentile(frame_rms, 5)

    alpha_ema = 0.9
    for fi in range(frame_num):
        s = fi * frame_shift
        e = s + frame_len
        for ch in range(F):
            frame = bm[ch, s:e, :]
            if np.sqrt(np.mean(frame ** 2)) < thr:
                continue  # skip low‑energy frame
            ic, gamma = calculate_ic(frame, fs, frame_shift, max_delay, tau)

            if ic < c0:
                continue  # IC gating

            itd = calculate_itd(gamma, fs, max_delay)
            ild = calculate_ild(frame)
            if fi > 0:
                spatial_cues[ch, fi, 0] = alpha_ema * itd + (1 - alpha_ema) * spatial_cues[ch, fi - 1, 0]
                spatial_cues[ch, fi, 1] = alpha_ema * ild + (1 - alpha_ema) * spatial_cues[ch, fi - 1, 1]
            else:
                spatial_cues[ch, fi] = [itd, ild]
            gamma_all[ch, fi] = gamma
            ics_all[ch, fi] = ic
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