import numpy as np
from auditory_model import Audiotory_peripheral, Haircell_model

def calculate_itd_ild_ic(in1, in2, sfreq, maxitd=1.0, maxild=7.0, tau=10, ofst=0):

    in1 = np.ravel(in1).astype(float)
    in2 = np.ravel(in2).astype(float)
    N = max(in1.size, in2.size)

    if in1.size != in2.size:
        if in1.size < N:
            in1 = np.pad(in1, (0, N - in1.size))
        if in2.size < N:
            in2 = np.pad(in2, (0, N - in2.size))

    if np.isscalar(tau):
        tau_ild = tau_itd = tau_ic = tau
    else:
        tau_ild, tau_itd, tau_ic = tau

    alpha_ild = 1.0 / (tau_ild * sfreq / 1000.0) if tau_ild > 0 else 1.0
    alpha_itd = 1.0 / (tau_itd * sfreq / 1000.0) if tau_itd > 0 else 1.0
    alpha_ic = 1.0 / (tau_ic * sfreq / 1000.0) if tau_ic > 0 else 1.0

    maxlag = int(round(maxitd * sfreq / 1000.0))
    pad = maxlag
    in2_padded = np.pad(in2, (pad, pad))
    n_lags = 2 * maxlag + 1
    lag_indices = np.arange(-maxlag, maxlag + 1, dtype=int)

    itd = np.zeros(N)
    ild = np.zeros(N)
    ic = np.zeros(N)
    pow_signal = np.zeros(N)

    cross_fast = np.zeros(n_lags)
    left_fast = np.full(n_lags, 1e-40)
    right_fast = np.full(n_lags, 1e-40)
    cross_slow = np.zeros(n_lags)
    left_slow = np.full(n_lags, 1e-40)
    right_slow = np.full(n_lags, 1e-40)

    for i in range(N):
        j_pad = lag_indices + i + pad
        right_vals = in2_padded[j_pad]
        left_val = in1[i]

        cross_fast = cross_fast * (1 - alpha_itd) + alpha_itd * (left_val * right_vals)
        left_fast = left_fast * (1 - alpha_itd) + alpha_itd * (left_val ** 2)
        right_fast = right_fast * (1 - alpha_itd) + alpha_itd * (right_vals ** 2)

        cross_slow = cross_slow * (1 - alpha_ic) + alpha_ic * (left_val * right_vals)
        left_slow = left_slow * (1 - alpha_ic) + alpha_ic * (left_val ** 2)
        right_slow = right_slow * (1 - alpha_ic) + alpha_ic * (right_vals ** 2)

        denom_fast = np.sqrt(left_fast * right_fast) + 1e-20
        corr_fast = cross_fast / denom_fast

        best_idx = np.argmax(corr_fast)
        itd[i] = (best_idx - maxlag) * 1000.0 / sfreq

        left_energy = left_fast[best_idx]
        right_energy = right_fast[best_idx]
        ild_val = 10.0 * np.log10((left_energy + 1e-40) / (right_energy + 1e-40))
        ild[i] = np.clip(ild_val, -maxild, maxild)

        denom_slow = np.sqrt(left_slow * right_slow) + 1e-20
        corr_slow = cross_slow / denom_slow
        ic[i] = np.max(corr_slow)

        pow_signal[i] = left_energy + right_energy

    if ofst > 0 and ofst < N:
        return ild[ofst:], itd[ofst:], ic[ofst:], pow_signal[ofst:]
    else:
        return ild, itd, ic, pow_signal

def GetCues_clean(signal, fs, frame_len, filter_type, cfs, alpha=None, frame_shift=None,
                   max_delay=None, ihc_type=None):
    """Compute ITD, ILD, IC cues per critical‑band frame (vectorised ITD)."""
    from auditory_model import Audiotory_peripheral  # keep existing front‑end

    # Auditory periphery → envelope (freq_ch, N, 2)
    filtered_env, _ = Audiotory_peripheral(signal, fs, cfs, filter_type, ihc_type)
    F, N, _ = filtered_env.shape

    itd_all = np.zeros((F, N))
    ild_all = np.zeros((F, N))
    ic_all = np.zeros((F, N))
    power_all = np.zeros((F, N))

    for f_idx, fcenter in enumerate(cfs):
        left = filtered_env[f_idx, :, 0]
        right = filtered_env[f_idx, :, 1]
        ild, itd, ic, power = calculate_itd_ild_ic(left, right, sfreq=fs)
        itd_all[f_idx, :len(itd)] = itd
        ild_all[f_idx, :len(ild)] = ild
        ic_all[f_idx, :len(ic)] = ic
        power_all[f_idx, :len(power)] = power
    return {
        "itd": itd_all,
        "ild": ild_all,
        "ic": ic_all,
        "power": power_all
    }
