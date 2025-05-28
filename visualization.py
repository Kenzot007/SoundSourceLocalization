from scipy.fftpack import fft
import gammatone.filters as gt_filters
import scipy.signal as sp


def compute_fft(signal, fs):
    N = len(signal)
    freq = np.linspace(0, fs / 2, N // 2)
    fft_magnitude = np.abs(fft(signal))[:N // 2]
    fft_magnitude_db = 20 * np.log10(fft_magnitude + 1e-10)
    return freq, fft_magnitude_db

def visualize_time(left_signal, right_signal, t, duration):
    plt.figure(figsize=(12, 6))
    plt.plot(t * 1e3, left_signal, label="Left Ear Signal", linestyle='-', color='b')
    plt.plot(t * 1e3, right_signal, label="Right Ear Signal", linestyle='--', color='r')
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (dB)")
    plt.title("Time Domain")
    plt.legend()
    plt.grid()
    plt.xlim(0, duration * 1e3)
    plt.tight_layout()
    plt.show()
    return None

def visualize_frequency(left_signal, right_signal, fs):
    freqs, left_fft_db = compute_fft(left_signal, fs)
    _, right_fft_db = compute_fft(right_signal, fs)
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, left_fft_db, label="Left Ear FFT", linestyle='-', color='b')
    plt.plot(freqs, right_fft_db, label="Right Ear FFT", linestyle='--', color='r')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.title("Frequency Domain")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    return None

def visualize_binary_cues(itd_data, ild_data, frames):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(frames, itd_data, marker='o', linestyle='-', color='blue')
    plt.title('ITD vs Frame Index')
    plt.xlabel('Frame Index')
    plt.ylim(-1, 1)
    plt.ylabel('ITD (ms)')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(frames, ild_data, marker='s', linestyle='-', color='g')
    plt.title('ILD vs Frame Index')
    plt.xlabel('Frame Index')
    plt.ylabel('ILD (dB)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    return None

# def visualize_ccf(left_signal, right_signal, fs):
#     ic = Efficient_ccf(left_signal, right_signal)
#     lags = np.arange(-len(ic) // 2, len(ic) // 2)
#     plt.plot(lags / fs * 1e3, ic)
#     plt.xlabel("Lag (seconds)")
#     plt.ylabel("Cross-Correlation")
#     plt.title("Efficient CCF")
#     plt.show()

# Audiogram Visualization
def visualize_audiogram(audiogram):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(audiogram[:, :, 0], aspect='auto', origin='lower', cmap='hot')
    plt.colorbar(label='Amplitude')
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Channels')
    plt.title('Gammatone Spectrogram (Left Ear)')

    plt.subplot(1, 2, 2)
    plt.imshow(audiogram[:, :, 1], aspect='auto', origin='lower', cmap='hot')
    plt.colorbar(label='Amplitude')
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Channels')
    plt.title('Gammatone Spectrogram (Right Ear)')

    plt.tight_layout()
    plt.show()
    return None

# def visualize_spatial_cues(spatial_cues):
#     itd = spatial_cues[:, :, 0]
#     ild = spatial_cues[:, :, 1]
#
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#
#     # ITD Heatmap
#     im1 = axes[0].imshow(itd, aspect='auto', cmap='coolwarm', origin='lower')
#     axes[0].set_title("ITD Heatmap")
#     axes[0].set_xlabel("Frame Index")
#     axes[0].set_ylabel("Frequency Channel")
#     fig.colorbar(im1, ax=axes[0], label="ITD (ms)")
#
#     # ILD Heatmap
#     im2 = axes[1].imshow(ild, aspect='auto', cmap='coolwarm', origin='lower')
#     axes[1].set_title("ILD Heatmap")
#     axes[1].set_xlabel("Frame Index")
#     axes[1].set_ylabel("Frequency Channel")
#     fig.colorbar(im2, ax=axes[1], label="ILD (dB)")
#
#     plt.tight_layout()
#     plt.show()

import numpy as np
import matplotlib.pyplot as plt

def visualize_spatial_cues(spatial_cues, nonlinearity="log"):
    """
    Visualize ITD and ILD heatmaps similar to a cochleagram.

    Args:
        spatial_cues: NumPy array of shape (num_filters, num_frames, 2).
        nonlinearity: Apply 'log' or 'power' transformation to enhance visibility.
    """
    itd = spatial_cues[:, :, 0]  # ITD values
    ild = spatial_cues[:, :, 1]  # ILD values
    num_filters, num_frames = itd.shape

    # Apply nonlinearity transformations
    if nonlinearity == "log":
        itd = np.log1p(np.abs(itd)) * np.sign(itd)
        ild = np.log1p(np.abs(ild)) * np.sign(ild)
    elif nonlinearity == "power":
        itd = np.power(itd, 0.3)
        ild = np.power(ild, 0.3)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    ax = axes[0]
    im = ax.imshow(itd, aspect='auto', cmap='magma', origin='lower')
    ax.set_title("ITD Cochleagram", fontsize=16)
    ax.set_ylabel("Filter #", fontsize=14)
    ax.set_xlabel("Time", fontsize=14)
    fig.colorbar(im, ax=ax)

    ax = axes[1]
    im = ax.imshow(ild, aspect='auto', cmap='magma', origin='lower')
    ax.set_title("ILD Cochleagram", fontsize=16)
    ax.set_ylabel("Filter #", fontsize=14)
    ax.set_xlabel("Time", fontsize=14)
    fig.colorbar(im, ax=ax)
    plt.show()

def plot_left_audiogram_spectrogram(left_audiogram, fs, num_cols=4):
    """
    Plot the spectrograms of all gammatone filtered signals, arranged in one large graph.

    :param left_audiogram: num_filters, num_samples
    :param fs: samping rate
    :param num_cols: Number of filter channels displayed per line (Default: 4)
    """
    num_filters, num_samples = left_audiogram.shape
    num_rows = int(np.ceil(num_filters / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 2), sharex=True, sharey=True)
    axes = axes.flatten()

    for i in range(num_filters):
        f, t, Sxx = sp.spectrogram(left_audiogram[i], fs, nperseg=256)
        axes[i].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='auto', cmap='magma')
        axes[i].set_title(f'Filter {i}')
        axes[i].set_ylabel('Freq (Hz)')

    for j in range(num_filters, len(axes)):
        fig.delaxes(axes[j])

    plt.xlabel('Time (s)')
    plt.suptitle('Spectrogram of Left Audiogram Across Filters', fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_right_audiogram_spectrogram(right_audiogram, fs, num_cols=4):
    """
    Plot the spectrograms of all gammatone filtered signals, arranged in one large graph.

    :param left_audiogram: num_filters, num_samples
    :param fs: samping rate
    :param num_cols: Number of filter channels displayed per line (Default: 4)
    """
    num_filters, num_samples = right_audiogram.shape
    num_rows = int(np.ceil(num_filters / num_cols))  # 计算行数

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 2), sharex=True, sharey=True)
    axes = axes.flatten()  # 把 axes 转成一维数组，方便索引

    for i in range(num_filters):
        f, t, Sxx = sp.spectrogram(right_audiogram[i], fs, nperseg=256)
        axes[i].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='auto', cmap='magma')
        axes[i].set_title(f'Filter {i}')
        axes[i].set_ylabel('Freq (Hz)')

    # 隐藏多余的 subplot
    for j in range(num_filters, len(axes)):
        fig.delaxes(axes[j])

    plt.xlabel('Time (s)')
    plt.suptitle('Spectrogram of Right Audiogram Across Filters', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_audiogram(audiogram, title="Cochleagram"):
    plt.figure(figsize=(8, 6))
    plt.imshow(audiogram, aspect='auto', cmap='magma', origin='lower')

    plt.xlabel("Time")
    plt.ylabel("Filter #")
    plt.title(title)
    plt.colorbar(label="Energy")
    plt.show()

def plot_gammatone_frequency_response(fs, cfs, coefs, impulse_len=512):
    plt.figure(figsize=(10, 6))

    impulse = np.zeros(impulse_len)
    impulse[0] = 1.0

    responses = gt_filters.erb_filterbank(impulse, coefs)
    freqs = np.fft.rfftfreq(impulse_len, 1/fs)
    for i, h in enumerate(responses):
        H = np.fft.rfft(h, n=impulse_len)
        magnitude = 20 * np.log10(np.abs(H))
        plt.plot(freqs, magnitude, label=f'{int(cfs[i])} Hz')

    plt.title("Gammatone Filter Frequency Responses")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain (dB)")
    plt.ylim([-40, 5])
    plt.grid(True)
    plt.legend(loc="lower left", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()