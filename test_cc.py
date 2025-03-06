import numpy as np
import matplotlib.pyplot as plt

def generate_test_signals(fs, duration, freq, delay):
    """生成左右耳音频信号，右耳比左耳延迟 delay 秒"""
    t = np.arange(0, duration, 1/fs)  # 时间向量
    x = np.sin(2 * np.pi * freq * t)  # 生成正弦波
    y = np.roll(x, -int(delay * fs))   # 负号确保 y 是延迟，而不是提前
    return t, x, y

def Efficient_ccf(left_signal, right_signal):
    """使用 FFT 计算互相关"""
    if left_signal.shape[0] != right_signal.shape[0]:
        raise Exception('length mismatch')

    signal_len = left_signal.shape[0]
    window = np.hanning(signal_len)  # 加窗减少泄露
    left_signal = left_signal * window
    right_signal = right_signal * window

    left_fft = np.fft.fft(left_signal, 2 * signal_len)  # zero-padding
    right_fft = np.fft.fft(right_signal, 2 * signal_len)
    ccf_unshift = np.real(np.fft.ifft(left_fft * np.conjugate(right_fft)))
    ccf = np.fft.fftshift(ccf_unshift)

    # 归一化
    ccf = np.real(ccf) / np.max(np.abs(ccf))
    return ccf

# 生成信号
fs = 16000       # 采样率
duration = 0.02  # 只取 20ms 以便可视化
freq = 500       # 频率 (Hz)
true_delay = 5e-3  # 真实延迟 5ms

t, x, y = generate_test_signals(fs, duration, freq, true_delay)

# 计算互相关（时域方法）
# ccf = np.correlate(x, y, mode='full')
# ccf = ccf / np.max(np.abs(ccf))  # 归一化

# frequency domain
ccf = Efficient_ccf(x, y)



# 正确计算 lags 轴
lags = np.arange(-len(x), len(x)) / fs  # 计算时间轴（单位：秒）

# 绘制互相关函数
plt.figure(figsize=(8, 4))
plt.plot(lags * 1e3, ccf, label="CCF (Time Domain)")
plt.axvline(x=true_delay * 1e3, color="r", linestyle="--", label=f"True Delay: {true_delay*1e3:.1f} ms")
plt.xlabel("Lag (ms)")
plt.ylabel("Cross-Correlation")
plt.title("Cross-Correlation Function Test")
plt.legend()
plt.grid()
plt.show()

# 绘制信号波形
plt.figure(figsize=(8, 4))
plt.plot(t * 1e3, x, label="Left Channel (x)", color='r', alpha=0.7)
plt.plot(t * 1e3, y, label="Right Channel (y, delayed)", color='b', linestyle="--", alpha=0.7)
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.title("Left & Right Ear Signals")
plt.legend()
plt.grid()
plt.show()

# 计算估计延迟
max_lag_index = np.argmax(ccf)
estimated_delay = lags[max_lag_index] * 1e3  # 单位 ms
print(f"Estimated Delay: {estimated_delay:.2f} ms")


# 计算 FFT（快速傅立叶变换）
left_fft = np.fft.fft(x)  # 左耳频谱
right_fft = np.fft.fft(y) # 右耳频谱

# 计算每个频率分量的能量
left_power = np.abs(left_fft) ** 2
right_power = np.abs(right_fft) ** 2

# 计算频域 ILD
ild_freq = 10 * np.log10(right_power / left_power)
print(ild_freq)

