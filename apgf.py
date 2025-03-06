import numpy as np
from scipy.signal import lfilter, lfilter_zi
from scipy.special import factorial

class APGF:
    """
    All-Pole Gammatone Filter (APGF) 实现，用于处理音频信号。
    """
    def __init__(self, sample_rate, low_frequency=50, high_frequency=3500, channel_count=32):
        """
        初始化 APGF 滤波器组
        :param sample_rate: 采样率
        :param low_frequency: 最低频率
        :param high_frequency: 最高频率
        :param channel_count: 滤波器组的通道数
        """
        self.sample_rate = sample_rate
        self.low_frequency = low_frequency
        self.high_frequency = high_frequency
        self.channel_count = channel_count

        # 计算中心频率和带宽
        self.center_frequencies = self._compute_center_frequencies()
        self.bandwidths = self._compute_bandwidths()

    def _compute_center_frequencies(self):
        """计算滤波器组的中心频率"""
        low_erb = 21.4 * np.log10(4.37e-3 * self.low_frequency + 1)
        high_erb = 21.4 * np.log10(4.37e-3 * self.high_frequency + 1)
        erb_space = np.linspace(low_erb, high_erb, self.channel_count)
        return (10 ** (erb_space / 21.4) - 1) / 4.37e-3

    def _compute_bandwidths(self):
        """计算每个滤波器的带宽"""
        erb = 24.7 * (4.37e-3 * self.center_frequencies + 1)
        an_inverse = (factorial(3) ** 2) / (np.pi * factorial(6) * 2 ** -5)
        return erb * an_inverse

    def _design_filter(self, center_frequency, bandwidth):
        """设计每个滤波器"""
        t = 1 / self.sample_rate
        phi = 2 * np.pi * bandwidth * t
        theta = 2 * np.pi * center_frequency * t
        p = np.exp(-phi + 1j * theta)  # 极点位置
        a = np.poly([p, np.conjugate(p)])  # 滤波器系数
        b = [1.0]  # 系数 B
        return b, a

    def process(self, signal):
        """
        使用 APGF 滤波器组处理信号
        :param signal: 输入信号
        :return: 滤波后的结果 (channel_count, signal_length)
        """
        results = []
        for cf, bw in zip(self.center_frequencies, self.bandwidths):
            b, a = self._design_filter(cf, bw)
            zi = lfilter_zi(b, a) * signal[0]  # 初始化状态
            filtered_signal, _ = lfilter(b, a, signal, zi=zi)
            results.append(filtered_signal)
        return np.array(results)
