import os
import numpy as np
import soundfile as sf
import gammatone.filters as gt_filters
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from binaural_features import GetCues_clean


# 路径配置
dataset_dir = r'G:\GitHub\SoundSourceLocalization\Dataset'
feature_save_dir = r'G:\GitHub\SoundSourceLocalization\features'
image_save_dir = r'G:\GitHub\SoundSourceLocalization\images'
image_itds_save_dir = r'G:\GitHub\SoundSourceLocalization\images\itds'
image_ilds_save_dir = r'G:\GitHub\SoundSourceLocalization\images\ilds'
os.makedirs(feature_save_dir, exist_ok=True)
os.makedirs(image_save_dir, exist_ok=True)

# 获取所有 .wav 文件
audio_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.wav')])

# 定义处理函数
def process_audio_file(filepath, filename):
    signal, fs = sf.read(filepath)
    cfs = gt_filters.centre_freqs(cutoff=80, fs=fs, num_freqs=32)
    frame_len = int(fs * 0.032)
    frame_shift = int(frame_len * 0.5)
    max_delay = int(fs * 0.001)

    # 提取 binaural 特征
    spatial_cues, gamma_all, ics_all = GetCues_clean(
        signal=signal,
        fs=fs,
        frame_len=frame_len,
        filter_type='Gammatone',
        cfs=cfs,
        tau=10,
        frame_shift=frame_shift,
        max_delay=max_delay,
        ihc_type='Christof'
    )

    # 保存特征数据
    feature_path = os.path.join(feature_save_dir, filename.replace('.wav', '.npz'))
    np.savez_compressed(feature_path,
                        spatial_cues=spatial_cues,
                        itd=spatial_cues[..., 0],
                        ild=spatial_cues[..., 1],
                        ic=ics_all)

    itd = spatial_cues[:, :, 0]  # ITD values
    ild = spatial_cues[:, :, 1]  # ILD values
    num_filters, num_frames = itd.shape

    # Apply nonlinearity transformations
    itd = np.log1p(np.abs(itd)) * np.sign(itd)
    ild = np.log1p(np.abs(ild)) * np.sign(ild)

    plt.figure(figsize=(10, 4))
    im_itd = plt.imshow(itd, aspect='auto', cmap='magma', origin='lower')
    plt.title("ITD Cochleagram", fontsize=16)
    plt.ylabel("Filter #", fontsize=14)
    plt.xlabel("Time", fontsize=14)
    plt.colorbar(im_itd)
    image_itds_path = os.path.join(image_itds_save_dir, filename.replace('.wav', '_itd.png'))
    plt.savefig(image_itds_path)
    plt.close()

    plt.figure(figsize=(10, 4))
    im_ild = plt.imshow(ild, aspect='auto', cmap='magma', origin='lower')
    plt.title("ILD Cochleagram", fontsize=16)
    plt.ylabel("Filter #", fontsize=14)
    plt.xlabel("Time", fontsize=14)
    plt.colorbar(im_ild)
    image_ilds_path = os.path.join(image_ilds_save_dir, filename.replace('.wav', '_ild.png'))
    plt.savefig(image_ilds_path)
    plt.close()


# 批量处理所有文件
for fname in audio_files:
    full_path = os.path.join(dataset_dir, fname)
    process_audio_file(full_path, fname)
    print(f"✅{fname}已处理完成 ，特征文件和 ITD/ILD 图像已保存。")

print("✅ 全部音频处理完成，特征文件和 ITD/ILD 图像已保存。")
