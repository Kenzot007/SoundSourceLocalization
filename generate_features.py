import os
import numpy as np
import soundfile as sf
from binaural_features import GetCues_clean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# 路径设置
dataset_dir = r'C:\Users\TIANY1\OneDrive - Trinity College Dublin\Documents\SoundSourceLocalization\Dataset'
feature_save_dir = r'C:\Users\TIANY1\OneDrive - Trinity College Dublin\Documents\SoundSourceLocalization\features'
image_ilds_save_dir = r'C:\Users\TIANY1\OneDrive - Trinity College Dublin\Documents\SoundSourceLocalization\images\ilds'
image_itds_save_dir = r'C:\Users\TIANY1\OneDrive - Trinity College Dublin\Documents\SoundSourceLocalization\images\itds'
image_ics_save_dir = r'C:\Users\TIANY1\OneDrive - Trinity College Dublin\Documents\SoundSourceLocalization\images\ics'
os.makedirs(feature_save_dir, exist_ok=True)
os.makedirs(image_ilds_save_dir, exist_ok=True)
os.makedirs(image_itds_save_dir, exist_ok=True)
os.makedirs(image_ics_save_dir, exist_ok=True)

# 文件列表
audio_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.wav')])

# 并行处理的函数（必须是顶层函数）
def process_one_audio_file(filename):
    try:
        feature_path = os.path.join(feature_save_dir, filename.replace('.wav', '.npz'))
        if os.path.exists(feature_path):
            print(f"⏭️ 已存在，跳过：{filename}")
            return
        
        filepath = os.path.join(dataset_dir, filename)
        signal, fs = sf.read(filepath)

        # 参数
        from gammatone.filters import centre_freqs
        cfs = centre_freqs(cutoff=80, fs=fs, num_freqs=32)
        frame_len = int(fs * 0.032)
        frame_shift = int(frame_len * 0.5)
        max_delay = int(fs * 0.001)

        cues = GetCues_clean(signal=signal, fs=fs,
                             frame_len=frame_len,
                             filter_type='Gammatone',
                             cfs=cfs,
                             alpha=10,
                             frame_shift=frame_shift,
                             max_delay=max_delay,
                             ihc_type='Christof')

        itd, ild, ic = cues["itd"], cues["ild"], cues["ic"]

        # 保存特征文件
        np.savez_compressed(feature_path, itd=itd, ild=ild, ic=ic)

        # 可视化
        itd_plot = np.log1p(np.abs(itd)) * np.sign(itd)
        ild_plot = np.log1p(np.abs(ild)) * np.sign(ild)
        ic_plot = np.log1p(np.abs(ic)) * np.sign(ic)

        plt.figure(figsize=(10, 4))
        im_itd = plt.imshow(itd_plot, aspect='auto', cmap='coolwarm', origin='lower')
        plt.title("ITD Cochleagram", fontsize=16)
        plt.ylabel("Filter #", fontsize=14)
        plt.xlabel("Time", fontsize=14)
        plt.colorbar(im_itd)
        image_itds_path = os.path.join(image_itds_save_dir, filename.replace(".wav", "_itd.png"))
        plt.savefig(image_itds_path)
        plt.close()

        plt.figure(figsize=(10, 4))
        im_ild = plt.imshow(ild_plot, aspect='auto', cmap='coolwarm', origin='lower')
        plt.title("ILD Cochleagram", fontsize=16)
        plt.ylabel("Filter #", fontsize=14)
        plt.xlabel("Time", fontsize=14)
        plt.colorbar(im_ild)
        image_ilds_path = os.path.join(image_ilds_save_dir, filename.replace(".wav", "_ild.png"))
        plt.savefig(image_ilds_path)
        plt.close()

        plt.figure(figsize=(10, 4))
        im_ic = plt.imshow(ic_plot, aspect='auto', cmap='viridis', origin='lower', vmin=0, vmax=1)
        plt.title("IC Cochleagram", fontsize=16)
        plt.ylabel("Filter #", fontsize=14)
        plt.xlabel("Time", fontsize=14)
        plt.colorbar(im_ic)
        image_ics_path = os.path.join(image_ics_save_dir, filename.replace(".wav", "_ic.png"))
        plt.savefig(image_ics_path)
        plt.close()
        

        print(f"{filename} 已处理完成。")
    except Exception as e:
        print(f"处理 {filename} 时出错：{e}")

# 主函数入口
if __name__ == '__main__':
    num_workers = min(cpu_count(), 16)  # 控制最大线程数
    with Pool(num_workers) as pool:
        pool.map(process_one_audio_file, audio_files)
    print("所有音频处理完成 ✅")
