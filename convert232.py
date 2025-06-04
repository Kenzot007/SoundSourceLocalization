import os
import numpy as np
import tempfile
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# 设置你的特征目录路径
feature_dir = r'C:\Users\TIANY1\OneDrive - Trinity College Dublin\Documents\SoundSourceLocalization\features'

def convert_to_float32(filename):
    npz_path = os.path.join(feature_dir, filename)
    try:
        with tempfile.NamedTemporaryFile(delete=False, dir=feature_dir, suffix='.npz') as tmpfile:
            with np.load(npz_path) as data:
                converted = {k: v.astype(np.float32) for k, v in data.items()}
                np.savez_compressed(tmpfile.name, **converted)
        os.replace(tmpfile.name, npz_path)
        return f"✅ {filename} 转换完成"
    except Exception as e:
        return f"❌ {filename} 失败: {e}"

if __name__ == '__main__':
    # 找到所有 .npz 文件
    all_files = [f for f in os.listdir(feature_dir) if f.endswith('.npz')]

    # 设置进程数（你有 20 核，可自由调整）
    num_workers = min(cpu_count(), 16)

    print(f"开始转换 {len(all_files)} 个 .npz 文件为 float32...")
    with Pool(num_workers) as pool:
        for result in tqdm(pool.imap_unordered(convert_to_float32, all_files), total=len(all_files)):
            print(result)

    print("✅ 所有文件转换完成，节省空间成功！")
