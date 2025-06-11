import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import re
from tqdm.notebook import tqdm
import ipywidgets as widgets
widgets.IntProgress(value=50, min=0, max=100)
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class BinauralCueDataset(Dataset):
    def __init__(self, npz_dir, audio_ids=range(1, 701)):
        self.dir = npz_dir
        pattern = re.compile(r'main_audio_(\d+)_azi(\d+)\.npz')
        self.files = []
        for f in os.listdir(npz_dir):
            if f.endswith('.npz'):
                match = pattern.match(f)
                if match and int(match.group(1)) in audio_ids:
                    self.files.append(f)
        self.files.sort()

        print(f"📁 已加载 {len(self.files)} 个 .npz 文件，共 {len(self)} 个样本。")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.dir, self.files[idx])
        data = np.load(path)
        itd = data["itd"].astype(np.float32)
        ild = data["ild"].astype(np.float32)
        ic = data["ic"].astype(np.float32)

        cue = np.stack([itd, ild, ic], axis=0)  # [3, filters, frames]

        # 提取 azimuth label
        azimuth = int(re.search(r'azi(\d+)', self.files[idx]).group(1))
        label = azimuth // 5  # 共72类（0-71）

        return cue, label
    

# 一维卷积分支：包含 Conv1d 层、BatchNorm1d 和 ReLU 激活
class ConvBranch(nn.Module):
    def __init__(self, input_channels=32, conv_channels=64, kernel_size=5, stride=1, num_layers=2, use_batchnorm=True):
        super(ConvBranch, self).__init__()
        layers = []
        in_ch = input_channels
        for i in range(num_layers):
            out_ch = conv_channels
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding=kernel_size//2))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        # x: [B, input_channels, L]  （L为时间长度，如44100）
        return self.conv(x)         # 输出: [B, conv_channels, L]

# 自注意力池化：将可变长度的时间序列特征加权汇聚为一个固定向量
class SelfAttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttentionPooling, self).__init__()
        # 可学习的线性层，用于计算每个时间步的注意力分数
        self.attn_score = nn.Linear(embed_dim, 1)
    def forward(self, x):
        # x: [B, L, embed_dim]  （输入特征需先调换维度到 [时间步, 特征]）
        scores = self.attn_score(x)                 # 计算注意力分数: [B, L, 1]
        weights = torch.softmax(scores, dim=1)      # 对时间维度做softmax归一化
        context = (x * weights).sum(dim=1)          # 加权求和得到上下文向量: [B, embed_dim]
        return context

# 单个线索分支模块：Conv1D 提取特征 + 注意力池化得到线索表示向量
class CueBranch(nn.Module):
    def __init__(self, input_channels=32, conv_channels=64, kernel_size=5, stride=1, num_layers=2, embed_dim=None, use_batchnorm=True):
        super(CueBranch, self).__init__()
        self.conv_net = ConvBranch(input_channels, conv_channels, kernel_size, stride, num_layers, use_batchnorm)
        # 设置嵌入维度：若未指定则与卷积输出通道数相同
        self.embed_dim = conv_channels if embed_dim is None else embed_dim
        # 若需要将卷积输出映射到不同的embed_dim，可添加线性层
        if embed_dim is not None and embed_dim != conv_channels:
            self.proj = nn.Linear(conv_channels, embed_dim)
        else:
            self.proj = None
        self.attn_pool = SelfAttentionPooling(self.embed_dim)
    def forward(self, x):
        # x: [B, input_channels, L]
        feat = self.conv_net(x)               # 卷积提取特征: [B, conv_channels, L]
        feat = feat.transpose(1, 2)           # 调整为 [B, L, conv_channels] 以方便注意力计算
        if self.proj is not None:
            feat = self.proj(feat)           # 可选：映射到指定的嵌入维度 [B, L, embed_dim]
        cue_vector = self.attn_pool(feat)     # 注意力池化得到线索上下文向量: [B, embed_dim]
        return cue_vector

# 主模型：包含三个线索分支、跨线索自注意力层和最终分类器
class SoundLocalizationModel(nn.Module):
    def __init__(self, num_classes=72, input_channels_per_cue=32, conv_channels=64, kernel_size=5, stride=1,
                 num_layers=2, embed_dim=64, num_heads=4, use_batchnorm=True):
        super(SoundLocalizationModel, self).__init__()
        # 三个独立的线索分支
        self.cue_branches = nn.ModuleList([
            CueBranch(input_channels_per_cue, conv_channels, kernel_size, stride, num_layers, embed_dim, use_batchnorm)
            for _ in range(3)
        ])
        # 跨线索多头自注意力层，将 embed_dim 维的3个线索向量作为序列
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)            # 层归一化，规范注意力输出
        # 全连接分类层，将3*embed_dim映射为num_classes（72个方位角类别）
        self.fc = nn.Linear(embed_dim * 3, num_classes)
    def forward(self, x):
        # x: [B, 3, 32, L]  其中3表示三个线索通道 (ITD, ILD, IC)
        B, cue_count, C, L = x.shape
        assert cue_count == 3, "模型期望输入包含3个线索通道"
        # 分别通过每个线索分支提取表示向量
        cue_vectors = []  # 将收集每个分支输出 [B, embed_dim]
        for i, branch in enumerate(self.cue_branches):
            cue_input = x[:, i]                   # 取出第 i 个线索: [B, 32, L]
            vec = branch(cue_input)               # 得到该线索的表示向量: [B, embed_dim]
            cue_vectors.append(vec)
        # 将三个线索向量堆叠成序列，形状 [B, 3, embed_dim]
        seq = torch.stack(cue_vectors, dim=1)
        # 自注意力层：让每个线索向量与其他线索交互得到新的表示
        attn_out, _ = self.cross_attn(seq, seq, seq)   # [B, 3, embed_dim]
        attn_out = self.norm(attn_out)                 # 层归一化输出
        # 将3个线索向量展平为单一向量 [B, 3*embed_dim]
        combined = attn_out.reshape(B, -1)
        # 全连接分类，输出72维类别分数
        logits = self.fc(combined)                     # [B, 72]
        # 模型输出为未归一化的得分，可在需要时使用 Softmax 做归一化:
        # probs = torch.softmax(logits, dim=1)
        return logits

random.seed(42)
random_ids = random.sample(range)
full_dataset = BinauralCueDataset(r"C:\Users\TIANY1\OneDrive - Trinity College Dublin\Documents\SoundSourceLocalization\features")
train_dataset, val_dataset = random_split(full_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("The device type is: ", device)
model = model = SoundLocalizationModel(num_classes=72, input_channels_per_cue=32, conv_channels=64, kernel_size=5,
                               stride=1, num_layers=2, embed_dim=64, num_heads=4, use_batchnorm=True).to(device)
model_path = r"C:\Users\TIANY1\OneDrive - Trinity College Dublin\Documents\SoundSourceLocalization\checkpoints\model2\epoch_20.pth"
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


val_path = r"C:\Users\TIANY1\OneDrive - Trinity College Dublin\Documents\SoundSourceLocalization\features"
val_dataset = BinauralCueDataset(val_path, audio_ids=range(561, 701))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)


all_preds, all_labels = [], []

with torch.no_grad():
    for X, y in val_loader:
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

accuracy = np.mean(all_preds == all_labels)
print(f"Top-1 Accuracy: {accuracy * 100:.2f}%")

def mean_class_accuracy(y_true, y_pred, num_classes=72):
    class_accs = []
    for cls in range(num_classes):
        cls_mask = (y_true == cls)
        if cls_mask.sum() == 0: continue
        acc = (y_pred[cls_mask] == y_true[cls_mask]).sum() / cls_mask.sum()
        class_accs.append(acc)
    return np.mean(class_accs)

mean_acc = mean_class_accuracy(all_labels, all_preds)
print(f"Mean Accuracy per Class: {mean_acc * 100:.2f}%")

from sklearn.metrics import classification_report
print(classification_report(all_labels, all_preds, digits=3))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

