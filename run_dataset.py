import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset 
import re
from tqdm import tqdm

class BinauralCueDataset(Dataset):
    def __init__(self, npz_dir, audio_ids=range(1, 101)):
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

import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class AzimuthResNetCNN(nn.Module):
    def __init__(self, num_classes=72):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer1 = ResBlock(32, 64, downsample=True)
        self.layer2 = ResBlock(64, 128, downsample=True)
        self.layer3 = ResBlock(128, 128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)        # shape: [B, 128, 1, 1]
        x = torch.flatten(x, 1) # shape: [B, 128]
        x = self.dropout(x)
        return self.fc(x)


import torch
from torch.utils.data import DataLoader

train_dataset = BinauralCueDataset(r"C:\Users\TIANY1\OneDrive - Trinity College Dublin\Documents\SoundSourceLocalization\features")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = AzimuthResNetCNN(num_classes=72).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.02)
criterion = nn.CrossEntropyLoss()

print("模型所在设备：", next(model.parameters()).device)

for epoch in range(20):
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for cues, labels in train_loader:
        cues, labels = cues.cuda(), labels.cuda()
        outputs = model(cues)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    print(f"✅ Epoch {epoch+1} 完成，总损失: {total_loss:.4f}")