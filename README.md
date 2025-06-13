# Sound Source Localization

This project includes a series of Python scripts for binaural feature extraction, auditory modeling, and audio processing.

---

## Project Structure

### **1. `binaural_feature.py`**
Functions for extracting binaural audio features:
- `calculate_itd_ild_ic`: Calculate ITD, ILD and IC spatial cues are extracted frame by frame from the left and right ear signals to simulate the time difference, intensity difference and coherence in human binaural hearing.
- `GetCues_clean`: Perform cochlear filtering and envelope extraction on binaural audio signals, calculate and output complete ITD, ILD, IC and energy feature maps by frequency band.

---

### **2. `auditory_model.py`**
Implements an auditory peripheral model:
- `filters`: Filtering functions
- `auditory_peripheral` & `haircell_model`: Peripheral auditory processing and hair cell modeling

**Configuration options**:
- `filter_type`: `GF_spectrogram` (Gammatone filterbank)
- `model_type`: `Roman, Half-wave rectification + signal envelope`

**Database creation**:
1. 700 main audio files with 1 second long. Audio from human speech, LibriSpeech ASR: http://www.openslr.org/12
2. 10 noisy types from Esc-50 dataset https://github.com/karolpiczak/ESC-50. Each type contains 20 noisy files.
3. Combine the main audio with 2 noisy audio. The main audio and 2 noisy audio are processed with HRTF and each main audio has 72 classes(from 0° to 355°). The direction of 2 noisy audio are randomly and snr of them are randomly as well.

**Result Analysis**:
1. A sorrounded audio to see the changes of itd and ild over the time:
<img width="767" alt="image" src="https://github.com/user-attachments/assets/6b11f06b-e4e4-41f2-8961-fcefde4d213e" />

2. An audio with different noisy level:

SNR=0:

<img width="780" alt="image" src="https://github.com/user-attachments/assets/1a7af932-84cd-4d94-b0bb-5bd2946c3c7f" />

SNR=15:

<img width="791" alt="image" src="https://github.com/user-attachments/assets/5708f0d3-559d-4637-bb1d-b4bd10fe8c94" />

**Result Records**:
***Model1***:
Epoch 1/20: Train Loss=3.5827, Top-1=5.90%, Top-5=63.85% | Evaluation Loss=3.1251, Top-1=10.28%, Top-5=76.11%

Epoch 2/20: Train Loss=3.1532, Top-1=10.42%, Top-5=74.20% | Evaluation Loss=2.9404, Top-1=14.24%, Top-5=78.61%

Epoch 3/20: Train Loss=2.9757, Top-1=13.49%, Top-5=76.75% | Evaluation Loss=2.8897, Top-1=12.99%, Top-5=80.69%

Epoch 4/20: Train Loss=2.8280, Top-1=15.57%, Top-5=80.35% | Evaluation Loss=2.7377, Top-1=17.78%, Top-5=83.54%

Epoch 5/20: Train Loss=2.7288, Top-1=17.76%, Top-5=81.56% | Evaluation Loss=2.9018, Top-1=16.39%, Top-5=78.82%

Epoch 6/20: Train Loss=2.6285, Top-1=20.21%, Top-5=83.51% | Evaluation Loss=2.6296, Top-1=20.56%, Top-5=85.14%

Epoch 7/20: Train Loss=2.5368, Top-1=23.02%, Top-5=85.43% | Evaluation Loss=2.6923, Top-1=19.38%, Top-5=81.18%

Epoch 8/20: Train Loss=2.4715, Top-1=23.30%, Top-5=86.48% | Evaluation Loss=2.5902, Top-1=22.29%, Top-5=84.44%

Epoch 9/20: Train Loss=2.3948, Top-1=25.61%, Top-5=87.22% | Evaluation Loss=2.6165, Top-1=21.04%, Top-5=86.25%

Epoch 10/20: Train Loss=2.3453, Top-1=26.02%, Top-5=88.40% | Evaluation Loss=2.5304, Top-1=23.68%, Top-5=85.69%

Epoch 11/20: Train Loss=2.2509, Top-1=28.99%, Top-5=89.39% | Evaluation Loss=2.6195, Top-1=22.29%, Top-5=85.35%

Epoch 12/20: Train Loss=2.2040, Top-1=30.05%, Top-5=90.03% | Evaluation Loss=2.6638, Top-1=19.38%, Top-5=84.38%


***Model 2***
Epoch 1/20: Train Loss=2.9714, Top-1=13.36%, Top-5=77.45% | Evaluation Loss=2.6607, Top-1=19.13%, Top-5=81.98%

Epoch 2/20: Train Loss=2.5272, Top-1=21.35%, Top-5=85.37% | Evaluation Loss=2.3689, Top-1=24.14%, Top-5=88.78%

Epoch 3/20: Train Loss=2.3454, Top-1=25.62%, Top-5=88.01% | Evaluation Loss=2.2165, Top-1=28.12%, Top-5=90.50%

Epoch 4/20: Train Loss=2.2260, Top-1=28.45%, Top-5=89.49% | Evaluation Loss=2.2727, Top-1=27.52%, Top-5=86.33%

Epoch 5/20: Train Loss=2.1367, Top-1=30.21%, Top-5=90.62% | Evaluation Loss=2.1183, Top-1=30.82%, Top-5=91.47%

Epoch 6/20: Train Loss=2.0545, Top-1=32.42%, Top-5=91.53% | Evaluation Loss=1.9529, Top-1=34.68%, Top-5=93.01%

Epoch 7/20: Train Loss=1.9992, Top-1=34.06%, Top-5=92.03% | Evaluation Loss=1.9746, Top-1=34.73%, Top-5=92.00%

Epoch 8/20: Train Loss=1.9475, Top-1=35.10%, Top-5=92.53% | Evaluation Loss=1.8884, Top-1=36.40%, Top-5=93.02%

Epoch 9/20: Train Loss=1.8921, Top-1=36.73%, Top-5=93.14% | Evaluation Loss=1.8829, Top-1=37.41%, Top-5=92.95%

Epoch 10/20: Train Loss=1.8479, Top-1=37.78%, Top-5=93.57% | Evaluation Loss=1.8335, Top-1=37.81%, Top-5=93.68%

Epoch 11/20: Train Loss=1.8186, Top-1=38.83%, Top-5=93.78% | Evaluation Loss=1.8356, Top-1=38.33%, Top-5=92.90%

Epoch 12/20: Train Loss=1.7883, Top-1=39.37%, Top-5=93.91% | Evaluation Loss=1.7813, Top-1=40.16%, Top-5=93.81%

Epoch 13/20: Train Loss=1.7544, Top-1=40.63%, Top-5=94.14% | Evaluation Loss=1.7681, Top-1=39.86%, Top-5=94.15%

Epoch 14/20: Train Loss=1.7223, Top-1=41.68%, Top-5=94.59% | Evaluation Loss=1.7591, Top-1=40.43%, Top-5=94.12%

Epoch 15/20: Train Loss=1.7102, Top-1=42.25%, Top-5=94.61% | Evaluation Loss=1.7147, Top-1=41.83%, Top-5=94.63%

Epoch 16/20: Train Loss=1.6896, Top-1=42.75%, Top-5=94.74% | Evaluation Loss=1.7011, Top-1=42.00%, Top-5=94.80%

Epoch 17/20: Train Loss=1.6652, Top-1=43.40%, Top-5=95.00% | Evaluation Loss=1.7963, Top-1=39.49%, Top-5=94.15%

Epoch 18/20: Train Loss=1.6484, Top-1=44.03%, Top-5=94.96% | Evaluation Loss=1.7475, Top-1=40.66%, Top-5=93.85%

Epoch 19/20: Train Loss=1.6231, Top-1=44.78%, Top-5=95.28% | Evaluation Loss=1.7002, Top-1=42.18%, Top-5=94.16%

Epoch 20/20: Train Loss=1.6008, Top-1=45.39%, Top-5=95.31% | Evaluation Loss=1.6856, Top-1=43.49%, Top-5=94.51%

Reduced Learning rate from epoch 16.

Epoch 17/40: Train Loss=1.5311, Top-1=47.5%, Top-5=95.82% | Evaluation Loss=1.3605, Top-1=53.53%, Top-5=96.86%

Epoch 18/40: Train Loss=1.5182, Top-1=48.17%, Top-5=95.86% | Evaluation Loss=1.3565, Top-1=53.42%, Top-5=96.43%

Epoch 19/40: Train Loss=1.4850, Top-1=49.26%, Top-5=95.99% | Evaluation Loss=1.3579, Top-1=53.70%, Top-5=96.63%

Epoch 20/40: Train Loss=1.4637, Top-1=49.70%, Top-5=96.25% | Evaluation Loss=1.3067, Top-1=54.98%, Top-5=97.44%

![image](https://github.com/user-attachments/assets/f18be047-9476-490b-80fb-620d97738ada)
