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
Epoch 1/60: Train Loss=2.9714, Top-1=13.36%, Top-5=77.45% | Evaluation Loss=2.6607, Top-1=19.13%, Top-5=81.98%

Epoch 2/60: Train Loss=2.5272, Top-1=21.35%, Top-5=85.37% | Evaluation Loss=2.3689, Top-1=24.14%, Top-5=88.78%

Epoch 3/60: Train Loss=2.3454, Top-1=25.62%, Top-5=88.01% | Evaluation Loss=2.2165, Top-1=28.12%, Top-5=90.50%

Epoch 4/60: Train Loss=2.2260, Top-1=28.45%, Top-5=89.49% | Evaluation Loss=2.2727, Top-1=27.52%, Top-5=86.33%

Epoch 5/60: Train Loss=2.1367, Top-1=30.21%, Top-5=90.62% | Evaluation Loss=2.1183, Top-1=30.82%, Top-5=91.47%

Epoch 6/60: Train Loss=2.0545, Top-1=32.42%, Top-5=91.53% | Evaluation Loss=1.9529, Top-1=34.68%, Top-5=93.01%

Epoch 7/60: Train Loss=1.9992, Top-1=34.06%, Top-5=92.03% | Evaluation Loss=1.9746, Top-1=34.73%, Top-5=92.00%

Epoch 8/60: Train Loss=1.9475, Top-1=35.10%, Top-5=92.53% | Evaluation Loss=1.8884, Top-1=36.40%, Top-5=93.02%

Epoch 9/60: Train Loss=1.8921, Top-1=36.73%, Top-5=93.14% | Evaluation Loss=1.8829, Top-1=37.41%, Top-5=92.95%

Epoch 10/60: Train Loss=1.8479, Top-1=37.78%, Top-5=93.57% | Evaluation Loss=1.8335, Top-1=37.81%, Top-5=93.68%

Epoch 11/60: Train Loss=1.8186, Top-1=38.83%, Top-5=93.78% | Evaluation Loss=1.8356, Top-1=38.33%, Top-5=92.90%

Epoch 12/60: Train Loss=1.7883, Top-1=39.37%, Top-5=93.91% | Evaluation Loss=1.7813, Top-1=40.16%, Top-5=93.81%

Epoch 13/60: Train Loss=1.7544, Top-1=40.63%, Top-5=94.14% | Evaluation Loss=1.7681, Top-1=39.86%, Top-5=94.15%

Epoch 14/60: Train Loss=1.7223, Top-1=41.68%, Top-5=94.59% | Evaluation Loss=1.7591, Top-1=40.43%, Top-5=94.12%

Epoch 15/60: Train Loss=1.7102, Top-1=42.25%, Top-5=94.61% | Evaluation Loss=1.7147, Top-1=41.83%, Top-5=94.63%

Epoch 16/60: Train Loss=1.6896, Top-1=42.75%, Top-5=94.74% | Evaluation Loss=1.7011, Top-1=42.00%, Top-5=94.80%

Epoch 17/60: Train Loss=1.6652, Top-1=43.40%, Top-5=95.00% | Evaluation Loss=1.7963, Top-1=39.49%, Top-5=94.15%

Epoch 18/60: Train Loss=1.6484, Top-1=44.03%, Top-5=94.96% | Evaluation Loss=1.7475, Top-1=40.66%, Top-5=93.85%

Epoch 19/60: Train Loss=1.6231, Top-1=44.78%, Top-5=95.28% | Evaluation Loss=1.7002, Top-1=42.18%, Top-5=94.16%

Epoch 20/60: Train Loss=1.6008, Top-1=45.39%, Top-5=95.31% | Evaluation Loss=1.6856, Top-1=43.49%, Top-5=94.51%

Reduced Learning rate from epoch 16.

Epoch 17/60: Train Loss=1.5311, Top-1=47.5%, Top-5=95.82% | Evaluation Loss=1.3605, Top-1=53.53%, Top-5=96.86%

Epoch 18/60: Train Loss=1.5182, Top-1=48.17%, Top-5=95.86% | Evaluation Loss=1.3565, Top-1=53.42%, Top-5=96.43%

Epoch 19/60: Train Loss=1.4850, Top-1=49.26%, Top-5=95.99% | Evaluation Loss=1.3579, Top-1=53.70%, Top-5=96.63%

Epoch 20/60: Train Loss=1.4637, Top-1=49.70%, Top-5=96.25% | Evaluation Loss=1.3067, Top-1=54.98%, Top-5=97.44%

Epoch 21/60: Train Loss=1.4481, Top-1=50.16%, Top-5=96.25% | Evaluation Loss=1.3017, Top-1=55.71%, Top-5=97.28%

Epoch 22/60: Train Loss=1.4222, Top-1=50.83%, Top-5=96.45% | Evaluation Loss=1.2803, Top-1=55.91%, Top-5=96.93%

Epoch 23/60: Train Loss=1.4016, Top-1=51.89%, Top-5=96.61% | Evaluation Loss=1.2607, Top-1=57.26%, Top-5=97.39%

Epoch 24/60: Train Loss=1.3777, Top-1=52.34%, Top-5=96.71% | Evaluation Loss=1.2614, Top-1=56.80%, Top-5=97.20%

Epoch 25/60: Train Loss=1.3666, Top-1=53.08%, Top-5=96.68% | Evaluation Loss=1.2113, Top-1=58.67%, Top-5=97.76%

Epoch 26/60: Train Loss=1.3460, Top-1=53.64%, Top-5=96.89% | Evaluation Loss=1.2156, Top-1=58.19%, Top-5=97.69%

Epoch 27/60: Train Loss=1.3082, Top-1=55.10%, Top-5=97.16% | Evaluation Loss=1.2039, Top-1=58.96%, Top-5=97.60%

Epoch 28/60: Train Loss=1.2906, Top-1=55.19%, Top-5=97.15% | Evaluation Loss=1.1604, Top-1=60.77%, Top-5=97.82%

Epoch 29/60: Train Loss=1.2725, Top-1=56.10%, Top-5=97.26% | Evaluation Loss=1.1417, Top-1=61.13%, Top-5=97.91%

Epoch 30/60: Train Loss=1.2486, Top-1=57.01%, Top-5=97.30% | Evaluation Loss=1.1160, Top-1=61.83%, Top-5=97.89%

Epoch 31/60: Train Loss=1.2294, Top-1=57.53%, Top-5=97.48% | Evaluation Loss=1.1215, Top-1=61.91%, Top-5=98.03%

Epoch 32/60: Train Loss=1.2086, Top-1=58.40%, Top-5=97.52% | Evaluation Loss=1.1052, Top-1=62.23%, Top-5=98.05%

Epoch 33/60: Train Loss=1.1912, Top-1=58.82%, Top-5=97.70% | Evaluation Loss=1.0798, Top-1=63.39%, Top-5=98.11%

Epoch 34/60: Train Loss=1.1725, Top-1=59.46%, Top-5=97.76% | Evaluation Loss=1.0889, Top-1=62.93%, Top-5=98.12%

Epoch 35/60: Train Loss=1.1568, Top-1=60.13%, Top-5=97.78% | Evaluation Loss=1.0691, Top-1=63.31%, Top-5=98.17%

Epoch 36/60: Train Loss=1.1598, Top-1=60.16%, Top-5=97.79% | Evaluation Loss=1.0682, Top-1=63.85%, Top-5=98.09%

Epoch 37/60: Train Loss=1.1511, Top-1=60.31%, Top-5=97.80% | Evaluation Loss=1.0727, Top-1=63.53%, Top-5=98.20%

Epoch 38/60: Train Loss=1.1548, Top-1=60.21%, Top-5=97.86% | Evaluation Loss=1.0681, Top-1=63.71%, Top-5=98.24% 

Epoch 39/60: Train Loss=1.1518, Top-1=60.50%, Top-5=97.73% | Evaluation Loss=1.0667, Top-1=63.86%, Top-5=98.23%

Epoch 40/60: Train Loss=1.1549, Top-1=60.10%, Top-5=97.69% | Evaluation Loss=1.0677, Top-1=63.74%, Top-5=98.13%

Epoch 41/60: Train Loss=1.1726, Top-1=59.87%, Top-5=97.75% | Evaluation Loss=1.0712, Top-1=63.65%, Top-5=98.24%

Epoch 42/60: Train Loss=1.1597, Top-1=59.66%, Top-5=97.80% | Evaluation Loss=1.0729, Top-1=63.32%, Top-5=98.09%

Epoch 43/60: Train Loss=1.1685, Top-1=59.81%, Top-5=97.74% | Evaluation Loss=1.0709, Top-1=63.60%, Top-5=98.14%

Epoch 44/60: Train Loss=1.1621, Top-1=59.76%, Top-5=97.75% | Evaluation Loss=1.0879, Top-1=62.86%, Top-5=98.14%

Epoch 45/60: Train Loss=1.1601, Top-1=60.03%, Top-5=97.79% | Evaluation Loss=1.0779, Top-1=63.52%, Top-5=98.24%

Epoch 46/60: Train Loss=1.1513, Top-1=60.12%, Top-5=97.91% | Evaluation Loss=1.0571, Top-1=63.98%, Top-5=98.16%

Epoch 47/60: Train Loss=1.1488, Top-1=60.34%, Top-5=97.78% | Evaluation Loss=1.0718, Top-1=64.05%, Top-5=98.14%

Epoch 48/60: Train Loss=1.1489, Top-1=60.17%, Top-5=97.90% | Evaluation Loss=1.0554, Top-1=64.29%, Top-5=98.17%

Epoch 49/60: Train Loss=1.1389, Top-1=60.62%, Top-5=97.90% | Evaluation Loss=1.0599, Top-1=64.19%, Top-5=98.15%

Epoch 50/60: Train Loss=1.1431, Top-1=60.46%, Top-5=97.81% | Evaluation Loss=1.0493, Top-1=64.37%, Top-5=98.22%

Epoch 51/60: Train Loss=1.1361, Top-1=60.63%, Top-5=97.92% | Evaluation Loss=1.0436, Top-1=64.90%, Top-5=98.24%

Epoch 52/60: Train Loss=1.1374, Top-1=60.76%, Top-5=97.89% | Evaluation Loss=1.0473, Top-1=64.52%, Top-5=98.15%

Epoch 53/60: Train Loss=1.1289, Top-1=60.92%, Top-5=97.89% | Evaluation Loss=1.0386, Top-1=64.90%, Top-5=98.31%

Epoch 54/60: Train Loss=1.1317, Top-1=60.94%, Top-5=97.92% | Evaluation Loss=1.0511, Top-1=64.41%, Top-5=98.20%

Epoch 55/60: Train Loss=1.1246, Top-1=60.82%, Top-5=97.91% | Evaluation Loss=1.0446, Top-1=64.62%, Top-5=98.25%

Epoch 56/60: Train Loss=1.1249, Top-1=61.34%, Top-5=97.92% | Evaluation Loss=1.0426, Top-1=64.54%, Top-5=98.22%

Epoch 57/60: Train Loss=1.1229, Top-1=61.28%, Top-5=98.01% | Evaluation Loss=1.0432, Top-1=64.74%, Top-5=98.20%

Epoch 58/60: Train Loss=1.1252, Top-1=61.19%, Top-5=97.91% | Evaluation Loss=1.0424, Top-1=64.96%, Top-5=98.20%

Epoch 59/60: Train Loss=1.1210, Top-1=61.18%, Top-5=97.95% | Evaluation Loss=1.0479, Top-1=64.12%, Top-5=98.26%

Epoch 60/60: Train Loss=1.1188, Top-1=61.42%, Top-5=97.87% | Evaluation Loss=1.0610, Top-1=64.49%, Top-5=98.20%

![image](https://github.com/user-attachments/assets/f18be047-9476-490b-80fb-620d97738ada)
