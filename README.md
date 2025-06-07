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