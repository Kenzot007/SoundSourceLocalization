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

