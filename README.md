# Sound Source Localization

This project includes a series of Python scripts for binaural feature extraction, auditory modeling, and audio processing.

---

## Project Structure

### **1. `binaural_feature.py`**
Functions for extracting binaural audio features:
- `efficient_ccf`: Efficient cross-correlation computation
- `calculate_ild`: Compute Interaural Level Difference (ILD)
- `calculate_itd`: Compute Interaural Time Difference (ITD)
- `GetCues_clean`: Extract clean binaural cues

---

### **2. `auditory_model.py`**
Implements an auditory peripheral model:
- `filters`: Filtering functions
- `auditory_peripheral` & `haircell_model`: Peripheral auditory processing and hair cell modeling

**Configuration options**:
- `filter_type`: `GF_spectrogram` (Gammatone filterbank)
- `model_type`: `Roman, Half-wave rectification + signal envelope`

---

### **3. `gen_audio.py`**
Generates sine wave or impulse audio files.

**Example usage**:
```bash
python gen_audio.py sin --frequency 2000 --duration 1 --sample_rate 44100
```

### **4. ‘binauralize_mixaudio.py'**
Merge two mono audios and add angle to both audios.

**Example usage**:
```bash
python binauralize_mixaudio.py --file1 audio1.wav --file2 audio2.wav --location1 "90,0" --location2 "270,0"
```
