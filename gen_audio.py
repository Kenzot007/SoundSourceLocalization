import numpy as np
import soundfile as sf
import scipy.io.wavfile as wav
import argparse

# generate sin wave
def generate_sine_wave(frequency, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    return audio

def generate_impulse(samples):
    impulse = np.zeros(samples, dtype=np.float32)
    impulse[0] = 1
    return impulse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test audio signals.")

    parser.add_argument("wave_type", type=str, choices=["sin", "impulse"], help="choose audio type, sin or impulse...")
    parser.add_argument("--frequency", type=float, default=440, help="Hz, for sin")
    parser.add_argument("--duration", type=float, default=1.0, help="duration")
    parser.add_argument("--sample_rate", type=int, default=44100, help="sample rate")

    args = parser.parse_args()

    if args.wave_type == "sin":
        audio_data = generate_sine_wave(args.frequency, args.duration, args.sample_rate)
        output_file = f"sin_{int(args.frequency)}Hz.wav"
    elif args.wave_type == "impulse":
        audio_data = generate_impulse(int(args.sample_rate * args.duration))
        output_file = "impulse.wav"

    sf.write(output_file, audio_data, args.sample_rate)
    print(f"Generated: {output_file}")

