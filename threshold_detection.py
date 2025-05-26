import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import random
from pathlib import Path
NOISE_DIR = "data/noise"
SPEECH_DIR = "data/clean_wav"
SAMPLE_RATE = 16000
FRAME_SIZE = int(0.025 * SAMPLE_RATE)
HOP_LENGTH = int(0.010 * SAMPLE_RATE)
N_FFT = 512
NUM_SAMPLES = 100

def compute_average_log_energy(wav_path):
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    stft = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=FRAME_SIZE))**2
    log_energy = np.log(np.sum(stft, axis=0) + 1e-10)
    return np.mean(log_energy)

def plot_energy_comparison(noise_energies, speech_energies, threshold):
    plt.figure(figsize=(10, 5))
    plt.plot(noise_energies, marker='o', color='royalblue', label='Noise Samples')
    plt.plot(speech_energies, marker='x', color='seagreen', label='Speech Samples')
    plt.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f"Threshold = {threshold:.2f}")
    plt.title("Avg Log Energy of Noise vs Speech Samples")
    plt.xlabel("Sample Index")
    plt.ylabel("Average Log Energy")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def main():
    noise_files = list(Path(NOISE_DIR).rglob("*.wav"))
    noise_sample_files = random.sample(noise_files, NUM_SAMPLES)
    noise_energies = [compute_average_log_energy(str(f)) for f in noise_sample_files]
    speech_files = list(Path(SPEECH_DIR).rglob("*.wav"))
    speech_sample_files = random.sample(speech_files, NUM_SAMPLES)
    speech_energies = [compute_average_log_energy(str(f)) for f in speech_sample_files]
    threshold = np.mean(noise_energies) + np.std(noise_energies)

    with open("adaptive_threshold.txt", "w") as f:
        f.write(f"{threshold:.4f}")

    print(f"Adaptive Threshold: {threshold:.4f}")
    plot_energy_comparison(noise_energies, speech_energies, threshold)
