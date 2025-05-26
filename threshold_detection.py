import os
import numpy as np
import librosa
import random
from pathlib import Path

NOISE_DIR = "data/noise"
SAMPLE_RATE = 16000
FRAME_SIZE = int(0.025 * SAMPLE_RATE)
HOP_LENGTH = int(0.010 * SAMPLE_RATE)
N_FFT = 512
NUM_NOISE_SAMPLES = 100

def compute_average_log_energy(wav_path):
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    stft = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=FRAME_SIZE))**2
    log_energy = np.log(np.sum(stft, axis=0) + 1e-10)
    return np.mean(log_energy)

def main():
    noise_files = list(Path(NOISE_DIR).rglob("*.wav"))
    selected_files = random.sample(noise_files, NUM_NOISE_SAMPLES)
    log_energies = [compute_average_log_energy(str(file)) for file in selected_files]

    adaptive_threshold = np.mean(log_energies) + np.std(log_energies)
    with open("adaptive_threshold.txt", "w") as f:
        f.write(f"{adaptive_threshold:.4f}")
