import numpy as np
np.complex = complex
import os
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
import random

SAMPLE_RATE = 16000
FRAME_SIZE = int(0.025 * SAMPLE_RATE)
HOP_LENGTH = int(0.010 * SAMPLE_RATE)
N_FFT = 512

FEATURES_DIR = "data/features"
PREDICTIONS_DIR = "data/predictions"
MIXED_AUDIO_DIR = "data/mixed"

def plot_spectrogram_with_sad_overlay():
    feature_files = list(Path(FEATURES_DIR).rglob("*.csv"))
    pred_files = list(Path(PREDICTIONS_DIR).rglob("*_pred.csv"))

    common_stems = set(Path(f).stem.replace("_pred", "") for f in pred_files) & \
                   set(Path(f).stem for f in feature_files)

    sample_stem = random.choice(list(common_stems))

    wav_path = Path(MIXED_AUDIO_DIR) / f"{sample_stem}.wav"
    csv_path = Path(PREDICTIONS_DIR) / f"{sample_stem}_pred.csv"

    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    pred = pd.read_csv(csv_path)['prediction'].values

    stft = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=FRAME_SIZE))**2
    db_stft = librosa.power_to_db(stft, ref=np.max)

    plt.figure(figsize=(12, 6))
    librosa.display.specshow(db_stft, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spectrogram + SAD Overlay: {sample_stem}")

    for i, val in enumerate(pred):
        color = 'deepskyblue' if val == 1 else 'lightgray'
        plt.axvspan(i * HOP_LENGTH / SAMPLE_RATE,
                    (i + 1) * HOP_LENGTH / SAMPLE_RATE,
                    color=color, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return y,sr
def plot_cepstrum_analysis(y, sr):
    frame = y[0:FRAME_SIZE] * np.hamming(FRAME_SIZE)
    spectrum = np.fft.fft(frame, n=N_FFT)
    log_spectrum = np.log(np.abs(spectrum) + 1e-10)
    cepstrum = np.abs(np.fft.ifft(log_spectrum))

    quefrency_axis = np.arange(len(cepstrum)) / sr

    plt.figure(figsize=(10, 4))
    plt.plot(quefrency_axis * 1000, cepstrum, color='darkorange', label="Cepstrum Magnitude")
    plt.axvspan(0, 1.5, color='lightblue', alpha=0.3, label="Vocal Tract Region")
    plt.axvspan(1.5, 15, color='lightgreen', alpha=0.3, label="Excitation Region")
    plt.title("Cepstrum of Mixed Signal (First Frame)")
    plt.xlabel("Quefrency (ms)")
    plt.ylabel("Amplitude")
    plt.xlim(0, 20)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def main():
    y,sr=plot_spectrogram_with_sad_overlay()
    plot_cepstrum_analysis(y, sr)