import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import random
from pathlib import Path
from tqdm import tqdm

SAMPLE_RATE = 16000
FRAME_SIZE = int(0.025 * SAMPLE_RATE)
HOP_LENGTH = int(0.010 * SAMPLE_RATE)
N_FFT = 512
N_MFCC = 13

INPUT_DIR = "data/mixed"
FEATURES_DIR = "data/features"
os.makedirs(FEATURES_DIR, exist_ok=True)

def extract_features_from_file(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    stft = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=FRAME_SIZE))**2
    log_energy = np.log(np.sum(stft, axis=0) + 1e-10)
    centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(stft), sr=sr, n_mfcc=N_MFCC)

    features = np.vstack([log_energy, centroid, mfccs])
    feature_names = ['log_energy', 'spectral_centroid'] + [f'mfcc_{i+1}' for i in range(N_MFCC)]
    df = pd.DataFrame(features.T, columns=feature_names)
    return df, y, sr, stft

def main():
    mixed_files = list(Path(INPUT_DIR).rglob("*.wav"))
    for file_path in tqdm(mixed_files, desc="Extracting features"):
        df, _, _, _ = extract_features_from_file(str(file_path))
        out_csv = Path(FEATURES_DIR) / (file_path.stem + ".csv")
        df.to_csv(out_csv, index=False)