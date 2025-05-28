import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path

PREDICTIONS_DIR = "data/predictions"

def plot_random_prediction():
    pred_files = list(Path(PREDICTIONS_DIR).rglob("*_pred.csv"))
    sample_csv = random.choice(pred_files)
    df = pd.read_csv(sample_csv)
    predictions = df['prediction'].values
    frame_axis = np.arange(len(predictions))

    plt.figure(figsize=(12, 3))
    plt.step(frame_axis, predictions, where='mid', color='crimson', linewidth=1.5)
    plt.title(f"Rule-Based SAD Prediction (from CSV)\n{sample_csv.stem}")
    plt.xlabel("Frame Index")
    plt.ylabel("Prediction")
    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 1], ['Silence', 'Speech'])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

def main():
    plot_random_prediction()