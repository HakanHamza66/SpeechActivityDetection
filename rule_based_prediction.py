import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

FEATURES_DIR = "data/features"
PREDICTIONS_DIR = "data/predictions"
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

with open("adaptive_threshold.txt", "r") as f:
    ENERGY_THRESHOLD = float(f.read().strip())

def rule_based_sad(features_csv, threshold=ENERGY_THRESHOLD):
    df = pd.read_csv(features_csv)
    log_energy = df['log_energy'].values
    predictions = (log_energy > threshold).astype(int)
    return predictions

def main():
    feature_files = list(Path(FEATURES_DIR).rglob("*.csv"))
    for file_path in tqdm(feature_files, desc="Running rule-based SAD"):
        preds = rule_based_sad(str(file_path))
        out_name = Path(file_path).stem + "_pred.csv"
        out_path = os.path.join(PREDICTIONS_DIR, out_name)
        pd.DataFrame(preds, columns=["prediction"]).to_csv(out_path, index=False)
