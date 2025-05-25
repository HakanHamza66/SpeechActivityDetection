import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path

# Prediction CSV klasörü
PREDICTIONS_DIR = "data/predictions"

# .csv prediction dosyalarını al
pred_files = list(Path(PREDICTIONS_DIR).rglob("*_pred.csv"))
assert pred_files, "Prediction CSV dosyası bulunamadı!"

# Rastgele bir tanesini seç
sample_csv = random.choice(pred_files)
df = pd.read_csv(sample_csv)

# Prediction vektörü
predictions = df['prediction'].valuesgit
frame_axis = np.arange(len(predictions))

# Görselleştirme
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
