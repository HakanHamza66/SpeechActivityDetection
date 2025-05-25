import os
import random
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

SPEECH_DIR = "data/clean_wav"
NOISE_DIR = "data/noise"
OUTPUT_DIR = "data/mixed"
SNR_dB = 5  # you can loop over [0, 5, 10] if needed


def get_all_wav_files(directory):
    return list(Path(directory).rglob("*.wav"))


def mix_signals(speech, noise, snr_db):
    if len(noise) < len(speech):
        noise = np.tile(noise, int(np.ceil(len(speech) / len(noise))))
    noise = noise[:len(speech)]

    speech_power = np.sum(speech ** 2)
    noise_power = np.sum(noise ** 2)
    snr_linear = 10 ** (snr_db / 10)
    scale = np.sqrt(speech_power / (noise_power * snr_linear))
    noise_scaled = noise * scale
    return speech + noise_scaled


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    speech_files = get_all_wav_files(SPEECH_DIR)
    noise_files = get_all_wav_files(NOISE_DIR)

    print(f"Found {len(speech_files)} speech files, {len(noise_files)} noise files.")

    for speech_path in tqdm(speech_files):
        speech, sr_s = sf.read(speech_path)
        noise_path = random.choice(noise_files)
        noise, sr_n = sf.read(noise_path)

        if sr_s != 16000 or sr_n != 16000:
            print(f"Skipping {speech_path} or {noise_path} due to sample rate.")
            continue

        mixed = mix_signals(speech, noise, SNR_dB)
        out_filename = f"mixed_{Path(speech_path).stem}.wav"
        out_path = os.path.join(OUTPUT_DIR, out_filename)
        sf.write(out_path, mixed, 16000)


if __name__ == "__main__":
    main()
