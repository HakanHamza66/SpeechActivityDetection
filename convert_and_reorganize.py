import os
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

INPUT_DIR = "data/speech"
OUTPUT_DIR = "data/clean_wav"
TARGET_SR = 16000
FILES_PER_FOLDER = 20

def find_all_flac_files(root):
    return list(Path(root).rglob("*.flac"))

def convert_and_group(flac_files):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    folder_count = 1
    file_count = 0
    current_output_folder = os.path.join(OUTPUT_DIR, str(folder_count))
    os.makedirs(current_output_folder, exist_ok=True)

    for flac_file in tqdm(flac_files, desc="Converting FLAC to WAV"):
        data, sr = sf.read(flac_file)
        if file_count == FILES_PER_FOLDER:
            folder_count += 1
            file_count = 0
            current_output_folder = os.path.join(OUTPUT_DIR, str(folder_count))
            os.makedirs(current_output_folder, exist_ok=True)

        out_filename = f"{folder_count}_{file_count + 1}.wav"
        out_path = os.path.join(current_output_folder, out_filename)
        sf.write(out_path, data, TARGET_SR)
        file_count += 1

def main():
    all_flacs = find_all_flac_files(INPUT_DIR)
    print(f"Found {len(all_flacs)} FLAC files.")
    convert_and_group(all_flacs)
