from convert_and_reorganize import convert_and_group, find_all_flac_files
from mix_signals import main as mix_main
from extract_features import main as extract_features_main
from threshold_detection import main as threshold_main
from rule_based_prediction import main as prediction_main
from visualization import main as visualization_main
from spectrogram_overlay import main as sad_spectrogram_main
from video_sad_overlay import run_video_sad
def main():
    print("1. Files converting and reorganizing...")
    flacs = find_all_flac_files("data/speech")
    convert_and_group(flacs)

    print("2. Mixed signal generating...")
    mix_main()

    print("3. STFT based feature extraction...")
    extract_features_main()

    print("4. Adaptive threshold (mean + std) calculating...")
    threshold_main()

    print("5. Rule-based SAD predicting...")
    prediction_main()

    print("6. Random sample visualization ...")
    visualization_main()

    print("7. Video SAD")
    run_video_sad()
if __name__ == "__main__":
    main()
