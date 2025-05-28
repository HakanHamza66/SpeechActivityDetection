import os
import cv2
import numpy as np
import librosa
import soundfile as sf
import moviepy.editor as mp
import pandas as pd
from pathlib import Path
from extract_features import extract_features_from_file
from rule_based_prediction import rule_based_sad
from moviepy.editor import VideoFileClip, AudioFileClip
# Sabitler
VIDEO_PATH = "data/video/murat_saraclar.mp4"
AUDIO_PATH = "data/video/audio_extracted.wav"
FEATURE_CSV = "data/video/audio_features.csv"
PREDICTION_CSV = "data/video/audio_prediction.csv"
OUTPUT_VIDEO = "data/video/video_with_sad_overlay.mp4"
SAMPLE_RATE = 16000
HOP_LENGTH = int(0.010 * SAMPLE_RATE)

def extract_audio_from_video(video_path, audio_out_path):
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_out_path, fps=SAMPLE_RATE, verbose=False, logger=None)

def extract_features_and_predict(audio_path, features_csv, prediction_csv):
    df, _, _, _ = extract_features_from_file(audio_path)
    df.to_csv(features_csv, index=False)

    preds = rule_based_sad(features_csv)
    pd.DataFrame(preds, columns=["prediction"]).to_csv(prediction_csv, index=False)

def overlay_predictions_on_video(video_path, prediction_csv, output_path):
    df = pd.read_csv(prediction_csv)
    predictions = df["prediction"].values

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_id = 0
    success, frame = cap.read()
    while success:
        time_sec = frame_id / fps
        pred_index = int((time_sec * SAMPLE_RATE) / HOP_LENGTH)
        if pred_index < len(predictions):
            label = "Speech" if predictions[pred_index] == 1 else "Silence"
            color = (0, 255, 0) if label == "Speech" else (0, 0, 255)
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        out.write(frame)
        success, frame = cap.read()
        frame_id += 1

    cap.release()
    out.release()
    print(f"✅ Annotated video saved at: {output_path}")
def merge_audio_with_video(video_path, audio_path, output_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
def run_video_sad():
    extract_audio_from_video(VIDEO_PATH, AUDIO_PATH)
    extract_features_and_predict(AUDIO_PATH, FEATURE_CSV, PREDICTION_CSV)
    overlay_predictions_on_video(VIDEO_PATH, PREDICTION_CSV, OUTPUT_VIDEO)
    merge_audio_with_video(OUTPUT_VIDEO, AUDIO_PATH, "data/video/video_with_sad_and_audio.mp4")

