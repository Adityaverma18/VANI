import argparse
import os
import time
import json
import threading
import tempfile
import numpy as np

import sounddevice as sd
import soundfile as sf
from keras.models import load_model

from src.features import extract_features, pad_or_truncate, SR


# ------------------------
# Utility functions
# ------------------------

def record_audio(duration, sr, device=None):
    """Record audio from microphone."""
    print(f"[REC] Recording {duration}s...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32", device=device)
    sd.wait()
    return audio.flatten()


def save_wav(path, data, sr):
    sf.write(path, data, sr)
    return path


def energy_vad(y, threshold=0.0006):
    """Simple energy-based Voice Activity Detection (skip silence)."""
    rms = np.sqrt(np.mean(y * y))
    return rms > threshold


def predict_emotion(model, classes, audio_path, max_len):
    """Predict emotion for the recorded audio."""
    feat = extract_features(audio_path)
    x = pad_or_truncate(feat, max_len)
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x, verbose=0)[0]
    idx = int(pred.argmax())
    return classes[idx], float(pred[idx])


def predict_speaker(model, speakers, audio_path, max_len):
    """Predict speaker ID."""
    feat = extract_features(audio_path)
    x = pad_or_truncate(feat, max_len)
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x, verbose=0)[0]
    idx = int(pred.argmax())
    return speakers[idx], float(pred[idx])


def assistive_message(emotion_label, speaker_label):
    msg = []

    # Emotion message
    if emotion_label:
        emo, conf = emotion_label
        msg.append(f"Emotion: {emo} ({conf:.2f})")

        if emo in ("angry", "fear", "sad", "disgust"):
            msg.append("Try relaxing. Deep breaths can help.")
        elif emo in ("happy", "surprise"):
            msg.append("Positive! Keep going.")

    # Speaker message
    if speaker_label:
        sp, conf = speaker_label
        msg.append(f"Speaker: {sp} ({conf:.2f})")

    return " | ".join(msg)


# ------------------------
# Main Loop
# ------------------------

def main(args):

    # Load emotion model
    print("[LOAD] Loading emotion model...")
    emotion_model = load_model(args.emotion_model)
    with open(args.emotion_classes, "r") as f:
        emotion_classes = json.load(f)

    # Load speaker model
    print("[LOAD] Loading speaker model...")
    speaker_model = load_model(args.speaker_model)
    with open(args.speaker_classes, "r") as f:
        speaker_classes = json.load(f)

    print("\n[READY] Real-time voice analysis started.")
    print("Press CTRL+C to stop.\n")

    try:
        while True:

            # 1. Record audio
            y = record_audio(args.duration, args.sample_rate, device=args.device)

            # 2. Apply simple VAD
            if not energy_vad(y, threshold=args.vad_threshold):
                print(f"[{time.strftime('%H:%M:%S')}] Silence skipped.\n")
                continue

            # 3. Save temp wav
            tmpf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmpf.name
            tmpf.close()
            save_wav(tmp_path, y, args.sample_rate)

            # 4. Run predictions in parallel
            out = {}

            def emo_thread():
                out["emotion"] = predict_emotion(
                    emotion_model, emotion_classes, tmp_path, args.max_len
                )

            def spk_thread():
                out["speaker"] = predict_speaker(
                    speaker_model, speaker_classes, tmp_path, args.max_len
                )

            t1 = threading.Thread(target=emo_thread)
            t2 = threading.Thread(target=spk_thread)

            t1.start()
            t2.start()
            t1.join()
            t2.join()

            # 5. Display output
            emo = out.get("emotion")
            spk = out.get("speaker")

            print(f"[{time.strftime('%H:%M:%S')}] {assistive_message(emo, spk)}\n")

            # remove temp file
            try:
                os.remove(tmp_path)
            except:
                pass

            time.sleep(args.pause)

    except KeyboardInterrupt:
        print("\n[STOPPED] User ended real-time voice detection.")
    except Exception as e:
        print("[ERROR]", str(e))


# ------------------------
# CLI
# ------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Emotion + Speaker Detection")

    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--pause", type=float, default=0.3)
    parser.add_argument("--sample_rate", type=int, default=SR)
    parser.add_argument("--max_len", type=int, required=True)

    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--vad_threshold", type=float, default=0.0006)

    parser.add_argument("--emotion_model", default="models/emotion_model_best.keras")
    parser.add_argument("--emotion_classes", default="models/classes.json")

    parser.add_argument("--speaker_model", default="models/speaker_model_best.keras")
    parser.add_argument("--speaker_classes", default="models/speakers.json")

    args = parser.parse_args()
    main(args)
