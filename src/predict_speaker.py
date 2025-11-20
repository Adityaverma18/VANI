import argparse, os, json, numpy as np
from keras.models import load_model
from src.features import extract_features, pad_or_truncate
import joblib

def predict_one(path, model, speakers, max_len):
    feat = extract_features(path)
    x = pad_or_truncate(feat, max_len)
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)[0]
    idx = int(pred.argmax())
    label = speakers[idx]
    conf = float(pred[idx])
    return label, conf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--model", default="models/speaker_model_best.keras")
    parser.add_argument("--speakers", default="models/speakers.json")
    parser.add_argument("--max_len", type=int, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print("Audio file not found:", args.file); exit(1)
    if not os.path.exists(args.model):
        print("Model not found:", args.model); exit(1)
    if not os.path.exists(args.speakers):
        print("Speakers file not found:", args.speakers); exit(1)

    model = load_model(args.model)
    with open(args.speakers, "r") as f:
        speakers = json.load(f)

    label, conf = predict_one(args.file, model, speakers, args.max_len)
    print(f"Speaker: {label} (confidence {conf:.3f})")