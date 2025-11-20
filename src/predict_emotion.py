import argparse, os, json
import numpy as np
from keras.models import load_model
from src.features import extract_features, pad_or_truncate
from src.utils import load_json


def predict_one(path, model, classes, max_len):
    mfcc = extract_features(path)
    x = pad_or_truncate(mfcc, max_len)
    
    x = np.expand_dims(x, axis=0)  # (1, max_len, n_mfcc)
    pred = model.predict(x)[0]
    idx = int(pred.argmax())
    label = classes[idx]
    conf = float(pred[idx])
    return label, conf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True)
    parser.add_argument('--model', default='../models/emotion_model_best.h5')
    parser.add_argument('--classes', default='../models/classes.json')
    parser.add_argument('--max_len', type=int, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print("Audio file not found:", args.file); exit(1)
    if not os.path.exists(args.model):
        print("Model not found:", args.model); exit(1)
    if not os.path.exists(args.classes):
        print("Classes file not found:", args.classes); exit(1)

    model = load_model(args.model)
    with open(args.classes, 'r') as f:
        classes = json.load(f)

    label, conf = predict_one(args.file, model, classes, args.max_len)
    print(f"Prediction: {label} (confidence {conf:.3f})")