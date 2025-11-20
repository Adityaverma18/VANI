import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.utils import to_categorical
import math
import joblib

from src.features import compute_max_len, build_features
from src.model_speaker import build_speaker_model
from src.utils import set_seed, save_json

def cosine_annealing(epoch, lr, epochs=80):
    lr_min = 1e-6
    lr_max = 1e-3
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(epoch * math.pi / epochs))

def main(args):
    set_seed(42)

    df = pd.read_csv(args.data_csv)
    df = df[df['Path'].notna()].reset_index(drop=True)
    labels = df['Speaker'].values
    paths = df['Path'].values

    # label encode
    le = LabelEncoder()
    y_enc = le.fit_transform(labels)
    joblib.dump(le, os.path.join(args.out_dir, "speaker_label_encoder.joblib"))
    save_json(list(le.classes_), os.path.join(args.out_dir, "speakers.json"))
    n_speakers = len(le.classes_)

    # compute/choose max_len
    if args.max_len is None or args.max_len <= 0:
        print("[INFO] Estimating max_len from data (sample_frac=0.25)...")
        max_len = compute_max_len(paths, sample_frac=0.25)
        if max_len <= 0:
            max_len = 130
    else:
        max_len = args.max_len
    print("[INFO] max_len =", max_len)

    # build features
    os.makedirs(args.out_dir, exist_ok=True)
    cache_path = os.path.join(args.out_dir, "speaker_features.npz")
    X = build_features(paths, max_len, cache_path=cache_path)
    y = to_categorical(y_enc, num_classes=n_speakers)

    # split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.test_size, stratify=y_enc, random_state=42)
    print("Train:", X_train.shape, "Val:", X_val.shape)

    # class weights
    cw = class_weight.compute_class_weight("balanced", classes=np.unique(y_enc), y=y_enc)
    class_weights = dict(enumerate(cw))
    print("Class weights:", class_weights)

    # build model
    n_features = X.shape[2]
    model = build_speaker_model(time_steps=max_len, features=n_features, n_speakers=n_speakers, lr=args.lr)
    model.summary()

    # callbacks
    ckpt = ModelCheckpoint(os.path.join(args.out_dir, "speaker_model_best.keras"), monitor="val_accuracy", save_best_only=True, verbose=1)
    early = EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1)
    cosine = LearningRateScheduler(lambda epoch, lr: cosine_annealing(epoch, lr, epochs=args.epochs))

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=[ckpt, early, cosine]
    )

    model.save(os.path.join(args.out_dir, "speaker_model_final.keras"))
    print("Saved speaker model to", args.out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", default="speaker_data.csv")
    parser.add_argument("--out_dir", default="../models")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_len", type=int, default=None)
    args = parser.parse_args()
    main(args)
    
#python main.py --max_len 130 --emotion_model "models/emotion_model_final.keras" --speaker_model "models/speaker_model_best.keras"
