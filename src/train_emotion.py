import os
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.utils import to_categorical

from src.features import compute_max_len, build_features
from src.model_emotion import build_emotion_model
from src.utils import set_seed, fit_label_encoder, save_json

import math
import joblib


# ======================================
# COSINE LEARNING RATE SCHEDULER
# ======================================
def cosine_annealing(epoch, lr):
    lr_min = 1e-6
    lr_max = 1e-3
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(epoch * math.pi / 80))


def main(args):
    set_seed(42)

    # ----------------------------------
    # LOAD DATA
    # ----------------------------------
    df = pd.read_csv(args.data_csv)
    df = df[df['Emotions'].notna()].reset_index(drop=True)
    print("Loaded dataset:", df.shape)

    labels = df['Emotions'].values
    paths = df['Path'].values

    # ----------------------------------
    # LABEL ENCODING
    # ----------------------------------
    le, y_enc = fit_label_encoder(
        labels,
        save_path=os.path.join(args.out_dir, "label_encoder.joblib")
    )

    classes = list(le.classes_)
    save_json(classes, os.path.join(args.out_dir, "classes.json"))
    print("Classes:", classes)

    # ----------------------------------
    # COMPUTE max_len
    # ----------------------------------
    if args.max_len is None or args.max_len <= 0:
        print("Estimating max_len (30% sampling)...")
        max_len = compute_max_len(paths, sample_frac=0.3)
        if max_len < 30:
            max_len = 130
        print("Estimated max_len =", max_len)
    else:
        max_len = args.max_len

    # ----------------------------------
    # FEATURE EXTRACTION
    # ----------------------------------
    cache_path = os.path.join(args.out_dir, "features.npz")
    print("Extracting features...")
    X = build_features(paths, max_len, cache_path=cache_path)
    y = to_categorical(y_enc, num_classes=len(classes))

    # SPLIT
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, stratify=y_enc, random_state=42
    )

    print("Train Shape:", X_train.shape)
    print("Val Shape:", X_val.shape)

    # ----------------------------------
    # CLASS WEIGHTS  ⭐ IMPORTANT
    # ----------------------------------
    cw = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_enc),
        y=y_enc
    )
    class_weights = dict(enumerate(cw))
    print("Class Weights:", class_weights)

    # ----------------------------------
    # BUILD MODEL ⭐ IMPROVED ARCHITECTURE
    # ----------------------------------
    n_features = X.shape[2]  # 168 (128 mel + 40 mfcc)

    model = build_emotion_model(
        time_steps=max_len,
        features=n_features,
        classes=len(classes),
        lr=args.lr
    )

    model.summary()

    # ----------------------------------
    # CALLBACKS ⭐ ADD COSINE SCHEDULER
    # ----------------------------------
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = ModelCheckpoint(
        os.path.join(args.out_dir, "emotion_model_best.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    early = EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
        verbose=1
    )

    cosine = LearningRateScheduler(cosine_annealing)

    # ----------------------------------
    # TRAIN ⭐ NOW WITH CLASS WEIGHTS + COSINE
    # ----------------------------------
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,     # ⭐ IMPORTANT
        callbacks=[ckpt, early, cosine] # ⭐ NO ReduceLROnPlateau needed
    )

    # ----------------------------------
    # SAVE FINAL MODEL
    # ----------------------------------
    model.save(os.path.join(args.out_dir, "emotion_model_final.keras"))
    print("Training complete. Model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_csv", default="data_path.csv")
    parser.add_argument("--out_dir", default="models")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_len", type=int, default=None)

    args = parser.parse_args()
    main(args)
