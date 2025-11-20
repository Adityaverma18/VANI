# src/utils.py
import os, json, random, numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    import json
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def set_seed(seed=42):
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def fit_label_encoder(labels, save_path=None):
    le = LabelEncoder()
    y_enc = le.fit_transform(labels)
    if save_path:
        joblib.dump(le, save_path)
    return le, y_enc

def load_label_encoder(path):
    return joblib.load(path)
