# src/features.py
import numpy as np
import librosa
import warnings

warnings.filterwarnings("ignore")

# ================================
# GLOBAL CONFIG
# ================================
SR = 22050
DURATION = 3
SAMPLES = SR * DURATION

N_MFCC = 40          # MFCC count
N_MEL = 128          # mel bands
HYBRID_FEATURES = N_MEL + N_MFCC   # 168 total features

AUGMENT = True       # enable/disable augmentation


# ================================
# AUDIO LOADING
# ================================
def load_audio(path, sr=SR):
    """Loads audio and pads/truncates to exact 3 seconds."""
    try:
        y, _ = librosa.load(path, sr=sr)
    except Exception:
        print(f"[WARN] Could not load file: {path}")
        return np.zeros(SAMPLES)

    if len(y) < SAMPLES:
        y = np.pad(y, (0, SAMPLES - len(y)))
    else:
        y = y[:SAMPLES]
    return y


# ================================
# OPTIONAL AUGMENTATION
# ================================
def augment(y, sr):
    """Random audio augmentation for higher accuracy."""
    if not AUGMENT:
        return y

    # pitch shift
    if np.random.rand() < 0.3:
        try:
            y = librosa.effects.pitch_shift(y, sr, n_steps=np.random.uniform(-2, 2))
        except:
            pass

    # time stretch
    if np.random.rand() < 0.3:
        rate = np.random.uniform(0.85, 1.15)
        y = librosa.effects.time_stretch(y=y, rate= rate)

        # After time-stretch, fix length
        if len(y) < SAMPLES:
            y = np.pad(y, (0, SAMPLES - len(y)))
        else:
            y = y[:SAMPLES]

    # random noise
    if np.random.rand() < 0.3:
        y = y + 0.005 * np.random.randn(len(y))

    return y


# ================================
# FEATURE EXTRACTION (MFCC + MEL)
# ================================
def extract_features(path):
    """Extract HYBRID FEATURES: mel-spectrogram + MFCC."""
    y = load_audio(path)

    # apply augmentations only during training
    if AUGMENT:
        y = augment(y, SR)

    # mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MEL, fmax=8000
    )
    mel = librosa.power_to_db(mel).T  # (T1, 128)

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=y, sr=SR, n_mfcc=N_MFCC
    ).T  # (T2, 40)

    # match time frames
    T = min(len(mel), len(mfcc))
    mel = mel[:T]
    mfcc = mfcc[:T]

    # concat → (T, 168)
    return np.hstack([mel, mfcc]).astype(np.float32)


# ================================
# FIXED LENGTH FEATURES
# ================================
def pad_or_truncate(feat, max_len):
    if feat.shape[0] >= max_len:
        return feat[:max_len]
    else:
        pad = max_len - feat.shape[0]
        return np.pad(feat, ((0, pad), (0, 0)), mode="constant")


# ================================
# COMPUTE MAX LENGTH
# ================================
def compute_max_len(paths, sample_frac=0.25):
    """Calculate best max_len based on sample files."""
    import random

    sample_paths = list(paths)
    random.shuffle(sample_paths)

    sample_paths = sample_paths[:int(len(sample_paths) * sample_frac)]

    max_len = 0
    print(f"[INFO] Computing max_len from {len(sample_paths)} samples...")

    for p in sample_paths:
        try:
            f = extract_features(p)
            max_len = max(max_len, f.shape[0])
        except:
            continue

    print("[INFO] Best max_len =", max_len)
    return max_len


# ================================
# BUILD FEATURE MATRIX
# ================================
def build_features(paths, max_len, cache_path=None):
    """Build final dataset array (N, max_len, HYBRID_FEATURES)."""
    X = np.zeros((len(paths), max_len, HYBRID_FEATURES), dtype=np.float32)

    for i, p in enumerate(paths):
        feat = extract_features(p)
        X[i] = pad_or_truncate(feat, max_len)

        if i % 500 == 0:
            print(f"[INFO] Processed {i}/{len(paths)} files")

    if cache_path:
        np.savez_compressed(cache_path, X=X)
        print(f"[INFO] Saved cached features: {cache_path}")

    return X


# ================================
# LOAD CACHED NPZ
# ================================
def load_cached(cache_path):
    data = np.load(cache_path)
    return data['X']
