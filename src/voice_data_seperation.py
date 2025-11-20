import pandas as pd
import numpy as np

import os
import sys

# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later.
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import Audio

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import kagglehub

srad = kagglehub.dataset_download("vjcalling/speaker-recognition-audio-dataset",force_download=True)
srd = kagglehub.dataset_download("kongaevans/speaker-recognition-dataset")

speaker_folders = os.listdir(srad)

speaker_ids = []
paths = []

for folder in speaker_folders:
    folder_path = os.path.join(srad, folder)

    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        if file.lower().endswith(".wav"):
            paths.append(os.path.join(folder_path, file))
            speaker_ids.append(folder)     # speaker is folder name

df1 = pd.DataFrame({
    "Speaker": speaker_ids,
    "Path": paths
})


root = os.path.join(srd, "16000_pcm_speeches")   # important

speaker_folders = os.listdir(root)

speaker_ids2 = []
paths2 = []

for folder in speaker_folders:
    folder_path = os.path.join(root, folder)

    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        if file.lower().endswith(".wav"):
            paths2.append(os.path.join(folder_path, file))
            speaker_ids2.append(folder)     # speaker = folder name

df2 = pd.DataFrame({
    "Speaker": speaker_ids2,
    "Path": paths2
})

speaker_df = pd.concat([df1, df2], ignore_index=True)
speaker_df.to_csv("speaker_data.csv", index=False)

print(speaker_df.head())
print(speaker_df.Speaker.value_counts())
