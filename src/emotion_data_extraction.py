import pandas as pd
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

ravdess = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
ravdess_directory_list = os.listdir(ravdess)

Crema = kagglehub.dataset_download("ejlok1/cremad")
Tess = kagglehub.dataset_download("ejlok1/toronto-emotional-speech-set-tess")
Savee = kagglehub.dataset_download("ejlok1/surrey-audiovisual-expressed-emotion-savee")


# Ravdess dataset
file_emotion = []
file_path = []
for folder in ravdess_directory_list:
    # Skip non-actor folders like: audio_speech_actors_01-24
    if not folder.startswith("Actor_"):
        continue

    actor_folder = os.path.join(ravdess, folder)
    if not os.path.isdir(actor_folder):
        continue

    for f in os.listdir(actor_folder):
        full_path = os.path.join(actor_folder, f)

        # Skip anything that isn't a .wav file
        if not f.lower().endswith(".wav"):
            continue

        
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
ravdess_df = pd.concat([emotion_df, path_df], axis=1)
# changing integers to actual emotions.
ravdess_df.Emotions.replace({1:'neutral', 2:'neutral', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust',
                             8:'surprise'},
                            inplace=True)


#Cream datasets
Crema = os.path.join(Crema, "AudioWAV")
crema_directory_list = os.listdir(Crema)

file_path = []
file_emotion = []

emotion_map = {
    "SAD": "sad",
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral"
}

for file in crema_directory_list:

    full_path = os.path.join(Crema, file)

    # Only process WAV files
    if not file.lower().endswith(".wav"):
        continue

    parts = file.split('_')

    # Valid CREMA files MUST have 4 parts
    # 1001_DFA_ANG_XX.wav → 4 parts
    if len(parts) < 4:
        print("Skipping malformed file:", file)
        continue

    emotion_code = parts[2].upper()   # ANG, SAD, NEU, etc.

    if emotion_code not in emotion_map:
        print("Unknown emotion code:", emotion_code, "in", file)
        continue

    file_emotion.append(emotion_map[emotion_code])
    file_path.append(full_path)

Crema_df = pd.DataFrame({
    "Emotions": file_emotion,
    "Path": file_path
})



#Tess datasets
Tess = os.path.join(Tess, "TESS Toronto emotional speech set data")
level1_folders = os.listdir(Tess)

file_emotion = []
file_path = []

for folder1 in level1_folders:
    folder1_path = os.path.join(Tess, folder1)

    if not os.path.isdir(folder1_path):
        continue

    level2_folders = os.listdir(folder1_path)

    for folder2 in level2_folders:
        folder2_path = os.path.join(folder1_path, folder2)

        if not os.path.isdir(folder2_path):
            continue

        files = os.listdir(folder2_path)

        for file in files:
            if not file.lower().endswith(".wav"):
                continue

            full_path = os.path.join(folder2_path, file)

            # Remove .wav
            name = file[:-4]

            # Split by "_"
            parts = name.split("_")

            # Emotion is always the LAST element
            emotion = parts[-1].lower()

            # fix "ps" → surprise if it appears
            if emotion == "ps":
                emotion = "surprise"

            file_emotion.append(emotion)
            file_path.append(full_path)

Tess_df = pd.DataFrame({
    "Emotions": file_emotion,
    "Path": file_path
})



#Savee datasets
Savee = os.path.join(Savee, "ALL")
savee_directory_list = os.listdir(Savee)

file_path = []
file_emotion = []

for file in savee_directory_list:

    if not file.lower().endswith(".wav"):
        continue

    full_path = os.path.join(Savee, file)

    parts = file.split('_')
    if len(parts) < 2:
        print("Skipping malformed:", file)
        continue

    code = parts[1][:-6]  # remove last 6 characters e.g. "a01.wav" -> "a"

    emotion_map = {
        "a": "angry",
        "d": "disgust",
        "f": "fear",
        "h": "happy",
        "n": "neutral",
        "sa": "sad",
        "su": "surprise"
    }

    # Only append if emotion exists
    if code not in emotion_map:
        print("Unknown emotion code:", code, " -> skipping", file)
        continue

    file_path.append(full_path)
    file_emotion.append(emotion_map[code])   # append emotion only when valid

Savee_df = pd.DataFrame({
    "Emotions": file_emotion,
    "Path": file_path
})



data_path = pd.concat([ravdess_df, Crema_df, Tess_df, Savee_df], axis = 0)
data_path.to_csv("data_path.csv",index=False)

print(data_path.Emotions.value_counts())


plt.title('Count of Emotions', size=16)
sns.countplot(data_path.Emotions)
plt.ylabel('Count', size=12)
plt.xlabel('Emotions', size=12)
sns.despine(top=True, right=True, left=False, bottom=False)
plt.show()

data,sr = librosa.load(file_path[0])

# CREATE LOG MEL SPECTROGRAM
plt.figure(figsize=(10, 5))
spectrogram = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128,fmax=8000) 
log_spectrogram = librosa.power_to_db(spectrogram)
librosa.display.specshow(log_spectrogram, y_axis='mel', sr=sr, x_axis='time');
plt.title('Mel Spectrogram ')
plt.colorbar(format='%+2.0f dB')