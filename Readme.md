📌 VANI – Real-Time Speech Emotion & Speaker Recognition System.
🎙️ Emotion Detection + Speaker Identification (Dual Deep Learning System).
🧠 Project Overview.

VANI is a real-time Speech Emotion Recognition (SER) and Speaker Recognition system built using Deep Learning.
The system captures live audio, extracts MFCC features, and simultaneously predicts:

🎭 Emotion (Happy, Sad, Angry, Fear, Neutral, Disgust, Surprise).

🗣️ Speaker Identity (Unique voice-based recognition).

VANI aims to assist individuals with Attention Deficit, increase emotional awareness, and support interactive intelligent systems.

🚀 Key Features

✔ Real-time emotion & speaker prediction.
✔ Dual-model architecture (Emotion + Speaker).
✔ MFCC feature extraction.
✔ Conv1D + BiLSTM + Attention Network.
✔ Live microphone streaming using sounddevice.
✔ Dataset automation with KaggleHub.
✔ High accuracy with augmentation, cosine LR annealing, early stopping.
✔ Scalable, modular project structure.
✔ Train your own models or use pre-trained ones.

📂 Project Structure

```
VANI/
│── main.py
│── README.md
│── requirements.txt
│── data_path.csv
│── speaker_data.csv
│
├── models/
│    └── README.md  (placeholder, models not uploaded)
│
└── src/
     ├── emotion_data_extraction.py
     ├── voice_data_seperation.py
     ├── features.py
     ├── model_emotion.py
     ├── model_speaker.py
     ├── train_emotion.py
     ├── train_speaker.py
     ├── predict_emotion.py
     ├── predict_speaker.py
     ├── utils.py
     └── __init__.py
```


📊 Datasets Used
Emotion Recognition
✔ RAVDESS.
✔ TESS.
✔ CREMA-D.
✔ SAVEE.

Speaker Recognition
✔ Speaker Recognition Audio Dataset (Kaggle).
✔ VoxCeleb-style structured dataset.

Automatic download using:

```
import kagglehub
dataset_path = kagglehub.dataset_download("dataset/name")
```

🛠️ Technologies & Tools
Machine Learning / Deep Learning

TensorFlow / Keras

Conv1D, BiLSTM, Attention

MFCC Extraction (Librosa)

Cosine Annealing LR Scheduler

Early Stopping, Model Checkpoint

Audio Processing

Librosa

SoundDevice

NumPy

Data Science

Pandas

Scikit-Learn

Matplotlib

🧩 Methodology
1️⃣ Audio Input

Live microphone audio

3-second sampling

Normalized to target sample rate

2️⃣ Feature Extraction

Convert raw waveform → MFCC

Padding / Truncating length

Augmentation:
✔ Noise injection
✔ Time stretch
✔ Pitch shift

3️⃣ Model Processing
🎭 Emotion Model

Conv1D (feature extraction)

BiLSTM (temporal learning)

Attention (focus on strong features)

Softmax prediction

🗣️ Speaker Model

Conv1D + LSTM composite

Optimized with class balancing

4️⃣ Output

```
Emotion: Happy (0.87 confidence)
Speaker: User_03 (0.94 confidence)
```

5️⃣ Real-time Loop

System continuously records → processes → displays → repeats.

🖥️ Installation
Clone the Repo

```
git clone https://github.com/yourname/VANI.git
cd VANI
```

Install Dependencies
```
pip install -r requirements.txt
```

🎓 Training Emotion Model
```
python src/train_emotion.py --data_csv data_path.csv --out_dir models --epochs 60 --batch_size 32
```

🎓 Training Speaker Model
```
python src/train_speaker.py --data_csv speaker_data.csv --out_dir models --max_len 130
```

🎤 Run Real-Time System
```
python main.py --max_len 130 \
 --emotion_model models/emotion_model_final.keras \
 --speaker_model models/speaker_model_final.keras
```

📈 Results & Visualizations
📉 Loss Graph

(Generated during training)

📈 Accuracy Graph

(Training vs Validation)

🔁 Confusion Matrix

Emotion recognition performance per class

Speaker identification clarity

📦 models/ Folder (Important)

Do NOT upload large model files.
Include only this:

models/
└── README.md


With contents explaining how to download or train models.

🤝 Contributing

Pull requests are welcome!
Suggestions for model improvement or dataset expansion are encouraged.

📜 License

MIT License (recommended)

🧠 Conclusion

VANI successfully integrates dual deep learning pipelines for real-time:

Emotion Detection

Speaker Recognition

This system bridges ML and human interaction, enabling smarter emotional understanding and adaptive communication for real-world applications.
