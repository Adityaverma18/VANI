
# 📌 VANI – Real-Time Speech Emotion & Speaker Recognition System
### 🎙️ Dual Deep Learning System: Emotion Detection + Speaker Identification

## 🧠 Project Overview
VANI is a real-time deep learning system capable of recognizing **human emotions** and **speaker identity** from live audio.
It uses **MFCC-based audio processing**, **CNN + BiLSTM models**, and **real-time microphone streaming** to deliver fast, accurate predictions.

The system supports:
- Individuals with Attention Deficiency  
- Emotion-aware AI systems  
- Smart assistants and HCI applications

VANI predicts:
- 🎭 Emotion (Happy, Sad, Angry, Fear, Neutral, Disgust, Surprise)
- 🗣️ Speaker Identity (Who is speaking)

---

## 🚀 Key Features
- Real-time emotion & speaker prediction  
- MFCC feature extraction  
- Conv1D + BiLSTM + Attention  
- Live audio streaming using sounddevice  
- Automatic dataset download via KaggleHub  
- High accuracy with augmentation  
- Cosine LR annealing & early stopping  
- Modular project structure  

---

## 📂 Project Structure
VANI/
```
│── main.py  
│── README.md  
│── requirements.txt  
│── data_path.csv  
│── speaker_data.csv  
│  
├── models/  
│    └── README.md  (models not uploaded)  
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
---

## 📊 Datasets Used
Emotion:
- RAVDESS  
- TESS  
- CREMA-D  
- SAVEE  

Speaker:
- Kaggle Speaker Recognition Dataset  
- Speaker Recognition Audio Dataset  

Download:
```
import kagglehub
path = kagglehub.dataset_download("dataset/name")
```

---

## 🛠️ Technologies Used
- TensorFlow / Keras  
- MFCC (Librosa)  
- Conv1D, BiLSTM, Attention  
- SoundDevice  
- NumPy / Pandas  
- Scikit-Learn  
- Matplotlib  

---

## 🧩 Methodology
### 1️⃣ Audio Input
- Microphone captures 3s audio  
- Resampled & normalized  

### 2️⃣ Feature Extraction
- MFCC generation  
- Padding/truncation  
- Augmentation: noise, pitch shift, time-stretch  

### 3️⃣ Model Processing
- Emotion: Conv1D + BiLSTM + Attention  
- Speaker: Conv1D + LSTM  
- Softmax classification  

### 4️⃣ Output Example
Emotion: Happy (0.87)  
Speaker: User_03 (0.94)

### 5️⃣ Real-Time Loop
Record → MFCC → Emotion Model → Speaker Model → Display → Repeat

---

## 🖥️ Installation
```
git clone https://github.com/yourname/VANI.git
cd VANI
pip install -r requirements.txt
```

---

## 🎓 Training
Emotion:
```
python src/train_emotion.py --data_csv data_path.csv --out_dir models
```

Speaker:
```
python src/train_speaker.py --data_csv speaker_data.csv --out_dir models
```

---

## 🎤 Real-Time Execution
```
python main.py --max_len 130 --emotion_model models/emotion_model_final.keras --speaker_model models/speaker_model_final.keras
```

---

## 📈 Results
- Loss curve  
- Accuracy curve  
- Confusion matrix  

Generated during training.

---

## 📦 models/ Folder
Do NOT upload `.keras` or `.h5` files.

Include only:

```
models/  
└── README.md  
```

---

## 🤝 Contributing
Pull requests welcome.

## 📜 License
MIT License

## 🧠 Conclusion
VANI unifies **emotion detection** and **speaker recognition** in real time using modern deep learning, MFCC processing, and robust audio modeling.

