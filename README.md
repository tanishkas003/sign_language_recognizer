# 🤟 ASL Sign Language Recognizer

A real-time American Sign Language (ASL) letter recognition system built 
using computer vision and machine learning — trained on custom self-collected data.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-blue?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-web--app-red?logo=streamlit)

---

## 🎯 What it does

Shows your hand to the webcam → signs an ASL letter → model predicts 
and displays it in real time with a confidence score.

Recognizes all 26 letters of the ASL alphabet with 99%+ test accuracy.

---

## 🧠 How it works
Webcam frame
↓
MediaPipe hand tracking (21 landmark points)
↓
Landmark normalization (wrist-relative, scale-invariant)
↓
Random Forest classifier (100 trees, trained on custom dataset)
↓
Sliding window smoothing (last 10 predictions)
↓
Predicted letter + confidence score

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Test Accuracy | 99%+ |
| Training Samples | 2600 (100 per letter) |
| Features | 63 (21 landmarks × x,y,z) |
| Model | Random Forest (100 trees) |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| OpenCV | Webcam access and frame processing |
| MediaPipe | Hand landmark detection |
| scikit-learn | Random Forest classifier |
| pandas / NumPy | Data handling and feature engineering |
| Streamlit | Web application UI |
| Matplotlib / Seaborn | Model evaluation plots |

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/asl-recognizer.git
cd asl-recognizer

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Option 1 — Web App (recommended)
```bash
streamlit run app.py
```
Opens in your browser at `http://localhost:8501`

### Option 2 — Terminal only
```bash
python realtime.py
```

### Collect your own data
```bash
python collect_data.py     # press A-Z to record each gesture
python normalize_dataset.py
python train_model.py
```

---

## 📁 Project Structure
asl-recognizer/
│
├── app.py                      ← Streamlit web application
├── realtime.py                 ← Terminal-based live recognition
├── collect_data.py             ← Webcam data collection tool
├── normalize_dataset.py        ← Landmark normalization
├── train_model.py              ← Model training + evaluation
│
├── data/
│   ├── landmarks.csv           ← Raw collected landmark data
│   └── landmarks_normalized.csv← Normalized features
│
├── models/
│   ├── asl_model_normalized.pkl← Trained Random Forest model
│   └── confusion_matrix.png    ← Model evaluation plot
│
└── requirements.txt

---

## 💡 Key ML Concepts Applied

- **Feature engineering** — raw pixels → 21 hand landmarks → 63 normalized features
- **Landmark normalization** — wrist-relative coordinates make predictions 
  position and scale invariant
- **Train/test split** — 80/20 split with stratification per class
- **Sliding window smoothing** — reduces frame-to-frame prediction flicker
- **Confidence thresholding** — only display predictions above 40% confidence

---

## 🔮 Future Improvements

- [ ] Word-level prediction (not just letters)
- [ ] Support for both hands
- [ ] Deploy to Hugging Face Spaces for public access
- [ ] Add support for numbers 0–9