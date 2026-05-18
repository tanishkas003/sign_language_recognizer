# ASL Sign Language Recognizer

A real-time American Sign Language (ASL) letter recognition system built 
using computer vision and machine learning.

## How it works
1. Webcam captures live video
2. MediaPipe detects 21 hand landmark points per frame
3. A trained Random Forest classifier predicts the ASL letter
4. Prediction is displayed on screen in real-time

## Tech stack
- Python
- OpenCV — webcam access and video processing
- MediaPipe — hand landmark detection
- scikit-learn — Random Forest classifier
- pandas / NumPy — data handling

## Setup
```bash
git clone https://github.com/YOUR_USERNAME/asl-recognizer.git
cd asl-recognizer
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

## Usage
```bash
# Collect training data
python collect_data.py

# Train the model
python train_model.py

# Run live recognition
python realtime.py
```

## Project structure
```
asl-recognizer/
├── data/               # landmark CSV dataset
├── models/             # trained model + confusion matrix
├── collect_data.py     # data collection script
├── train_model.py      # model training script
├── realtime.py         # live recognition (Phase 4)
└── requirements.txt
```