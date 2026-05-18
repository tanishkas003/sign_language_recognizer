# app.py
# Run with: streamlit run app.py

import streamlit as st
import cv2
import mediapipe as mp
import pickle
import numpy as np
from collections import deque
import os
import warnings

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────
st.set_page_config(
    page_title="ASL Sign Language Recognizer",
    page_icon="🤟",
    layout="centered"
)

# ── Header ─────────────────────────────────────────
st.title("🤟 ASL Sign Language Recognizer")
st.markdown(
    "A real-time American Sign Language letter recognition system "
    "built with **MediaPipe** + **Random Forest**."
)
st.divider()

# ── Load model ─────────────────────────────────────
@st.cache_resource   
# @st.cache_resource means the model loads ONCE and stays in memory
# without this, it would reload on every single frame — very slow
def load_model():
    model_path = "models/asl_model_normalized.pkl"
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

if model is None:
    st.error("Model not found! Please run train_model.py first.")
    st.stop()

# ── Sidebar — controls ─────────────────────────────
st.sidebar.header("⚙️ Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.40,
    step=0.05,
    help="Only show prediction if model confidence exceeds this value"
)

show_landmarks = st.sidebar.toggle("Show hand landmarks", value=True)

st.sidebar.divider()
st.sidebar.markdown("### 📖 How it works")
st.sidebar.markdown(
    "1. Webcam captures your hand\n"
    "2. MediaPipe detects 21 landmark points\n"
    "3. Landmarks are normalized relative to wrist\n"
    "4. Random Forest predicts the ASL letter\n"
    "5. Prediction is smoothed over 10 frames"
)

st.sidebar.divider()
st.sidebar.markdown("### 🔤 ASL Alphabet Reference")

# ── Main layout ────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    run = st.toggle("▶ Start Camera", value=False)
    frame_placeholder = st.empty()   # this is where webcam frames go

with col2:
    st.markdown("### Prediction")
    letter_placeholder      = st.empty()
    confidence_placeholder  = st.empty()
    status_placeholder      = st.empty()

st.divider()

# ── Stats section ──────────────────────────────────
st.markdown("### 📊 Session Stats")
stats_col1, stats_col2, stats_col3 = st.columns(3)
frames_placeholder      = stats_col1.empty()
detections_placeholder  = stats_col2.empty()
predictions_placeholder = stats_col3.empty()

# ── MediaPipe setup ────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

def normalize_landmarks(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    features = []
    for lm in hand_landmarks.landmark:
        features += [lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z]
    ref   = hand_landmarks.landmark[9]
    scale = ((ref.x - wrist.x)**2 + (ref.y - wrist.y)**2) ** 0.5
    if scale > 0:
        features = [v / scale for v in features]
    return features

# ── Camera loop ────────────────────────────────────
if run:
    cap              = cv2.VideoCapture(0)
    prediction_buffer = deque(maxlen=10)
    frame_count      = 0
    detection_count  = 0
    prediction_count = 0

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Could not access webcam.")
            break

        frame_count += 1
        frame        = cv2.flip(frame, 1)
        rgb_frame    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result       = hands.process(rgb_frame)

        current_letter     = ""
        current_confidence = 0.0

        if result.multi_hand_landmarks:
            detection_count += 1

            for hand_landmarks in result.multi_hand_landmarks:
                if show_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            if frame_count % 3 == 0:
                features = normalize_landmarks(result.multi_hand_landmarks[0])
                features = np.array(features).reshape(1, -1)

                proba      = model.predict_proba(features)[0]
                confidence = np.max(proba)
                predicted  = model.classes_[np.argmax(proba)]

                prediction_buffer.append(predicted)
                smoothed = max(set(prediction_buffer),
                               key=list(prediction_buffer).count)

                if confidence >= confidence_threshold:
                    current_letter     = smoothed
                    current_confidence = confidence
                    prediction_count  += 1

        # Convert BGR → RGB for Streamlit display
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(display_frame, channels="RGB", use_container_width=True)

        # Update prediction panel
        if current_letter:
            letter_placeholder.markdown(
                f"<h1 style='text-align:center; font-size:120px; color:#32de84;'>"
                f"{current_letter}</h1>",
                unsafe_allow_html=True
            )
            confidence_placeholder.progress(
                current_confidence,
                text=f"Confidence: {current_confidence*100:.1f}%"
            )
            status_placeholder.success("Hand detected ✅")
        else:
            letter_placeholder.markdown(
                "<h1 style='text-align:center; font-size:120px; "
                "color:#888;'>—</h1>",
                unsafe_allow_html=True
            )
            confidence_placeholder.empty()
            if result.multi_hand_landmarks:
                status_placeholder.warning("Low confidence 🤔")
            else:
                status_placeholder.info("Show your hand 👋")

        # Update stats
        frames_placeholder.metric("Frames processed", frame_count)
        detections_placeholder.metric("Hand detections", detection_count)
        predictions_placeholder.metric("Predictions made", prediction_count)

    cap.release()

else:
    # Camera is off — show placeholder
    frame_placeholder.info("👆 Toggle 'Start Camera' to begin")
    letter_placeholder.markdown(
        "<h1 style='text-align:center; font-size:120px; color:#888;'>—</h1>",
        unsafe_allow_html=True
    )