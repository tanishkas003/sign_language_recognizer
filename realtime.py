# realtime.py  (updated)
# Press ESC to quit

import os
import warnings
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
warnings.filterwarnings("ignore")
import cv2
import mediapipe as mp
import pickle
import numpy as np
from collections import deque

# ── Load normalized model ──────────────────────────
with open("models/asl_model_normalized.pkl", "rb") as f:
    model = pickle.load(f)
print("Model loaded!")

# ── MediaPipe setup ────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# ── Settings ───────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.40
PREDICT_EVERY_N      = 3      # only predict every 3rd frame → 3x faster
prediction_buffer    = deque(maxlen=10)

cap         = cv2.VideoCapture(0)
frame_count = 0
current_letter     = ""
current_confidence = 0.0

print("Webcam started. Press ESC to quit.")


def normalize_landmarks(hand_landmarks):
    """Convert raw landmarks to wrist-relative, scale-invariant features."""
    wrist = hand_landmarks.landmark[0]

    features = []
    for lm in hand_landmarks.landmark:
        features += [
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z
        ]

    # Scale by wrist → middle knuckle (landmark 9) distance
    ref   = hand_landmarks.landmark[9]
    scale = ((ref.x - wrist.x)**2 + (ref.y - wrist.y)**2) ** 0.5

    if scale > 0:
        features = [v / scale for v in features]

    return features


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame        = cv2.flip(frame, 1)
    rgb_frame    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result       = hands.process(rgb_frame)
    h, w         = frame.shape[:2]

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

        # Only run the model every Nth frame
        if frame_count % PREDICT_EVERY_N == 0:
            features = normalize_landmarks(result.multi_hand_landmarks[0])
            features = np.array(features).reshape(1, -1)

            proba      = model.predict_proba(features)[0]
            confidence = np.max(proba)
            predicted  = model.classes_[np.argmax(proba)]

            prediction_buffer.append(predicted)
            smoothed = max(set(prediction_buffer),
                           key=list(prediction_buffer).count)

            if confidence >= CONFIDENCE_THRESHOLD:
                current_letter     = smoothed
                current_confidence = confidence
            else:
                current_letter     = "?"
                current_confidence = confidence

    else:
        prediction_buffer.clear()
        current_letter     = ""
        current_confidence = 0.0

    # ── UI ─────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 110), (20, 20, 20), -1)

    if current_letter and current_letter != "?":
        cv2.putText(frame, current_letter,
                    (40, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    3.5, (50, 220, 120), 6)
        bar_w = int(current_confidence * 300)
        cv2.rectangle(frame, (160, 28), (160 + bar_w, 56), (50, 220, 120), -1)
        cv2.rectangle(frame, (160, 28), (460, 56), (255, 255, 255), 1)
        cv2.putText(frame, f"Confidence: {current_confidence*100:.1f}%",
                    (160, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (200, 200, 200), 1)

    elif current_letter == "?":
        cv2.putText(frame, "Low confidence...",
                    (40, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (100, 180, 255), 2)

    else:
        cv2.putText(frame, "Show your hand",
                    (40, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, (120, 120, 120), 2)

    cv2.putText(frame, "ESC to quit",
                (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (100, 100, 100), 1)

    cv2.imshow("ASL Recognizer", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()