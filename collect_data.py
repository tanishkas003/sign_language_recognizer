# collect_data.py
# How to use:
#   1. Run this script
#   2. Show your hand to the camera
#   3. Press any letter key (A-Z) to start recording that gesture
#   4. Hold the gesture steady for a moment — it auto-captures 100 frames
#   5. Repeat for as many letters as you want
#   6. Press Q to quit and save

import cv2
import mediapipe as mp
import pandas as pd
import os

# --- Setup ---
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

SAMPLES_PER_LETTER = 100   # how many frames to capture per letter
DATA_FILE = "data/landmarks.csv"

# Build column names: x0,y0,z0, x1,y1,z1, ... x20,y20,z20, label
columns = []
for i in range(21):
    columns += [f"x{i}", f"y{i}", f"z{i}"]
columns.append("label")

# Load existing data so we don't overwrite previous sessions
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded existing dataset with {len(df)} rows.")
else:
    df = pd.DataFrame(columns=columns)
    print("Starting a fresh dataset.")

# --- State variables ---
collecting    = False   # are we currently recording?
current_label = ""      # which letter are we recording?
sample_count  = 0       # how many frames captured so far
pending_rows  = []      # rows collected but not yet saved

cap = cv2.VideoCapture(0)
print("\nPress A-Z to collect that letter. Press ESC to quit and save.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame      = cv2.flip(frame, 1)
    rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result     = hands.process(rgb_frame)

    hand_detected = result.multi_hand_landmarks is not None

    if hand_detected:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # If we're in recording mode, capture this frame's landmarks
        if collecting and sample_count < SAMPLES_PER_LETTER:
            lm_list = result.multi_hand_landmarks[0].landmark
            row = []
            for lm in lm_list:
                row += [lm.x, lm.y, lm.z]  # append x, y, z for each landmark
            row.append(current_label)
            pending_rows.append(row)
            sample_count += 1

        # Auto-finish when we hit 100 samples
        if collecting and sample_count >= SAMPLES_PER_LETTER:
            collecting = False
            new_df     = pd.DataFrame(pending_rows, columns=columns)
            df         = pd.concat([df, new_df], ignore_index=True)
            df.to_csv(DATA_FILE, index=False)
            pending_rows = []
            print(f"Saved 100 samples for '{current_label}'. "
                  f"Total rows in dataset: {len(df)}")

    # --- Draw the UI overlay on the video frame ---

    # Progress bar while collecting
    if collecting:
        progress = int((sample_count / SAMPLES_PER_LETTER) * 250)
        cv2.rectangle(frame, (30, 420), (30 + progress, 445), (50, 220, 120), -1)
        cv2.rectangle(frame, (30, 420), (280, 445), (255, 255, 255), 2)
        cv2.putText(frame,
                    f"Recording '{current_label}': {sample_count}/{SAMPLES_PER_LETTER}",
                    (30, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 220, 120), 2)

    # Status line at top
    status = "Hand detected" if hand_detected else "No hand detected"
    color  = (50, 220, 120) if hand_detected else (80, 80, 220)
    cv2.putText(frame, status, (30, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, "Press A-Z to record | ESC to quit",
                (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Show per-letter sample counts on the right side
    counts = df["label"].value_counts().sort_index() if len(df) > 0 else {}
    y_pos  = 35
    for letter, count in counts.items():
        bar_w = int((count / SAMPLES_PER_LETTER) * 60)
        cv2.rectangle(frame, (560, y_pos - 10), (560 + bar_w, y_pos + 2),
                      (50, 180, 100), -1)
        cv2.putText(frame, f"{letter}: {count}",
                    (490, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        y_pos += 18

    cv2.imshow("ASL Data Collector", frame)

    # --- Key handling ---
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
    elif 65 <= key <= 90 or 97 <= key <= 122:   # A-Z or a-z
        letter        = chr(key).upper()
        current_label = letter
        collecting    = True
        sample_count  = 0
        pending_rows  = []
        print(f"Starting recording for '{letter}'...")

# Save anything leftover
if pending_rows:
    new_df = pd.DataFrame(pending_rows, columns=columns)
    df     = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

cap.release()
cv2.destroyAllWindows()

print("\n--- Final dataset summary ---")
print(f"Saved to: {DATA_FILE}")
print(f"Total samples: {len(df)}")
if len(df) > 0:
    print(df["label"].value_counts().sort_index())