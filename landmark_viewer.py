# landmark_viewer.py
# Run this first — it shows you the 21 hand landmarks drawn live on your webcam
# Press Q to quit

import cv2
import mediapipe as mp

# Load MediaPipe's hand tracking module
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

# Set up the hand detector
# max_num_hands=1 because we're only signing with one hand
# min_detection_confidence=0.7 means it only tracks if 70% confident
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# Open your webcam (0 = default camera)
cap = cv2.VideoCapture(0)
print("Webcam started. Show your hand to the camera. Press Q to quit.")

while True:
    ret, frame = cap.read()      # Read one frame
    if not ret:
        print("Could not read from webcam.")
        break

    frame = cv2.flip(frame, 1)   # Mirror it so it feels natural (like a selfie)

    # MediaPipe needs RGB, but OpenCV gives us BGR — convert it
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run hand detection
    result = hands.process(rgb_frame)

    # If a hand was detected, draw all 21 landmarks on the frame
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS   # draws the skeleton lines too
            )

        # Print landmark 8 (index fingertip) coordinates to the terminal
        tip = result.multi_hand_landmarks[0].landmark[8]
        print(f"Index fingertip → x: {tip.x:.3f}  y: {tip.y:.3f}  z: {tip.z:.3f}", end="\r")

    cv2.imshow("Landmark Viewer — Press ESC to quit", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()