# check_setup.py  —  run this to confirm all libraries are installed

import cv2
import mediapipe as mp
import sklearn
import pandas as pd
import numpy as np
import matplotlib

print("✅ OpenCV     :", cv2.__version__)
print("✅ MediaPipe  :", mp.__version__)
print("✅ scikit-learn:", sklearn.__version__)
print("✅ pandas     :", pd.__version__)
print("✅ NumPy      :", np.__version__)
print("✅ Matplotlib :", matplotlib.__version__)
print()
print("🎉 All libraries installed successfully! You're ready for Phase 2.")