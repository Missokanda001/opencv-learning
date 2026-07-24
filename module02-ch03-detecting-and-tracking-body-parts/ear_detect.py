import cv2
import numpy as np
import os
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
xml_left_ear = os.path.join(script_dir, "cascades", "haarcascade_mcs_leftear.xml")
xml_right_ear = os.path.join(script_dir, "cascades", "haarcascade_mcs_rightear.xml")

left_ear_cascade = cv2.CascadeClassifier(xml_left_ear)
right_ear_cascade = cv2.CascadeClassifier(xml_right_ear)

# Check if cascades loaded successfully
ear_detection_available = True
if left_ear_cascade.empty() or right_ear_cascade.empty():
    print("⚠️ WARNING: Ear cascade XML files missing! Ear detection disabled.")
    ear_detection_available = False

cap = cv2.VideoCapture(1)  # USB webcam index
ds_factor = 0.5
paused = False
paused_frame = None

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera")
            break

        frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if ear_detection_available:
            left_ear = left_ear_cascade.detectMultiScale(gray, 1.3, 5)
            right_ear = right_ear_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in left_ear:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            for (x, y, w, h) in right_ear:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

        display_frame = frame.copy()
    else:
        display_frame = paused_frame.copy()
        cv2.putText(display_frame, "PAUSED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Ear Detector', display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        # ESC to quit
        break
    elif key == ord(' '):
        # Space: pause / resume
        paused = not paused
        if paused:
            paused_frame = frame.copy()
            print("⏸️ PAUSED | SPACE = resume, S = save snapshot")
        else:
            print("▶️ RESUMED")
    elif key == ord('s'):
        # S: save frame with unique timestamp filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"ear_detector_snap_{timestamp}.png"
        save_path = os.path.join(script_dir, save_name)
        cv2.imwrite(save_path, display_frame)
        print(f"✅ Saved snapshot: {save_name}")

cap.release()
cv2.destroyAllWindows()