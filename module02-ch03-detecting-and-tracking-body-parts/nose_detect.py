import cv2
import numpy as np
import os
from datetime import datetime

# Get absolute path of current script
script_dir = os.path.dirname(os.path.abspath(__file__))
nose_xml_path = os.path.join(script_dir, "cascades", "haarcascade_mcs_nose.xml")

nose_cascade = cv2.CascadeClassifier(nose_xml_path)
nose_detection_enabled = True

# Safe check for missing cascade file
if nose_cascade.empty():
    print("⚠️ WARNING: haarcascade_mcs_nose.xml missing! Nose detection disabled.")
    nose_detection_enabled = False

cap = cv2.VideoCapture(1)  # USB webcam
ds_factor = 0.5
paused = False
paused_frame = None

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read camera frame")
            break

        frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if nose_detection_enabled:
            nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in nose_rects:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                break  # draw only first detected nose

        display_frame = frame.copy()
    else:
        display_frame = paused_frame.copy()
        cv2.putText(display_frame, "PAUSED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Nose Detector', display_frame)
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
        # Save unique timestamp snapshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"nose_detector_snap_{timestamp}.png"
        save_path = os.path.join(script_dir, save_name)
        cv2.imwrite(save_path, display_frame)
        print(f"✅ Saved snapshot: {save_name}")

cap.release()
cv2.destroyAllWindows()