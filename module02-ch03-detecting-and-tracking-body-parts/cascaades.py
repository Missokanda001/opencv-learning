import cv2
import numpy as np
import os
from datetime import datetime

# Load pre-packaged Haar cascade inside OpenCV
xml_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt.xml")
face_cascade = cv2.CascadeClassifier(xml_path)

# Auto build absolute path to mask (fix working directory mismatch)
script_dir = os.path.dirname(os.path.abspath(__file__))
mask_path = os.path.join(script_dir, "mask_hannibal.png")
face_mask = cv2.imread(mask_path)

# Safety check for missing image
if face_mask is None:
    raise FileNotFoundError(f"ERROR: Cannot load mask_hannibal.png\nFull path searched: {mask_path}")
h_mask, w_mask = face_mask.shape[:2]

cap = cv2.VideoCapture(1)  # USB webcam

# Check if camera opened successfully
if not cap.isOpened():
    raise IOError("Cannot open camera index 1! Try index 0 or 2.")

scaling_factor = 0.5
paused = False
paused_frame = None

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in face_rects:
            if h > 0 and w > 0:
                # Adjust mask scaling and position
                h_new, w_new = int(1.4 * h), int(1.0 * w)
                y_new = y - int(0.1 * h_new)
                # Prevent coordinate out of bounds error
                y_new = max(y_new, 0)
                x_new = max(x, 0)

                # Extract face region
                frame_roi = frame[y_new:y_new+h_new, x_new:x_new+w_new]
                face_mask_small = cv2.resize(face_mask, (w_new, h_new), interpolation=cv2.INTER_AREA)

                gray_mask = cv2.cvtColor(face_mask_small, cv2.COLOR_BGR2GRAY)
                ret_mask, mask = cv2.threshold(gray_mask, 180, 255, cv2.THRESH_BINARY_INV)
                mask_inv = cv2.bitwise_not(mask)

                masked_face = cv2.bitwise_and(face_mask_small, face_mask_small, mask=mask)
                masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_inv)

                frame[y_new:y_new+h_new, x_new:x_new+w_new] = cv2.add(masked_face, masked_frame)

        display_frame = frame.copy()
    else:
        display_frame = paused_frame.copy()
        # Show "PAUSED" text
        cv2.putText(display_frame, "PAUSED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Face Detector', display_frame)
    c = cv2.waitKey(1) & 0xFF

    if c == 27:  # ESC to exit
        break
    elif c == ord('s'):  # Press s to save snapshot with unique name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"face_snapshot_{timestamp}.png"
        save_path = os.path.join(script_dir, filename)
        cv2.imwrite(save_path, display_frame)
        print(f"✅ Saved: {save_path}")
    elif c == ord(' '):  # Space bar to pause/resume
        paused = not paused
        if paused:
            paused_frame = frame.copy()
            print("⏸️ PAUSED — press SPACE to resume, S to save")
        else:
            print("▶️ Resumed")

cap.release()
cv2.destroyAllWindows()