import cv2
import numpy as np
import os
from datetime import datetime

# Get absolute path of current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path definitions
mouth_xml_path = os.path.join(script_dir, "cascades", "haarcascade_mcs_mouth.xml")
moustache_img_path = os.path.join(script_dir, "moustache.png")

# Load mouth cascade
mouth_cascade = cv2.CascadeClassifier(mouth_xml_path)
# Load moustache picture
moustache_mask = cv2.imread(moustache_img_path)

# Flag to toggle overlay
enable_moustache_filter = True

# Check missing resources
if mouth_cascade.empty():
    print("⚠️ WARNING: haarcascade_mcs_mouth.xml missing! Mouth detection disabled.")
    enable_moustache_filter = False
if moustache_mask is None:
    print("⚠️ WARNING: moustache.png missing! Cannot draw moustache overlay.")
    enable_moustache_filter = False
else:
    h_mask, w_mask = moustache_mask.shape[:2]

cap = cv2.VideoCapture(1)  # USB webcam
scaling_factor = 0.5
paused = False
paused_frame = None

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read camera frame")
            break

        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if enable_moustache_filter:
            mouth_rects = mouth_cascade.detectMultiScale(gray, 1.3, 5)
            if len(mouth_rects) > 0:
                (x, y, w, h) = mouth_rects[0]
                # Adjust moustache scale & position
                h, w = int(0.6 * h), int(1.2 * w)
                x -= int(0.05 * w)
                y -= int(0.55 * h)

                # Prevent out-of-bound crash
                y = max(y, 0)
                x = max(x, 0)
                if y + h <= frame.shape[0] and x + w <= frame.shape[1]:
                    frame_roi = frame[y:y+h, x:x+w]
                    moustache_mask_small = cv2.resize(moustache_mask, (w, h), interpolation=cv2.INTER_AREA)

                    gray_mask = cv2.cvtColor(moustache_mask_small, cv2.COLOR_BGR2GRAY)
                    ret_thresh, mask = cv2.threshold(gray_mask, 50, 255, cv2.THRESH_BINARY_INV)
                    mask_inv = cv2.bitwise_not(mask)

                    masked_mouth = cv2.bitwise_and(moustache_mask_small, moustache_mask_small, mask=mask)
                    masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_inv)
                    frame[y:y+h, x:x+w] = cv2.add(masked_mouth, masked_frame)

        display_frame = frame.copy()
    else:
        display_frame = paused_frame.copy()
        cv2.putText(display_frame, "PAUSED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Moustache Overlay', display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        # ESC to quit
        break
    elif key == ord(' '):
        # Space pause/resume
        paused = not paused
        if paused:
            paused_frame = frame.copy()
            print("⏸️ PAUSED | SPACE = resume, S = save snapshot")
        else:
            print("▶️ RESUMED")
    elif key == ord('s'):
        # Save timestamped screenshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"moustache_snap_{timestamp}.png"
        save_path = os.path.join(script_dir, save_name)
        cv2.imwrite(save_path, display_frame)
        print(f"✅ Saved snapshot: {save_path}")

cap.release()
cv2.destroyAllWindows()