import math
import cv2
import numpy as np
import os
from datetime import datetime

# Use built-in OpenCV face & eye cascades
face_xml = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt.xml")
eye_xml = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
face_cascade = cv2.CascadeClassifier(face_xml)
eye_cascade = cv2.CascadeClassifier(eye_xml)

cap = cv2.VideoCapture(1)
scaling_factor = 0.7
paused = False
paused_frame = None

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        display_img = frame.copy()
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect face first
        faces = face_cascade.detectMultiScale(gray_full, 1.3, 5)
        for (x_f, y_f, w_f, h_f) in faces:
            face_gray = gray_full[y_f:y_f+h_f, x_f:x_f+w_f]
            face_color = frame[y_f:y_f+h_f, x_f:x_f+w_f]
            eyes = eye_cascade.detectMultiScale(face_gray)

            for (x_e, y_e, w_e, h_e) in eyes:
                # Extract eye ROI, ONLY search pupil here
                eye_roi = face_color[y_e:y_e+h_e, x_e:x_e+w_e]
                gray_eye = cv2.cvtColor(~eye_roi, cv2.COLOR_BGR2GRAY)
                ret, thresh_gray = cv2.threshold(gray_eye, 220, 255, cv2.THRESH_BINARY)
                contours, hierarchy = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                for contour in contours:
                    area = cv2.contourArea(contour)
                    rect = cv2.boundingRect(contour)
                    x, y, width, height = rect
                    radius = 0.25 * (width + height)

                    area_condition = (20 <= area <= 120)
                    symmetry_condition = (abs(1 - float(width)/float(height)) <= 0.25)
                    fill_condition = (abs(1 - (area / (math.pi * math.pow(radius, 2.0)))) <= 0.35)

                    if area_condition and symmetry_condition and fill_condition:
                        # Map pupil local eye coordinates back to full frame
                        abs_x = x_f + x_e + int(x + radius)
                        abs_y = y_f + y_e + int(y + radius)
                        draw_radius = int(1.3 * radius)
                        cv2.circle(display_img, (abs_x, abs_y), draw_radius, (0, 180, 0), -1)

    else:
        display_img = paused_frame.copy()
        cv2.putText(display_img, "PAUSED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Pupil Detector', display_img)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord(' '):
        paused = not paused
        if paused:
            paused_frame = frame.copy()
            print("⏸️ PAUSED | SPACE = resume, S = save snapshot")
        else:
            print("▶️ RESUMED")
    elif key == ord('s'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"pupil_fixed_{timestamp}.png"
        cv2.imwrite(save_name, display_img)
        print(f"✅ Saved {save_name}")

cap.release()
cv2.destroyAllWindows()