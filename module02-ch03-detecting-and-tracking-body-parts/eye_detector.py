import cv2
import numpy as np
import os
from datetime import datetime

# Load built-in OpenCV Haar cascades (no path errors)
xml_face = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt.xml")
xml_eye = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")

face_cascade = cv2.CascadeClassifier(xml_face)
eye_cascade = cv2.CascadeClassifier(xml_eye)

if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')
if eye_cascade.empty():
    raise IOError('Unable to load the eye cascade classifier xml file')

# Load sunglasses image (put sunglasses.jpg in same folder as this script)
script_dir = os.path.dirname(os.path.abspath(__file__))
sunglasses_path = os.path.join(script_dir, "sunglasses.jpg")
sunglasses_img = cv2.imread(sunglasses_path)
if sunglasses_img is None:
    raise FileNotFoundError(f"Cannot find sunglasses.jpg\nSearched: {sunglasses_path}")

cap = cv2.VideoCapture(1)  # USB webcam
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
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            centers = []
            for (x_eye, y_eye, w_eye, h_eye) in eyes:
                centers.append((x + int(x_eye + 0.5 * w_eye),
                                y + int(y_eye + 0.5 * h_eye)))

            # Only overlay sunglasses if at least 2 eyes are detected
            if len(centers) >= 2:
                # Calculate sunglasses width from eye distance
                sunglasses_width = 2.12 * abs(centers[1][0] - centers[0][0])
                h_sg, w_sg = sunglasses_img.shape[:2]
                scaling_factor = sunglasses_width / w_sg

                overlay_sunglasses = cv2.resize(sunglasses_img, None,
                                                fx=scaling_factor, fy=scaling_factor,
                                                interpolation=cv2.INTER_AREA)

                # X position: leftmost eye
                x_sg = centers[0][0] if centers[0][0] < centers[1][0] else centers[1][0]
                x_sg = int(x_sg - 0.26 * overlay_sunglasses.shape[1])

                # Y position: average eye height
                y_sg = int((centers[0][1] + centers[1][1]) / 2 - 0.85 * overlay_sunglasses.shape[0])

                h_ov, w_ov = overlay_sunglasses.shape[:2]

                # Safety: keep sunglasses within frame bounds
                if y_sg >= 0 and x_sg >= 0 and y_sg + h_ov < frame.shape[0] and x_sg + w_ov < frame.shape[1]:
                    # Create white overlay canvas
                    overlay_img = np.ones(frame.shape, np.uint8) * 255
                    overlay_img[y_sg:y_sg+h_ov, x_sg:x_sg+w_ov] = overlay_sunglasses

                    # Create mask from sunglasses
                    gray_sunglasses = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
                    ret_mask, mask = cv2.threshold(gray_sunglasses, 110, 255, cv2.THRESH_BINARY)
                    mask_inv = cv2.bitwise_not(mask)

                    # Blend sunglasses onto frame
                    temp = cv2.bitwise_and(frame, frame, mask=mask)
                    temp2 = cv2.bitwise_and(overlay_img, overlay_img, mask=mask_inv)
                    frame = cv2.add(temp, temp2)

        display_frame = frame.copy()
    else:
        display_frame = paused_frame.copy()
        cv2.putText(display_frame, "PAUSED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Sunglasses Filter', display_frame)
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
        save_name = f"sunglasses_snap_{timestamp}.png"
        save_path = os.path.join(script_dir, save_name)
        cv2.imwrite(save_path, display_frame)
        print(f"✅ Saved snapshot: {save_path}")

cap.release()
cv2.destroyAllWindows()