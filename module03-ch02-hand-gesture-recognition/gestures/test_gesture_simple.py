import cv2
import numpy as np
from datetime import datetime
# No gestures. prefix now
from hand_gesture import HandGestureRecognition

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    recognizer = HandGestureRecognition()

    print("Hand Gesture Recognition - Simple Test")
    print("--------------------------------------")
    print("Press ESC or 'q' to quit")
    print("Press 's' to save screenshot")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]

        cv2.circle(frame, (int(width / 2), int(height / 2)), 3, (255, 102, 0), 2)
        cv2.rectangle(frame,
                      (int(width / 3), int(height / 3)),
                      (int(width * 2 / 3), int(height * 2 / 3)),
                      (255, 102, 0), 2)

        num_fingers, annotated = recognizer.recognize(gray)

        cv2.putText(annotated, f"Fingers: {num_fingers}",
                    (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)

        cv2.imshow('Hand Gesture Recognition', annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gesture_screenshot_{timestamp}.png"
            cv2.imwrite(filename, annotated)
            print(f"Screenshot saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()