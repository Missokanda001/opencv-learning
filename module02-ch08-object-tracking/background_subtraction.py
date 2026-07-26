import cv2
import numpy as np
import os

# Capture the input frame
def get_frame(cap, scaling_factor=0.5):
    ret, frame = cap.read()
    # Resize the frame
    frame = cv2.resize(frame, None, fx=scaling_factor,
            fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return frame

if __name__ == '__main__':
    # Initialize the video capture object
    cap = cv2.VideoCapture(1)
    # Create the background subtractor object
    bgSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
    
    # This factor controls the learning rate of the algorithm.
    # The learning rate refers to the rate at which your model
    # will learn about the background. Higher value for
    # 'history' indicates a slower learning rate.
    history = 100
    save_counter = 1

    print(f"Working directory: {os.getcwd()}")
    print("Press S to save screenshot | Press ESC to exit")

    # Iterate until the user presses the ESC key
    while True:
        frame = get_frame(cap, 0.5)
        # Apply the background subtraction model to the input frame
        mask = bgSubtractor.apply(frame, learningRate=1.0/history)
        # Convert from grayscale to 3-channel RGB
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        result = mask & frame

        cv2.imshow('Input frame', frame)
        cv2.imshow('Moving Objects', result)

        # Check key press
        c = cv2.waitKey(10) & 0xFF
        # Save screenshot when S key is pressed
        if c == ord('s'):
            save_name = f"bg_subtraction_{save_counter:03d}.png"
            cv2.imwrite(save_name, result)
            print(f"Saved moving object output: {save_name}")
            save_counter += 1
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()