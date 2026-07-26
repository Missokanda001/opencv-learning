import cv2
import os

# Compute the frame difference
def frame_diff(prev_frame, cur_frame, next_frame):
    diff_frames1 = cv2.absdiff(next_frame, cur_frame)
    diff_frames2 = cv2.absdiff(cur_frame, prev_frame)
    combined_diff = cv2.bitwise_and(diff_frames1, diff_frames2)
    
    # Add threshold to remove tiny noise and brighten motion edges
    _, thresholded = cv2.threshold(combined_diff, 25, 255, cv2.THRESH_BINARY)
    return thresholded

# Capture the frame from webcam
def get_frame(cap, scaling_factor):
    ret, frame = cap.read()
    if not ret:
        return None
    frame = cv2.resize(
        frame, None,
        fx=scaling_factor,
        fy=scaling_factor,
        interpolation=cv2.INTER_AREA
    )
    # FIX: Webcam returns BGR, use COLOR_BGR2GRAY NOT RGB
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


if __name__ == '__main__':
    cap = cv2.VideoCapture(1)
    scaling_factor = 0.5
    save_counter = 1
    motion_threshold = 120

    prev_frame = get_frame(cap, scaling_factor)
    cur_frame = get_frame(cap, scaling_factor)
    next_frame = get_frame(cap, scaling_factor)

    print(f"Saving outputs to: {os.getcwd()}")
    print("Move your face/object slowly. Press ESC to exit.")

    while True:
        motion_result = frame_diff(prev_frame, cur_frame, next_frame)
        cv2.imshow("Object Movement", motion_result)

        # Save only frames with detected motion
        pixel_count = cv2.countNonZero(motion_result)
        if pixel_count > motion_threshold:
            save_name = f"motion_{save_counter:03d}.png"
            cv2.imwrite(save_name, motion_result)
            print(f"Saved {save_name} | Active motion pixels: {pixel_count}")
            save_counter += 1

        prev_frame = cur_frame
        cur_frame = next_frame
        next_frame = get_frame(cap, scaling_factor)

        key = cv2.waitKey(1)  # Reduced delay for smoother frame sampling
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()