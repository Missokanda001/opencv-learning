import cv2
import numpy as np
import os


def start_tracking():
    # Capture the input frame
    cap = cv2.VideoCapture(1)
    # Downsampling factor for the image
    scaling_factor = 0.5
    # Number of frames to keep in the buffer when you
    # are tracking. If you increase this number,
    # feature points will have more "inertia"
    num_frames_to_track = 5
    # Skip every 'n' frames. This is just to increase speed.
    num_frames_jump = 2
    tracking_paths = []
    frame_index = 0
    save_counter = 1

    # 'winSize' refers to the size of each patch. These patches
    # are the smallest blocks on which we operate and track
    # the feature points.
    tracking_params = dict(
        winSize=(11, 11),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    print(f"Optical Flow tracking started. Working directory: {os.getcwd()}")
    print("Press S to save a screenshot. Press ESC to quit.")

    # Iterate until the user presses the ESC key
    while True:
        # Read the input frame
        ret, frame = cap.read()
        # Downsample the input frame
        frame = cv2.resize(
            frame, None,
            fx=scaling_factor,
            fy=scaling_factor,
            interpolation=cv2.INTER_AREA
        )
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output_img = frame.copy()

        if len(tracking_paths) > 0:
            prev_img, current_img = prev_gray, frame_gray
            feature_points_0 = np.float32(
                [tp[-1] for tp in tracking_paths]
            ).reshape(-1, 1, 2)

            # Compute feature points using optical flow (forward)
            feature_points_1, _, _ = cv2.calcOpticalFlowPyrLK(
                prev_img, current_img, feature_points_0,
                None, **tracking_params
            )
            # Compute reverse optical flow for consistency check
            feature_points_0_rev, _, _ = cv2.calcOpticalFlowPyrLK(
                current_img, prev_img, feature_points_1,
                None, **tracking_params
            )

            # Compute the difference of the feature points
            diff_feature_points = abs(
                feature_points_0 - feature_points_0_rev
            ).reshape(-1, 2).max(-1)

            # Threshold and keep only the good points
            good_points = diff_feature_points < 1

            new_tracking_paths = []
            for tp, (x, y), good_points_flag in zip(
                tracking_paths,
                feature_points_1.reshape(-1, 2),
                good_points
            ):
                if not good_points_flag:
                    continue
                tp.append((x, y))
                # Using the queue structure (first in, first out)
                if len(tp) > num_frames_to_track:
                    del tp[0]
                new_tracking_paths.append(tp)
                # Draw green circles on top of the output image
                cv2.circle(output_img, (int(x), int(y)), 3, (0, 255, 0), -1)

            tracking_paths = new_tracking_paths
            # Draw green lines (trails) on top of the output image
            cv2.polylines(
                output_img,
                [np.int32(tp) for tp in tracking_paths],
                False, (0, 150, 0)
            )

        # Skip every 'n'th frame before detecting new features
        if not frame_index % num_frames_jump:
            mask = np.zeros_like(frame_gray)
            mask[:] = 255
            for x, y in [np.int32(tp[-1]) for tp in tracking_paths]:
                cv2.circle(mask, (x, y), 6, 0, -1)

            # Extract good features to track
            feature_points = cv2.goodFeaturesToTrack(
                frame_gray,
                mask=mask,
                maxCorners=500,
                qualityLevel=0.3,
                minDistance=7,
                blockSize=7
            )
            if feature_points is not None:
                for x, y in np.float32(feature_points).reshape(-1, 2):
                    tracking_paths.append([(x, y)])

        frame_index += 1
        prev_gray = frame_gray
        cv2.imshow('Optical Flow', output_img)

        # Check for key press
        c = cv2.waitKey(1) & 0xFF
        # Press S to save screenshot
        if c == ord('s'):
            save_name = f"optical_flow_{save_counter:03d}.png"
            cv2.imwrite(save_name, output_img)
            print(f"Saved: {save_name} -> {os.path.join(os.getcwd(), save_name)}")
            save_counter += 1
        # Press ESC to quit
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    start_tracking()