"""
Pose Estimation Module
----------------------
Provides feature-based pose estimation using ORB features + FLANN matcher,
with a homography-based tracker and an interactive ROI selector.

Classes:
    PoseEstimator  - Detects and tracks planar targets using feature matching
    ROISelector    - Mouse-based rectangle selector for OpenCV windows
    VideoHandler   - Simple demo: webcam + ROI selection + tracking
"""

import sys
from collections import namedtuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Pose Estimator
# ---------------------------------------------------------------------------

class PoseEstimator(object):
    """Feature-based planar target tracker using ORB + FLANN + homography."""

    def __init__(self):
        # FLANN parameters for LSH (Locality Sensitive Hashing) — works with
        # binary descriptors like ORB
        flann_params = dict(
            algorithm=6,
            table_number=6,
            key_size=12,
            multi_probe_level=1,
        )

        self.min_matches = 10

        # Named tuples for structured data
        self.cur_target = namedtuple(
            'Current', 'image, rect, keypoints, descriptors, data'
        )
        self.tracked_target = namedtuple(
            'Tracked', 'target, points_prev, points_cur, H, quad'
        )

        # Feature detector & matcher
        self.feature_detector = cv2.ORB_create(nfeatures=1000)
        self.feature_matcher = cv2.FlannBasedMatcher(flann_params, {})
        self.tracking_targets = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_target(self, image, rect, data=None):
        """Add a new target region to the tracking list."""
        x_start, y_start, x_end, y_end = rect
        keypoints, descriptors = [], []

        all_keypoints, all_descriptors = self.detect_features(image)
        for keypoint, descriptor in zip(all_keypoints, all_descriptors):
            x, y = keypoint.pt
            if x_start <= x <= x_end and y_start <= y <= y_end:
                keypoints.append(keypoint)
                descriptors.append(descriptor)

        descriptors = np.array(descriptors, dtype='uint8')
        self.feature_matcher.add([descriptors])

        target = self.cur_target(
            image=image,
            rect=rect,
            keypoints=keypoints,
            descriptors=descriptors,
            data=data,
        )
        self.tracking_targets.append(target)

    def track_target(self, frame):
        """Detect and match features; return list of tracked targets."""
        self.cur_keypoints, self.cur_descriptors = self.detect_features(frame)

        if len(self.cur_keypoints) < self.min_matches:
            return []

        # KNN match + Lowe's ratio test
        matches = self.feature_matcher.knnMatch(self.cur_descriptors, k=2)
        matches = [
            match[0] for match in matches
            if len(match) == 2
            and match[0].distance < match[1].distance * 0.75
        ]

        if len(matches) < self.min_matches:
            return []

        # Group matches by target image index
        matches_using_index = [[] for _ in range(len(self.tracking_targets))]
        for match in matches:
            matches_using_index[match.imgIdx].append(match)

        # Compute homography for each target
        tracked = []
        for image_index, matches in enumerate(matches_using_index):
            if len(matches) < self.min_matches:
                continue

            target = self.tracking_targets[image_index]

            points_prev = [target.keypoints[m.trainIdx].pt for m in matches]
            points_cur = [self.cur_keypoints[m.queryIdx].pt for m in matches]
            points_prev, points_cur = np.float32(
                (points_prev, points_cur)
            )

            H, status = cv2.findHomography(
                points_prev, points_cur, cv2.RANSAC, 3.0
            )
            status = status.ravel() != 0

            if status.sum() < self.min_matches:
                continue

            points_prev = points_prev[status]
            points_cur = points_cur[status]

            # Project the target rectangle to the current frame
            x_start, y_start, x_end, y_end = target.rect
            quad = np.float32([
                [x_start, y_start],
                [x_end,   y_start],
                [x_end,   y_end],
                [x_start, y_end],
            ])
            quad = cv2.perspectiveTransform(
                quad.reshape(1, -1, 2), H
            ).reshape(-1, 2)

            track = self.tracked_target(
                target=target,
                points_prev=points_prev,
                points_cur=points_cur,
                H=H,
                quad=quad,
            )
            tracked.append(track)

        # Sort by number of inliers (best match first)
        tracked.sort(key=lambda x: len(x.points_prev), reverse=True)
        return tracked

    def detect_features(self, frame):
        """Detect ORB keypoints and compute descriptors."""
        keypoints, descriptors = self.feature_detector.detectAndCompute(
            frame, None
        )
        if descriptors is None:
            descriptors = []
        return keypoints, descriptors

    def clear_targets(self):
        """Remove all tracking targets."""
        self.feature_matcher.clear()
        self.tracking_targets = []


# ---------------------------------------------------------------------------
# ROI Selector
# ---------------------------------------------------------------------------

class ROISelector(object):
    """Mouse-based rectangle selector for an OpenCV window."""

    def __init__(self, win_name, callback_func):
        self.win_name = win_name
        self.callback_func = callback_func
        cv2.setMouseCallback(self.win_name, self.on_mouse_event)
        self.selection_start = None
        self.selected_rect = None

    def on_mouse_event(self, event, x, y, flags, param):
        """Handle mouse events for ROI selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selection_start = (x, y)

        if self.selection_start:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                # Dragging — update the selection rectangle
                x_orig, y_orig = self.selection_start
                x_start, y_start = np.minimum([x_orig, y_orig], [x, y])
                x_end, y_end = np.maximum([x_orig, y_orig], [x, y])
                self.selected_rect = None
                if x_end > x_start and y_end > y_start:
                    self.selected_rect = (x_start, y_start, x_end, y_end)
            else:
                # Mouse released — fire callback with the final rect
                rect = self.selected_rect
                self.selection_start = None
                self.selected_rect = None
                if rect:
                    self.callback_func(rect)

    def draw_rect(self, img):
        """Draw the current selection rectangle on the image."""
        if not self.selected_rect:
            return False
        x_start, y_start, x_end, y_end = self.selected_rect
        cv2.rectangle(
            img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2
        )
        return True


# ---------------------------------------------------------------------------
# Video Handler (standalone demo)
# ---------------------------------------------------------------------------

class VideoHandler(object):
    """Simple demo: webcam feed + ROI selection + feature tracking."""

    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        self.paused = False
        self.frame = None
        self.pose_tracker = PoseEstimator()
        cv2.namedWindow('Tracker')
        self.roi_selector = ROISelector('Tracker', self.on_rect)

    def on_rect(self, rect):
        self.pose_tracker.add_target(self.frame, rect)

    def start(self):
        while True:
            is_running = (
                not self.paused
                and self.roi_selector.selected_rect is None
            )

            if is_running or self.frame is None:
                ret, frame = self.cap.read()
                scaling_factor = 0.5
                frame = cv2.resize(
                    frame, None,
                    fx=scaling_factor, fy=scaling_factor,
                    interpolation=cv2.INTER_AREA,
                )
                if not ret:
                    break
                self.frame = frame.copy()

            img = self.frame.copy()

            if is_running:
                tracked = self.pose_tracker.track_target(self.frame)
                for item in tracked:
                    cv2.polylines(
                        img, [np.int32(item.quad)],
                        True, (255, 255, 255), 2,
                    )
                    for (x, y) in np.int32(item.points_cur):
                        cv2.circle(img, (x, y), 2, (255, 255, 255))

            self.roi_selector.draw_rect(img)
            cv2.imshow('Tracker', img)

            ch = cv2.waitKey(1)
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == ord('c'):
                self.pose_tracker.clear_targets()
            if ch == 27:  # ESC
                break

        self.cap.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point (runs the standalone demo)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    VideoHandler().start()
