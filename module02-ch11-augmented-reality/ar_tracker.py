"""
Augmented Reality Tracker
-------------------------
Tracks a user-selected region of interest (ROI) and overlays a 3D pyramid
on top of it using pose estimation (solvePnP).

Supports two input modes:
  1. Webcam (live camera feed)
  2. Static image file

Keyboard shortcuts:
  Space  - Pause / resume
  C      - Clear targets
  S      - Save screenshot (saved to the same folder as this script)
  ESC    - Exit
"""

import sys, os
# Add current script directory to module search path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Then your original import
import cv2
import numpy as np
import os
from datetime import datetime
from pose_estimation import PoseEstimator, ROISelector

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Set to True to use a static image, False to use the webcam
USE_STATIC_IMAGE = False

# Path to the static image (only used if USE_STATIC_IMAGE = True)
STATIC_IMAGE_PATH = (
    r"D:\project_envs\endoscopy-pano\opencv-learning"
    r"\module02-ch11-augmented-reality\test.jpg"
)

# Webcam device index (only used if USE_STATIC_IMAGE = False)
WEBCAM_INDEX = 1

# Frame resize factor (smaller = faster, but less detail)
SCALING_FACTOR = 0.5

# Window name
WINDOW_NAME = "Augmented Reality"


# ---------------------------------------------------------------------------
# Tracker class
# ---------------------------------------------------------------------------

class Tracker(object):
    """AR tracker that overlays a 3D pyramid on a tracked ROI."""

    def __init__(self):
        # --- Input source ---------------------------------------------------
        if USE_STATIC_IMAGE:
            self._init_static_image()
        else:
            self._init_webcam()

        # --- Tracking state -------------------------------------------------
        self.paused = False
        self.tracker = PoseEstimator()

        # --- GUI setup ------------------------------------------------------
        cv2.namedWindow(WINDOW_NAME)
        self.roi_selector = ROISelector(WINDOW_NAME, self.on_rect)

        # --- AR overlay geometry --------------------------------------------
        self.overlay_vertices = np.float32([
            [0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
            [0.5, 0.5, 4],
        ])
        self.overlay_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (0, 4), (1, 4), (2, 4), (3, 4),
        ]
        self.color_base = (0, 255, 0)     # base face color (green)
        self.color_lines = (0, 0, 0)      # edge line color (black)

        # --- Animation counters ---------------------------------------------
        self.graphics_counter = 0
        self.time_counter = 0

        # --- Screenshot output ----------------------------------------------
        self.output_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_counter = 0

    # ------------------------------------------------------------------
    # Input source initializers
    # ------------------------------------------------------------------

    def _init_webcam(self):
        """Initialize webcam capture."""
        self.cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam (index {WEBCAM_INDEX})")
        self.frame = None
        self._is_static = False

    def _init_static_image(self):
        """Load and preprocess a static image file."""
        self.frame = cv2.imread(STATIC_IMAGE_PATH)
        if self.frame is None:
            raise FileNotFoundError(
                f"Could not load image from {STATIC_IMAGE_PATH}"
            )
        self.frame = cv2.resize(
            self.frame, None,
            fx=SCALING_FACTOR, fy=SCALING_FACTOR,
            interpolation=cv2.INTER_AREA,
        )
        self.cap = None
        self._is_static = True

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def on_rect(self, rect):
        """Called when the user draws a new ROI rectangle."""
        self.tracker.add_target(self.frame, rect)

    # ------------------------------------------------------------------
    # Screenshot
    # ------------------------------------------------------------------

    def save_screenshot(self, img):
        """Save the current frame with AR overlay to the same folder."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ar_screenshot_{timestamp}_{self.save_counter:04d}.png"
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, img)
        self.save_counter += 1
        print(f"[SAVED] {filepath}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def start(self):
        """Start the main AR tracking loop."""
        while True:
            # --- Grab a new frame (if webcam mode) -------------------------
            if not self._is_static:
                is_running = (
                    not self.paused
                    and self.roi_selector.selected_rect is None
                )
                if is_running or self.frame is None:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(
                        frame, None,
                        fx=SCALING_FACTOR, fy=SCALING_FACTOR,
                        interpolation=cv2.INTER_AREA,
                    )
                    self.frame = frame.copy()

            # --- Build the display image -----------------------------------
            img = self.frame.copy()

            # --- Track and overlay AR graphics -----------------------------
            if self.roi_selector.selected_rect is not None and not self.paused:
                tracked = self.tracker.track_target(self.frame)
                for item in tracked:
                    self._draw_tracking_overlay(img, item)
                    self.overlay_graphics(img, item)

            # --- Draw the ROI selection rectangle --------------------------
            self.roi_selector.draw_rect(img)

            # --- Show the frame --------------------------------------------
            cv2.imshow(WINDOW_NAME, img)

            # --- Handle keyboard input -------------------------------------
            ch = cv2.waitKey(1)
            if ch == ord(" "):
                self.paused = not self.paused
            elif ch == ord("c"):
                self.tracker.clear_targets()
            elif ch == ord("s"):
                self.save_screenshot(img)
            elif ch == 27:  # ESC
                break

        # --- Cleanup --------------------------------------------------------
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_tracking_overlay(self, img, tracked_item):
        """Draw the tracked quad and feature points."""
        cv2.polylines(
            img, [np.int32(tracked_item.quad)],
            True, self.color_lines, 2,
        )
        for (x, y) in np.int32(tracked_item.points_cur):
            cv2.circle(img, (x, y), 2, self.color_lines)

    def overlay_graphics(self, img, tracked):
        """Overlay a 3D pyramid on top of the tracked ROI using solvePnP."""
        x_start, y_start, x_end, y_end = tracked.target.rect

        # --- 3D reference points (on the tracked plane) ------------------
        quad_3d = np.float32([
            [x_start, y_start, 0],
            [x_end,   y_start, 0],
            [x_end,   y_end,   0],
            [x_start, y_end,   0],
        ])

        # --- Camera intrinsic matrix (simplified) ------------------------
        h, w = img.shape[:2]
        K = np.float64([
            [w, 0, 0.5 * (w - 1)],
            [0, w, 0.5 * (h - 1)],
            [0, 0, 1.0],
        ])
        dist_coef = np.zeros(4)

        # --- Estimate pose (rotation + translation) ----------------------
        ret, rvec, tvec = cv2.solvePnP(
            quad_3d, tracked.quad, K, dist_coef,
        )

        # --- Animate pyramid height --------------------------------------
        self.time_counter += 1
        if not self.time_counter % 20:
            self.graphics_counter = (self.graphics_counter + 1) % 8

        self.overlay_vertices = np.float32([
            [0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
            [0.5, 0.5, self.graphics_counter],
        ])

        # --- Project 3D vertices to 2D image plane -----------------------
        verts = (
            self.overlay_vertices
            * [(x_end - x_start), (y_end - y_start), -(x_end - x_start) * 0.3]
            + (x_start, y_start, 0)
        )
        verts = cv2.projectPoints(
            verts, rvec, tvec, K, dist_coef,
        )[0].reshape(-1, 2)
        verts_floor = np.int32(verts).reshape(-1, 2)

        # --- Draw colored pyramid faces ----------------------------------
        # Base
        cv2.drawContours(img, [verts_floor[:4]], -1, self.color_base, -3)
        # Face 1 (front)
        cv2.drawContours(
            img, [np.vstack((verts_floor[:2], verts_floor[4:5]))],
            -1, (0, 255, 0), -3,
        )
        # Face 2 (right)
        cv2.drawContours(
            img, [np.vstack((verts_floor[1:3], verts_floor[4:5]))],
            -1, (255, 0, 0), -3,
        )
        # Face 3 (back)
        cv2.drawContours(
            img, [np.vstack((verts_floor[2:4], verts_floor[4:5]))],
            -1, (0, 0, 150), -3,
        )
        # Face 4 (left)
        cv2.drawContours(
            img,
            [np.vstack(
                (verts_floor[3:4], verts_floor[0:1], verts_floor[4:5])
            )],
            -1, (255, 255, 0), -3,
        )

        # --- Draw edge lines ---------------------------------------------
        for i, j in self.overlay_edges:
            (x1, y1), (x2, y2) = verts[i], verts[j]
            cv2.line(
                img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                self.color_lines, 2,
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    Tracker().start()
