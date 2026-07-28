"""
Hand Gesture Recognition using Kinect Depth Data.

Implements finger counting via convex hull defect analysis on
depth-segmented hand regions.
"""

import cv2
import numpy as np


def angle_rad(v1, v2):
    """Compute the angle in radians between two vectors."""
    return np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))


def deg2rad(angle_deg):
    """Convert degrees to radians."""
    return angle_deg / 180.0 * np.pi


class HandGestureRecognition:
    """Detect and count extended fingers from a Kinect depth frame.

    Algorithm:
        1. Segment hand/arm using depth median thresholding
        2. Find contours and convex hull defects
        3. Count fingers based on defect angle threshold
    """

    def __init__(self):
        # Maximum depth deviation for a pixel to be part of the hand
        self.abs_depth_dev = 14

        # Cut-off angle (deg): below this = finger gap
        self.thresh_deg = 80.0

        # Frame dimensions (set on first frame)
        self.width = None
        self.height = None

    def recognize(self, img_gray):
        """Recognize hand gesture and count extended fingers.

        Args:
            img_gray: Grayscale depth image (uint8).

        Returns:
            tuple: (num_fingers, img_draw)
        """
        self.height, self.width = img_gray.shape[:2]

        # Step 1: Segment the hand/arm region
        segment = self._segment_arm(img_gray)

        # Step 2: Find contours and convex hull defects
        contour, defects = self._find_hull_defects(segment)

        # Step 3: Detect fingers and draw annotations
        img_draw = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        num_fingers, img_draw = self._detect_num_fingers(
            contour, defects, img_draw
        )

        return num_fingers, img_draw

    def _segment_arm(self, frame):
        """Segment the hand/arm region based on depth thresholding.

        Uses median depth of center region as reference, then flood fill
        from center to get connected hand region.
        """
        center_half = 10  # half-width of 21 is 21/2 - 1

        # Define center 21x21 region
        lowerHeight = int(self.height / 2 - center_half)
        upperHeight = int(self.height / 2 + center_half)
        lowerWidth = int(self.width / 2 - center_half)
        upperWidth = int(self.width / 2 + center_half)

        # Get median depth of center region
        center = frame[lowerHeight:upperHeight, lowerWidth:upperWidth]
        med_val = np.median(center)

        # Create mask: pixels within depth tolerance = 128, others = 0
        frame = np.where(
            abs(frame - med_val) <= self.abs_depth_dev,
            128, 0
        ).astype(np.uint8)

        # Morphological closing to fill small holes
        kernel = np.ones((3, 3), np.uint8)
        frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)

        # Ensure center pixel is foreground (flood fill seed)
        small_kernel = 3
        frame[
            int(self.height / 2 - small_kernel):
            int(self.height / 2 + small_kernel),
            int(self.width / 2 - small_kernel):
            int(self.width / 2 + small_kernel)
        ] = 128

        # Flood fill from center
        flood = frame.copy()
        mask = np.zeros((self.height + 2, self.width + 2), np.uint8)
        cv2.floodFill(
            flood, mask,
            (int(self.width / 2), int(self.height / 2)),
            255,
            flags=4 | (255 << 8)
        )

        # Threshold to binary
        ret, flooded = cv2.threshold(flood, 129, 255, cv2.THRESH_BINARY)

        return flooded

    def _find_hull_defects(self, segment):
        """Find largest contour and its convex hull defects."""
        contours, hierarchy = cv2.findContours(
            segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return None, None

        # Largest contour = the hand
        contour = max(contours, key=cv2.contourArea)

        # Compute convex hull and find defects
        hull = cv2.convexHull(contour, returnPoints=False)

        if hull is None or len(hull) < 3:
            return contour, None

        defects = cv2.convexityDefects(contour, hull)

        return contour, defects

    def _detect_num_fingers(self, contour, defects, img_draw):
        """Count extended fingers using convexity defect angles.

        Each defect with angle < threshold = gap between two fingers.
        Green dots = valid finger gaps, Red dots = invalid defects.
        """
        if contour is None or defects is None:
            return 0, img_draw

        num_fingers = 0

        for i in range(defects.shape[0]):
            start_idx, end_idx, farthest_idx, _ = defects[i, 0]

            start = tuple(contour[start_idx][0])
            end = tuple(contour[end_idx][0])
            far = tuple(contour[farthest_idx][0])

            # Draw hull line
            cv2.line(img_draw, start, end, [0, 255, 0], 2)

            # Calculate angle at the farthest point
            angle = angle_rad(
                np.subtract(start, far),
                np.subtract(end, far)
            )

            if angle < deg2rad(self.thresh_deg):
                num_fingers += 1
                cv2.circle(img_draw, far, 5, [0, 255, 0], -1)  # green
            else:
                cv2.circle(img_draw, far, 5, [255, 0, 0], -1)  # red

        return num_fingers, img_draw