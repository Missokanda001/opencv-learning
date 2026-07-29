"""
Multiple Object Tracker using Mean-Shift + Saliency Detection.

This module implements the MultipleObjectsTracker class that tracks
multiple objects in a video by combining:
  1. Saliency-based detection (finding new salient objects)
  2. Mean-shift tracking (following objects from frame to frame)

Based on "OpenCV with Python By Example" - Chapter 5.
Python 3 compatible.

Usage:
    from tracking import MultipleObjectsTracker
    mot = MultipleObjectsTracker(min_area=400, min_shift2=5)
    result_frame = mot.advance_frame(frame, proto_objects_map)
"""

import cv2
import numpy as np
import copy


class MultipleObjectsTracker:
    """
    Tracks multiple visually salient objects using mean-shift algorithm.

    Combines two sources of bounding boxes:
      - Saliency map: detects new salient objects in each frame
      - Mean-shift tracking: follows objects detected in previous frames

    Only keeps boxes confirmed by both methods (after first frame),
    which reduces false positives.

    Attributes:
        object_roi: list of HSV histograms for each tracked object
        object_box: list of (x, y, w, h) bounding boxes for each object
        min_cnt_area: minimum contour area to consider as an object
        min_shift2: minimum squared pixel shift required to keep tracking
        term_crit: termination criteria for mean-shift iterations
    """

    def __init__(self, min_area=400, min_shift2=5):
        """
        Initialize the multi-object tracker.

        Args:
            min_area: minimum contour area (pixels) for a salient
                      region to be considered a proto-object
            min_shift2: minimum squared distance (pixels^2) that an
                        object must move between frames to be kept.
                        Objects that don't move enough are discarded
                        as likely false positives.
        """
        self.object_roi = []
        self.object_box = []
        self.min_cnt_area = min_area
        self.min_shift2 = min_shift2

        # Termination criteria for mean-shift:
        # stop after 100 iterations OR when movement < 1 pixel
        self.term_crit = (cv2.TERM_CRITERIA_EPS |
                          cv2.TERM_CRITERIA_COUNT, 100, 1)

    def advance_frame(self, frame, proto_objects_map):
        """
        Process a new frame and update all object tracks.

        Main public method. Combines saliency detections and mean-shift
        predictions, groups overlapping boxes, and updates bookkeeping.

        Args:
            frame: current BGR video frame
            proto_objects_map: binary mask from saliency detector

        Returns:
            frame: copy of input frame with tracking boxes drawn on it
        """
        self.tracker = copy.deepcopy(frame)

        # Build list of all candidate bounding boxes
        box_all = []

        # Append boxes from the current frame's saliency map
        box_all = self._append_boxes_from_saliency(proto_objects_map,
                                                   box_all)

        # Append boxes extrapolated from last frame via mean-shift
        box_all = self._append_boxes_from_meanshift(frame, box_all)

        # Group overlapping bounding boxes to remove duplicates
        if len(self.object_roi) == 0:
            # No previous frame: keep all boxes from saliency
            group_thresh = 0
        else:
            # Previous frame: require overlap between saliency + mean-shift
            group_thresh = 1

        box_grouped, _ = cv2.groupRectangles(box_all, group_thresh, 0.1)

        # Update mean-shift bookkeeping (histograms) for next frame
        self._update_mean_shift_bookkeeping(frame, box_grouped)

        return self.tracker

    def _append_boxes_from_saliency(self, proto_objects_map, box_all):
        """
        Find bounding boxes of salient objects from the saliency mask.

        Finds all contours in the binary proto-objects map, filters
        by minimum area, and adds their bounding boxes to the list.

        Args:
            proto_objects_map: binary uint8 image (0 or 255)
            box_all: list of boxes to append to

        Returns:
            box_all: updated list with saliency-based boxes added
        """
        # Find all contours in the binary saliency mask
        contours, _ = cv2.findContours(
            proto_objects_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # For each contour, compute bounding box if large enough
        for cnt in contours:
            if cv2.contourArea(cnt) > self.min_cnt_area:
                box = cv2.boundingRect(cnt)
                box_all.append(box)

                # Draw saliency-detected box in GREEN
                x, y, w, h = box
                cv2.rectangle(self.tracker, (x, y), (x + w, y + h),
                              (0, 255, 0), 2)

        return box_all

    def _append_boxes_from_meanshift(self, frame, box_all):
        """
        Predict object positions using mean-shift tracking from last frame.

        For each tracked object, uses mean-shift on the HSV hue histogram
        to find where the object moved. Only keeps boxes that moved enough
        (filters out static false positives like field markings).

        Args:
            frame: current BGR video frame
            box_all: list of boxes to append to

        Returns:
            box_all: updated list with mean-shift predicted boxes added
        """
        # Convert frame to HSV for color histogram-based tracking
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Process all previously stored proto-objects
        for i in range(len(self.object_roi)):
            roi_hist = copy.deepcopy(self.object_roi[i])
            box_old = copy.deepcopy(self.object_box[i])

            # Calculate back projection of the histogram
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            # Run mean-shift to find new object position
            ret, box_new = cv2.meanShift(dst, tuple(box_old),
                                         self.term_crit)

            # Only keep if object moved enough (not false positive)
            xo, yo, wo, ho = box_old
            xn, yn, wn, hn = box_new

            co = [xo + wo / 2, yo + ho / 2]
            cn = [xn + wn / 2, yn + hn / 2]

            if (co[0] - cn[0]) ** 2 + (co[1] - cn[1]) ** 2 >= self.min_shift2:
                box_all.append(box_new)

                # Draw mean-shift tracked box in BLUE
                x, y, w, h = box_new
                cv2.rectangle(self.tracker, (x, y), (x + w, y + h),
                              (255, 0, 0), 2)

        return box_all

    def _update_mean_shift_bookkeeping(self, frame, box_grouped):
        """
        Update the HSV histograms for all tracked objects.

        For each bounding box, extracts the ROI, computes its HSV hue
        histogram, and stores both for use in next frame's mean-shift.

        Args:
            frame: current BGR video frame
            box_grouped: list of (x, y, w, h) boxes after grouping
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Reset storage
        self.object_roi = []
        self.object_box = []

        # Process each grouped bounding box
        for box in box_grouped:
            (x, y, w, h) = box

            # Extract region of interest from HSV image
            hsv_roi = hsv[y:y + h, x:x + w]

            # Mask: ignore dim/bright areas where hue is unreliable
            mask = cv2.inRange(hsv_roi,
                               np.array((0., 60., 32.)),
                               np.array((180., 255., 255.)))

            # Compute histogram of the Hue channel
            roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])

            # Normalize histogram to 0-255 range
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

            # Store for next frame's mean-shift tracking
            self.object_roi.append(roi_hist)
            self.object_box.append(box)

            # Draw final confirmed tracking box in RED
            cv2.rectangle(self.tracker, (x, y), (x + w, y + h),
                          (0, 0, 255), 2)

            # Add label
            cv2.putText(self.tracker, 'Tracked', (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)