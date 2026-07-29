import cv2
import numpy as np
import time
import os

class FeatureMatching:
    def __init__(self, train_image='salinger.jpg'):
        self.orb = cv2.ORB_create(nfeatures=4000, scaleFactor=1.2)
        self.img_train = cv2.imread(train_image, cv2.IMREAD_GRAYSCALE)
        if self.img_train is None:
            print(f"CRITICAL ERROR: Cannot load template image [{train_image}]")
            raise FileNotFoundError(f"{train_image} missing")

        self.key_train, self.desc_train = self.orb.detectAndCompute(self.img_train, None)
        print(f"Template loaded. Template keypoints: {len(self.key_train)}")

        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.last_hinv = np.zeros((3, 3))
        self.num_frames_no_success = 0
        self.max_frames_no_success = 5
        self.max_error_hinv = 80.0
        self.dst_size = (640, 480)

        self.SAVE_RESULTS = True
        print("Working directory for saves:", os.getcwd())

    def _detect_corner_points(self, key_query, good_matches):
        valid_matches = []
        max_q_idx = len(key_query) - 1
        max_t_idx = len(self.key_train) - 1
        # SAFE FILTER: skip any match with out-of-range index
        for m in good_matches:
            if 0 <= m.queryIdx <= max_q_idx and 0 <= m.trainIdx <= max_t_idx:
                valid_matches.append(m)

        if len(valid_matches) < 4:
            return None, None

        src_points = np.float32([self.key_train[m.trainIdx].pt for m in valid_matches])
        dst_points = np.float32([key_query[m.queryIdx].pt for m in valid_matches])
        H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 10.0)

        if H is None:
            return None, None

        inlier_count = np.sum(mask)
        if inlier_count < 4:
            print(f"Homography rejected, only {inlier_count} inliers")
            return None, None

        h_train, w_train = self.img_train.shape[:2]
        src_corners = np.float32([[0, 0], [w_train, 0], [w_train, h_train], [0, h_train]])
        dst_corners = cv2.perspectiveTransform(src_corners[None, :, :], H)[0]
        return dst_corners, H

    def match(self, frame):
        img_query = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sh_query = img_query.shape[:2]

        key_query, desc_query = self.orb.detectAndCompute(img_query, None)
        if desc_query is None or len(key_query) == 0:
            self.num_frames_no_success += 1
            return False, frame

        matches = self.matcher.knnMatch(self.desc_train, desc_query, k=2)
        good_matches = []
        for pair in matches:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < 0.78 * n.distance:
                good_matches.append(m)

        print(f"Good matches found: {len(good_matches)}")

        if len(good_matches) < 4:
            self.num_frames_no_success += 1
            return False, frame

        dst_corners, H = self._detect_corner_points(key_query, good_matches)
        if dst_corners is None or H is None:
            print("Rejected: Homography computation failed")
            self.num_frames_no_success += 1
            return False, frame

        # Wide boundary tolerance
        out_of_bounds = False
        for (x, y) in dst_corners:
            if x < -180 or y < -180 or x > sh_query[1] + 180 or y > sh_query[0] + 180:
                out_of_bounds = True
        if out_of_bounds:
            print("Rejected: Bounding box out of frame limits")
            self.num_frames_no_success += 1
            return False, frame

        area = 0.0
        for i in range(4):
            x1, y1 = dst_corners[i]
            x2, y2 = dst_corners[(i + 1) % 4]
            area += (x1 * y2 - x2 * y1) / 2.0
        area = abs(area)
        print(f"Detected polygon area = {area:.2f}")

        # Area check disabled temporarily
        # frame_area = sh_query[0] * sh_query[1]
        # if area < frame_area / 100.0 or area > frame_area / 1.01:
        #    print(f"Rejected: Area filter fail | Area = {area:.2f}")
        #    self.num_frames_no_success +=1
        #    return False, frame

        src_points = np.float32([self.key_train[m.trainIdx].pt for m in good_matches])
        dst_points = np.float32([key_query[m.queryIdx].pt for m in good_matches])
        Hinv, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)

        transform_diff = np.linalg.norm(Hinv - self.last_hinv)
        recent_failure = self.num_frames_no_success < self.max_frames_no_success
        if recent_failure and transform_diff > self.max_error_hinv:
            self.num_frames_no_success += 1
            return False, frame

        self.last_hinv = Hinv.copy()
        self.num_frames_no_success = 0

        img_warp = cv2.warpPerspective(img_query, Hinv, self.dst_size)
        img_out = cv2.cvtColor(img_warp, cv2.COLOR_GRAY2BGR)

        dst_corners_int = np.int32(dst_corners)
        cv2.polylines(frame, [dst_corners_int], True, (0, 255, 0), thickness=3)

        if self.SAVE_RESULTS:
            filename = f"detected_{int(time.time()*1000)}.png"
            cv2.imwrite(filename, img_out)
            print(f"✅ Saved file: {filename}")

        return True, img_out