"""
3D Scene Reconstruction from Motion using OpenCV.
Python 3 compatible.

Features: SURF/SIFT/ORB matching, optical flow, epipolar lines,
image rectification, and 3D point cloud.
"""

import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class SceneReconstruction3D:
    """3D scene reconstruction from two images using structure from motion."""

    def __init__(self, K, dist):
        self.K = K
        self.d = dist
        self.img1 = None
        self.img2 = None
        self.match_pts1 = None
        self.match_pts2 = None
        self.F = None
        self.E = None
        self.Rt2 = None

    def load_image_pair(self, img_path1, img_path2):
        self.img1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)
        self.img2 = cv2.imread(img_path2, cv2.IMREAD_COLOR)

        if len(self.img1.shape) == 2:
            self.img1 = cv2.cvtColor(self.img1, cv2.COLOR_GRAY2BGR)
        if len(self.img2.shape) == 2:
            self.img2 = cv2.cvtColor(self.img2, cv2.COLOR_GRAY2BGR)

        target_width = 600
        if self.img1.shape[1] > target_width:
            scale = target_width / self.img1.shape[1]
            self.img1 = cv2.pyrDown(self.img1)
            self.img2 = cv2.pyrDown(self.img2)
            self.K = self.K * scale
            self.K[2, 2] = 1.0

        self.img1 = cv2.undistort(self.img1, self.K, self.d)
        self.img2 = cv2.undistort(self.img2, self.K, self.d)

        if self.img1 is None:
            raise FileNotFoundError("Could not load: {}".format(img_path1))
        if self.img2 is None:
            raise FileNotFoundError("Could not load: {}".format(img_path2))

        print("Loaded image pair: {} x {}".format(
            self.img1.shape[1], self.img1.shape[0]))

    # --- Feature extraction ---

    def _extract_keypoints(self, feat_mode="SURF"):
        if feat_mode == "SURF":
            try:
                detector = cv2.xfeatures2d.SURF_create(hessianThreshold=200)
            except AttributeError:
                print("SURF not available. Falling back to SIFT.")
                detector = cv2.SIFT_create()
        elif feat_mode == "SIFT":
            detector = cv2.SIFT_create()
        elif feat_mode == "ORB":
            detector = cv2.ORB_create(nfeatures=2000)
        else:
            raise ValueError("Unknown feature mode: {}".format(feat_mode))

        first_key_points, first_desc = detector.detectAndCompute(self.img1, None)
        second_key_points, second_desc = detector.detectAndCompute(self.img2, None)

        print("Found {} keypoints in image 1, {} in image 2".format(
            len(first_key_points), len(second_key_points)))

        if feat_mode == "ORB":
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        matches = matcher.match(first_desc, second_desc)
        matches = sorted(matches, key=lambda x: x.distance)
        num_matches = min(len(matches), 200)
        matches = matches[:num_matches]

        print("Matched {} feature pairs".format(len(matches)))

        first_match_points = np.zeros((len(matches), 2), dtype=np.float32)
        second_match_points = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            first_match_points[i] = first_key_points[match.queryIdx].pt
            second_match_points[i] = second_key_points[match.trainIdx].pt

        self.match_pts1 = first_match_points
        self.match_pts2 = second_match_points

    # --- Optical flow ---

    def _extract_keypoints_flow(self):
        fast = cv2.FastFeatureDetector_create()
        first_key_points = fast.detect(self.img1, None)
        first_key_list = [kp.pt for kp in first_key_points]
        first_key_arr = np.array(first_key_list, dtype=np.float32)

        print("Found {} FAST corners in image 1".format(len(first_key_arr)))

        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS |
                                   cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        prev_gray = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)

        second_key_arr, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, next_gray, first_key_arr, None, **lk_params)

        status = status.flatten()
        good = status == 1

        self.match_pts1 = first_key_arr[good]
        self.match_pts2 = second_key_arr[good]

        print("Successfully tracked {} points with optical flow".format(
            len(self.match_pts1)))

    # --- Epipolar geometry ---

    def _find_fundamental_matrix(self):
        self.F, mask = cv2.findFundamentalMat(
            self.match_pts1, self.match_pts2, cv2.FM_RANSAC, 0.1, 0.99)

        inlier_mask = mask.ravel() == 1
        self.match_pts1 = self.match_pts1[inlier_mask]
        self.match_pts2 = self.match_pts2[inlier_mask]

        print("Fundamental matrix F:")
        print(self.F)
        print("Inliers after RANSAC: {}".format(len(self.match_pts1)))

    def _find_essential_matrix(self):
        self.E = self.K.T @ self.F @ self.K
        print("Essential matrix E:")
        print(self.E)

    def _find_camera_matrices_rt(self):
        U, S, Vt = np.linalg.svd(self.E)

        if np.linalg.det(U) < 0:
            U = -U
        if np.linalg.det(Vt) < 0:
            Vt = -Vt

        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        R1 = U @ W @ Vt
        R2 = U @ W.T @ Vt
        T = U[:, 2].reshape(3, 1)

        solutions = [
            np.hstack([R1,  T]),
            np.hstack([R1, -T]),
            np.hstack([R2,  T]),
            np.hstack([R2, -T]),
        ]

        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])

        best_solution = None
        max_in_front = 0

        for sol in solutions:
            P2 = self.K @ sol
            points_4d = cv2.triangulatePoints(P1, P2,
                                              self.match_pts1.T,
                                              self.match_pts2.T)
            points_3d = points_4d[:3] / points_4d[3]

            in_front_cam1 = np.sum(points_3d[2, :] > 0)

            R = sol[:, :3]
            t = sol[:, 3]
            points_cam2 = R @ points_3d + t.reshape(3, 1)
            in_front_cam2 = np.sum(points_cam2[2, :] > 0)

            in_front = min(in_front_cam1, in_front_cam2)

            if in_front > max_in_front:
                max_in_front = in_front
                best_solution = sol

        self.Rt2 = best_solution
        R = self.Rt2[:, :3]
        T = self.Rt2[:, 3]

        print("Camera 2 rotation R:")
        print(R)
        print("Camera 2 translation T:")
        print(T)
        print("Points in front of both cameras: {}".format(max_in_front))

    # --- Visualizations ---

    def draw_epipolar_lines(self, feat_mode="SURF"):
        if feat_mode == "flow":
            self._extract_keypoints_flow()
        else:
            self._extract_keypoints(feat_mode)

        self._find_fundamental_matrix()

        lines1 = cv2.computeCorrespondEpilines(
            self.match_pts1.reshape(-1, 1, 2), 1, self.F)
        lines1 = lines1.reshape(-1, 3)

        lines2 = cv2.computeCorrespondEpilines(
            self.match_pts2.reshape(-1, 1, 2), 2, self.F)
        lines2 = lines2.reshape(-1, 3)

        img1_lines = self.img1.copy()
        img2_lines = self.img2.copy()

        rng = np.random.default_rng(42)
        for i, (line1, line2, pt1, pt2) in enumerate(
                zip(lines1, lines2, self.match_pts1, self.match_pts2)):
            color = tuple(rng.integers(0, 255, 3).tolist())

            x0, y0 = map(int, [0, -line1[2] / line1[1]])
            x1, y1 = map(int, [self.img2.shape[1],
                               -(line1[2] + line1[0] * self.img2.shape[1]) / line1[1]])
            cv2.line(img2_lines, (x0, y0), (x1, y1), color, 1)
            cv2.circle(img2_lines, tuple(pt2.astype(int)), 3, color, -1)

            x0, y0 = map(int, [0, -line2[2] / line2[1]])
            x1, y1 = map(int, [self.img1.shape[1],
                               -(line2[2] + line2[0] * self.img1.shape[1]) / line2[1]])
            cv2.line(img1_lines, (x0, y0), (x1, y1), color, 1)
            cv2.circle(img1_lines, tuple(pt1.astype(int)), 3, color, -1)

        combined = np.hstack((img1_lines, img2_lines))
        cv2.imshow('Epipolar Lines', combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def plot_rectified_images(self, feat_mode="SURF"):
        if feat_mode == "flow":
            self._extract_keypoints_flow()
        else:
            self._extract_keypoints(feat_mode)

        self._find_fundamental_matrix()
        self._find_essential_matrix()
        self._find_camera_matrices_rt()

        R = self.Rt2[:, :3]
        T = self.Rt2[:, 3]

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            self.K, self.d, self.K, self.d,
            self.img1.shape[:2], R, T, alpha=1.0)

        print("Rectification Q matrix (disparity-to-depth):")
        print(Q)

        mapx1, mapy1 = cv2.initUndistortRectifyMap(
            self.K, self.d, R1, self.K,
            self.img1.shape[:2], cv2.CV_32F)
        mapx2, mapy2 = cv2.initUndistortRectifyMap(
            self.K, self.d, R2, self.K,
            self.img2.shape[:2], cv2.CV_32F)

        img_rect1 = cv2.remap(self.img1, mapx1, mapy1, cv2.INTER_LINEAR)
        img_rect2 = cv2.remap(self.img2, mapx2, mapy2, cv2.INTER_LINEAR)

        total_size = (max(img_rect1.shape[0], img_rect2.shape[0]),
                      img_rect1.shape[1] + img_rect2.shape[1], 3)
        img = np.zeros(total_size, dtype=np.uint8)
        img[:img_rect1.shape[0], :img_rect1.shape[1]] = img_rect1
        img[:img_rect2.shape[0], img_rect1.shape[1]:] = img_rect2

        for i in range(20, img.shape[0], 25):
            cv2.line(img, (0, i), (img.shape[1], i), (255, 0, 0), 1)

        cv2.imshow('Rectified Images', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def plot_optic_flow(self):
        self._extract_keypoints_flow()

        img = self.img1.copy()

        for i in range(len(self.match_pts1)):
            pt1 = tuple(self.match_pts1[i].astype(int))
            pt2 = tuple(self.match_pts2[i].astype(int))

            cv2.line(img, pt1, pt2, (0, 255, 0), 1)
            cv2.circle(img, pt1, 3, (0, 0, 255), -1)

            theta = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
            arrow_len = 6
            arrow_pt1 = (int(pt2[0] - arrow_len * np.cos(theta + np.pi/4)),
                         int(pt2[1] - arrow_len * np.sin(theta + np.pi/4)))
            arrow_pt2 = (int(pt2[0] - arrow_len * np.cos(theta - np.pi/4)),
                         int(pt2[1] - arrow_len * np.sin(theta - np.pi/4)))
            cv2.line(img, pt2, arrow_pt1, (0, 255, 0), 1)
            cv2.line(img, pt2, arrow_pt2, (0, 255, 0), 1)

        cv2.imshow('Optical Flow', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def plot_point_cloud(self, feat_mode="SURF"):
        if feat_mode == "flow":
            self._extract_keypoints_flow()
        else:
            self._extract_keypoints(feat_mode)

        self._find_fundamental_matrix()
        self._find_essential_matrix()
        self._find_camera_matrices_rt()

        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ self.Rt2

        points_4d = cv2.triangulatePoints(P1, P2,
                                          self.match_pts1.T,
                                          self.match_pts2.T)
        points_3d = points_4d[:3] / points_4d[3]

        z_values = points_3d[2, :]
        valid = (z_values > 0) & (z_values < np.percentile(z_values[z_values > 0], 95))
        points_filtered = points_3d[:, valid]

        print("Triangulated {} 3D points ({} after filtering)".format(
            points_3d.shape[1], points_filtered.shape[1]))

        colors = []
        for pt in self.match_pts1[valid]:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= y < self.img1.shape[0] and 0 <= x < self.img1.shape[1]:
                b, g, r = self.img1[y, x]
                colors.append([r / 255.0, g / 255.0, b / 255.0])
            else:
                colors.append([0.5, 0.5, 0.5])
        colors = np.array(colors)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(points_filtered[0, :],
                   points_filtered[1, :],
                   points_filtered[2, :],
                   c=colors, s=5, depthshade=True)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Point Cloud ({} points)'.format(points_filtered.shape[1]))
        ax.set_box_aspect((1, 1, 1))

        plt.tight_layout()
        plt.savefig('point_cloud.png', dpi=150)
        print("Point cloud saved to point_cloud.png")
        plt.show()