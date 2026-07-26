import cv2
import numpy as np
import os

def draw_lines(img_left, img_right, lines, pts_left, pts_right):
    h, w = img_left.shape
    img_left = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
    img_right = cv2.cvtColor(img_right, cv2.COLOR_GRAY2BGR)

    for line, pt_left, pt_right in zip(lines, pts_left, pts_right):
        x_start, y_start = map(int, [0, -line[2] / line[1]])
        x_end, y_end = map(int, [w, -(line[2] + line[0] * w) / line[1]])
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(img_left, (x_start, y_start), (x_end, y_end), color, 1)
        
        # Fix: Cast float point coordinates to int for cv2.circle
        pt_left_int = (int(round(pt_left[0])), int(round(pt_left[1])))
        pt_right_int = (int(round(pt_right[0])), int(round(pt_right[1])))
        
        cv2.circle(img_left, pt_left_int, 5, color, -1)
        cv2.circle(img_right, pt_right_int, 5, color, -1)

    return img_left, img_right


def get_descriptors(gray_image, feature_type):
    if feature_type == 'surf':
        feature_extractor = cv2.xfeatures2d.SURF_create()
    elif feature_type == 'sift':
        feature_extractor = cv2.SIFT_create()
    else:
        raise TypeError("Invalid feature type; should be either 'surf' or 'sift'")

    keypoints, descriptors = feature_extractor.detectAndCompute(gray_image, None)
    return keypoints, descriptors


if __name__ == '__main__':
    # Your file paths
    img_left_path = r"D:\project_envs\endoscopy-pano\opencv-learning\module02-ch10-stereo-vision-and-3d-reconstruction\left.jpg"
    img_right_path = r"D:\project_envs\endoscopy-pano\opencv-learning\module02-ch10-stereo-vision-and-3d-reconstruction\right.jpg"
    feature_type = 'sift'

    # Load images
    img_left = cv2.imread(img_left_path, 0)
    img_right = cv2.imread(img_right_path, 0)

    if img_left is None or img_right is None:
        print("Error: Cannot load input images, check file path!")
        exit()

    scaling_factor = 1.0
    img_left = cv2.resize(img_left, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    img_right = cv2.resize(img_right, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    kps_left, des_left = get_descriptors(img_left, feature_type)
    kps_right, des_right = get_descriptors(img_right, feature_type)

    # FLANN matcher
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_left, des_right, k=2)

    pts_left_image = []
    pts_right_image = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            pts_left_image.append(kps_left[m.queryIdx].pt)
            pts_right_image.append(kps_right[m.trainIdx].pt)

    pts_left_image = np.float32(pts_left_image)
    pts_right_image = np.float32(pts_right_image)
    F, mask = cv2.findFundamentalMat(pts_left_image, pts_right_image, cv2.FM_LMEDS)

    # Keep only inlier points
    pts_left_image = pts_left_image[mask.ravel() == 1]
    pts_right_image = pts_right_image[mask.ravel() == 1]

    # Draw epipolar lines
    lines1 = cv2.computeCorrespondEpilines(pts_right_image.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img_left_lines, img_right_pts = draw_lines(img_left, img_right, lines1, pts_left_image, pts_right_image)

    lines2 = cv2.computeCorrespondEpilines(pts_left_image.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img_right_lines, img_left_pts = draw_lines(img_right, img_left, lines2, pts_right_image, pts_left_image)

    # Show windows
    cv2.imshow('Epi lines on left image', img_left_lines)
    cv2.imshow('Feature points on right image', img_right_pts)
    cv2.imshow('Epi lines on right image', img_right_lines)
    cv2.imshow('Feature points on left image', img_left_pts)

    # ---------------- SAVE ALL OUTPUT IMAGES ----------------
    output_dir = r"D:\project_envs\endoscopy-pano\opencv-learning\module02-ch10-stereo-vision-and-3d-reconstruction\output"
    # Create output folder if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cv2.imwrite(os.path.join(output_dir, "epi_lines_left.png"), img_left_lines)
    cv2.imwrite(os.path.join(output_dir, "feature_points_right.png"), img_right_pts)
    cv2.imwrite(os.path.join(output_dir, "epi_lines_right.png"), img_right_lines)
    cv2.imwrite(os.path.join(output_dir, "feature_points_left.png"), img_left_pts)
    print(f"✅ All result images saved to: {output_dir}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()