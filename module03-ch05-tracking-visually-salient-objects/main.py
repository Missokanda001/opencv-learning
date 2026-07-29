"""
3D Scene Reconstruction from Motion
"""
import numpy as np
import cv2
import os
import sys

from scene3D import SceneReconstruction3D

# ============================================================
# CONFIG
# ============================================================
IMAGE_1 = r"D:\project_envs\endoscopy-pano\opencv-learning\module03-ch05-tracking-visually-salient-objects\img1.jpg"
IMAGE_2 = r"D:\project_envs\endoscopy-pano\opencv-learning\module03-ch05-tracking-visually-salient-objects\img2.jpg"
CALIB_FILE = r"D:\project_envs\endoscopy-pano\opencv-learning\module03-ch05-tracking-visually-salient-objects\calibration.npz"
FEATURE_METHOD = "SIFT"
OUTPUT_FOLDER = "outputs"
# ============================================================


def main():
    print("=" * 60)
    print("  3D Scene Reconstruction from Motion")
    print("=" * 60)
    print()

    # Create output folder
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print("✓ Created output directory: {}".format(OUTPUT_FOLDER))

    # Check images exist
    print("Checking image files...")
    if not os.path.exists(IMAGE_1):
        print("  ✗ Image 1 not found: {}".format(IMAGE_1))
        sys.exit(1)
    if not os.path.exists(IMAGE_2):
        print("  ✗ Image 2 not found: {}".format(IMAGE_2))
        sys.exit(1)
    print("  ✓ Both images found")
    print()

    # Load camera intrinsics (or use fallback)
    print("[1/4] Loading camera calibration...")
    if os.path.exists(CALIB_FILE):
        calib = np.load(CALIB_FILE)
        K = calib['camera_matrix']
        dist = calib['dist_coeffs']
        print("  ✓ Loaded calibration.npz")
    else:
        print("  ⚠ calibration.npz NOT FOUND → Using approximate intrinsics")
        K = np.array([[2759.48,    0.0, 1520.69],
                      [   0.0, 2764.16, 1006.81],
                      [   0.0,    0.0,    1.0]])
        dist = np.zeros((1, 5))

    print()
    print("[2/4] Loading image pair")
    scene = SceneReconstruction3D(K, dist)
    scene.load_image_pair(IMAGE_1, IMAGE_2)
    print("  ✓ Images loaded")
    print("    Image 1: {}x{}".format(scene.img1.shape[1], scene.img1.shape[0]))
    print("    Image 2: {}x{}".format(scene.img2.shape[1], scene.img2.shape[0]))

    print()
    print("[3/4] Compute features and stereo geometry")
    scene._extract_keypoints(FEATURE_METHOD)
    scene._find_fundamental_matrix()
    scene._find_essential_matrix()
    scene._find_camera_matrices_rt()
    print("  ✓ Geometry solved")

    print()
    print("[4/4] Launch visualizations")

    # Window 1: Epipolar Lines
    try:
        print("  → Window 1: Epipolar Lines")
        scene.draw_epipolar_lines(feat_mode=FEATURE_METHOD)
    except ValueError as e:
        print("  ⚠ Skipping Epipolar Lines: {}".format(e))
    except Exception as e:
        print("  ⚠ Epipolar Lines error: {}".format(e))

    # Window 2: Rectified Stereo Images
    try:
        print("  → Window 2: Rectified Stereo Images")
        scene.plot_rectified_images(feat_mode=FEATURE_METHOD)
    except Exception as e:
        print("  ⚠ Skipping Rectified Images: {}".format(e))

    # Window 3: 3D Point Cloud  (FIXED: feat_mode not feat_method)
    try:
        print("  → Window 3: 3D Point Cloud")
        scene.plot_point_cloud(feat_mode=FEATURE_METHOD)
    except Exception as e:
        print("  ⚠ Skipping Point Cloud: {}".format(e))

    print()
    print("=" * 60)
    print("  Pipeline Completed!")
    print("=" * 60)
    print()
    print("Press any key inside image windows to close all.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()