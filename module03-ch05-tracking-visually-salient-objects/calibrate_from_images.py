"""
Offline Camera Calibration from saved chessboard images.
Python 3 compatible.

Usage:
    python calibrate_from_images.py --images calib_images/ --pattern-size 9 6
"""

import cv2
import numpy as np
import os
import glob
import json
import argparse


def calibrate_from_images(image_folder, pattern_size=(9, 6), visualize=False):
    objp = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0],
                           0:pattern_size[1]].T.reshape(-1, 2)

    obj_points = []
    img_points = []

    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff')
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
    image_files.sort()

    if not image_files:
        print("No images found in {}".format(image_folder))
        return None, None, None, 0, 0

    print("Found {} images in {}".format(len(image_files), image_folder))

    img_size = None
    success_count = 0

    for img_path in image_files:
        print("Processing: {} ... ".format(os.path.basename(img_path)), end="")
        img = cv2.imread(img_path)
        if img is None:
            print("FAILED (could not read)")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            cv2.cornerSubPix(gray, corners, (9, 9), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners)
            success_count += 1
            print("OK")

            if visualize:
                cv2.drawChessboardCorners(img, pattern_size, corners, ret)
                cv2.imshow('Chessboard', img)
                cv2.waitKey(500)
        else:
            print("FAILED (no corners found)")

    if visualize:
        cv2.destroyAllWindows()

    if success_count == 0:
        print("\nERROR: No chessboard patterns detected in any image.")
        return None, None, None, 0, len(image_files)

    print("\nSuccessfully detected chessboard in {}/{} images".format(
        success_count, len(image_files)))
    print("Running calibration...")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None)

    print("\n" + "=" * 50)
    print("CALIBRATION RESULTS")
    print("=" * 50)
    print("RMS reprojection error: {:.4f} pixels".format(ret))
    print("Images used: {}/{}".format(success_count, len(image_files)))
    print("\nCamera Matrix (K):")
    print(K)
    print("\nDistortion Coefficients (dist):")
    print(dist)
    print("\nImage size: {} x {}".format(img_size[0], img_size[1]))
    print("=" * 50 + "\n")

    calib_data = {
        "camera_matrix": K.tolist(),
        "distortion_coefficients": dist.tolist(),
        "image_width": img_size[0],
        "image_height": img_size[1],
        "rms_error": float(ret),
        "chessboard_size": list(pattern_size),
        "num_images_used": success_count,
        "num_images_total": len(image_files)
    }

    with open("calibration.json", 'w') as f:
        json.dump(calib_data, f, indent=2)
    print("Results saved to: calibration.json")

    np.savez("calibration.npz",
             camera_matrix=K,
             dist_coeffs=dist,
             image_width=img_size[0],
             image_height=img_size[1])
    print("Results saved to: calibration.npz")

    return K, dist, ret, success_count, len(image_files)


def main():
    parser = argparse.ArgumentParser(
        description='Calibrate camera from saved chessboard images')
    parser.add_argument('--images', type=str, default='calib_images/',
                        help='Folder containing chessboard images')
    parser.add_argument('--pattern-size', type=int, nargs=2, default=[9, 6],
                        metavar=('COLS', 'ROWS'),
                        help='Chessboard inner corners (default: 9 6)')
    parser.add_argument('--visualize', action='store_true',
                        help='Show each image with detected corners')

    args = parser.parse_args()
    calibrate_from_images(
        args.images,
        pattern_size=tuple(args.pattern_size),
        visualize=args.visualize
    )


if __name__ == '__main__':
    main()