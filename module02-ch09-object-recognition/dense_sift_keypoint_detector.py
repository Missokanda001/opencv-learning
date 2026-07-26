import cv2
import numpy as np
import sys


class DenseDetector(object):
    def __init__(self, step_size=20, feature_scale=40, img_bound=20):
        self.step_size = step_size
        self.feature_scale = feature_scale
        self.img_bound = img_bound

    def detect(self, img):
        keypoints = []
        h, w = img.shape[:2]
        for y in range(self.img_bound, h - self.img_bound, self.step_size):
            for x in range(self.img_bound, w - self.img_bound, self.step_size):
                kp = cv2.KeyPoint(x, y, self.feature_scale)
                keypoints.append(kp)
        return keypoints


if __name__ == '__main__':
    # Your absolute image path
    image_path = r"D:\project_envs\endoscopy-pano\opencv-learning\module02-ch09-object-recognition\test.jpg"

    input_image = cv2.imread(image_path)
    if input_image is None:
        print("Error: Failed to load the image. Check file path!")
        sys.exit(1)

    input_image_sift = np.copy(input_image)
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Detect dense keypoints
    dense_detector = DenseDetector(step_size=20, feature_scale=20, img_bound=5)
    dense_kp = dense_detector.detect(input_image)
    img_dense = cv2.drawKeypoints(
        input_image,
        dense_kp,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Detect SIFT keypoints
    sift = cv2.SIFT_create()
    sift_kp = sift.detect(gray_image, None)
    img_sift = cv2.drawKeypoints(
        input_image_sift,
        sift_kp,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Show result windows
    cv2.imshow('Dense feature detector', img_dense)
    cv2.imshow('SIFT detector', img_sift)

    # Save output images to the SAME folder as test.jpg
    save_dir = r"D:\project_envs\endoscopy-pano\opencv-learning\module02-ch09-object-recognition"
    cv2.imwrite(f"{save_dir}\\dense_keypoints_output.jpg", img_dense)
    cv2.imwrite(f"{save_dir}\\sift_keypoints_output.jpg", img_sift)
    print(f"Dense result saved to: {save_dir}\\dense_keypoints_output.jpg")
    print(f"SIFT result saved to: {save_dir}\\sift_keypoints_output.jpg")

    cv2.waitKey(0)
    cv2.destroyAllWindows()