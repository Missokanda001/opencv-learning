import cv2
import numpy as np

gray_image = cv2.imread('input.jpg', 0)

# FAST with Non-Max Suppression (ENABLED, default)
fast_with_nonmax = cv2.FastFeatureDetector_create()
keypoints = fast_with_nonmax.detect(gray_image, None)
print("Number of keypoints with non max suppression:", len(keypoints))

img_keypoints_with_nonmax = cv2.drawKeypoints(gray_image, keypoints, None, color=(0,255,0))
cv2.imshow('FAST keypoints - with non max suppression', img_keypoints_with_nonmax)

# FAST with Non-Max Suppression (DISABLED)
fast_no_nonmax = cv2.FastFeatureDetector_create(nonmaxSuppression=False)
keypoints_no_nonmax = fast_no_nonmax.detect(gray_image, None)
print("Number of keypoints WITHOUT non max suppression:", len(keypoints_no_nonmax))

img_keypoints_no_nonmax = cv2.drawKeypoints(gray_image, keypoints_no_nonmax, None, color=(0,0,255))
cv2.imshow('FAST keypoints - without non max suppression', img_keypoints_no_nonmax)

cv2.waitKey(0)
cv2.destroyAllWindows()