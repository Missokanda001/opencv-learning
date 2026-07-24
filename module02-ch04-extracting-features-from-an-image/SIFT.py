import cv2
import numpy as np

input_image = cv2.imread('input.jpg')
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
keypoints = sift.detect(gray_image, None)

# Fix: create empty output image as 3rd argument
output_image = np.copy(input_image)
cv2.drawKeypoints(
    input_image,
    keypoints,
    output_image,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

cv2.imshow('SIFT features', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()