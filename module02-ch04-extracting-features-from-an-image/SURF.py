import cv2
import numpy as np

# SURF Implementation
img = cv2.imread('input.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initialize SURF
surf = cv2.xfeatures2d.SURF_create()
surf.hessianThreshold = 15000  # Controls keypoint quantity

# Detect keypoints + compute 64D descriptors
kp, des = surf.detectAndCompute(gray, None)

# Visualize SURF keypoints
img_kp = cv2.drawKeypoints(
    img, kp, None,
    color=(0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

cv2.imshow("SURF Features", img_kp)
cv2.waitKey(0)
cv2.destroyAllWindows()