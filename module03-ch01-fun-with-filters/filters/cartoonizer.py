import cv2
import numpy as np


class Cartoonizer:
    def render(self, img_rgb):
        numDownSamples = 2       # number of downscaling steps
        numBilateralFilters = 7  # number of bilateral filtering steps

        # STEP 1: Downsample & smooth color with bilateral filter
        img_color = img_rgb.copy()
        for _ in range(numDownSamples):
            img_color = cv2.pyrDown(img_color)

        for _ in range(numBilateralFilters):
            img_color = cv2.bilateralFilter(img_color, 9, 9, 7)

        for _ in range(numDownSamples):
            img_color = cv2.pyrUp(img_color)

        # STEP 2 & 3: Grayscale + median blur
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(img_gray, 7)

        # STEP 4: Detect edges
        img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 9, 2)

        # STEP 5: Convert edge map back to RGB
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)

        # Combine color image + edge mask to create cartoon effect
        cartoon_frame = cv2.bitwise_and(img_color, img_edge)
        return cartoon_frame

    def save_result(self, output_image, filename="output_cartoon.jpg"):
        # Auto convert RGB to BGR for correct OpenCV saving
        output_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, output_bgr)
        print(f"Saved cartoon output to: {filename}")