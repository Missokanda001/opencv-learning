import sys
import cv2
import numpy as np

def get_contours(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (3,3), 0)
    ret, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

if __name__ == '__main__':
    IMAGE_PATH = r"D:\project_envs\endoscopy-pano\opencv-learning\module02-ch07-detecting-shapes-and-segmenting-an-image\pizza_shapes.png"

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Error: Cannot load image file! Check the file path.")
        sys.exit(1)

    img_copy = img.copy()

    for contour in get_contours(img):
        if cv2.contourArea(contour) < 80:
            continue

        orig_contour = contour
        epsilon = 0.01 * cv2.arcLength(contour, True)
        smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)

        hull = cv2.convexHull(smoothed_contour, returnPoints=False)
        defects = cv2.convexityDefects(smoothed_contour, hull)

        # Draw both contours BEFORE checking defects
        cv2.drawContours(img_copy, [orig_contour], -1, (0, 0, 0), 3)      # Original: Thick BLACK
        cv2.drawContours(img_copy, [smoothed_contour], -1, (0, 0, 255), 2) # Smoothed: Thin RED

        if defects is None:
            continue

        for i in range(defects.shape[0]):
            data = defects[i]
            if data.shape[0] == 1:
                start_defect, end_defect, far_defect, _ = data[0]
                far = tuple(smoothed_contour[far_defect][0])
                cv2.circle(img_copy, far, 7, (255, 0, 0), -1)

    cv2.imwrite("smoothed_contour_result.png", img_copy)
    print("Saved smoothed_contour_result.png")

    cv2.imshow('Smoothed Contours | Convexity Defects', img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()