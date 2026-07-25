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
        if cv2.contourArea(contour) < 100:
            continue

        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        cv2.drawContours(img_copy, [contour], -1, (0, 0, 0), 3)

        if defects is None:
            continue

        for i in range(defects.shape[0]):
            data = defects[i]
            if data.shape[0] == 1:
                s_idx, e_idx, f_idx, distance = data[0]
                start = tuple(contour[s_idx][0])
                end = tuple(contour[e_idx][0])
                far = tuple(contour[f_idx][0])
                cv2.circle(img_copy, far, 6, (128, 0, 0), -1)

    # Save annotated result to the same folder
    cv2.imwrite("pizza_result.png", img_copy)
    print("Saved output image as pizza_result.png")

    cv2.imshow('Convexity Defects | Missing Pizza Slice Detector', img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()