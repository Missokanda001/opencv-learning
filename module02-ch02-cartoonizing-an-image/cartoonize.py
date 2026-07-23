import cv2
import numpy as np
import argparse

# CHAPTER: Cartoonizing an Image & Webcam Interaction

# Sections:
#   1. Basic Webcam Capture
#   2. Color Space Conversion (g / y / h keys)
#   3. Mouse Quadrant Detection
#   4. Mouse Rectangle Drawing on Webcam
#   5. Cartoonize Effect (s = sketch, c = cartoon)
#   6. Median Blur Filter
#   7. Gaussian vs Bilateral Filter




# 1. Basic Webcam Capture

def basic_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5,
                           interpolation=cv2.INTER_AREA)
        cv2.imshow('Input', frame)

        c = cv2.waitKey(1)
        if c == 27:   # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# 2. Color Space Conversion (Keyboard Controls)
#    g = Grayscale | y = YUV | h = HSV
# ============================================================
def color_space_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    cur_char = -1
    prev_char = -1

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5,
                           interpolation=cv2.INTER_AREA)

        c = cv2.waitKey(1)
        if c == 27:
            break

        if c > -1 and c != prev_char:
            cur_char = c
        prev_char = c

        if cur_char == ord('g'):
            output = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif cur_char == ord('y'):
            output = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        elif cur_char == ord('h'):
            output = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        else:
            output = frame

        cv2.imshow('Webcam - Color Space', output)

    cap.release()
    cv2.destroyAllWindows()



# 3. Mouse Quadrant Detection
#    Click anywhere — the clicked quadrant lights up green.
# ============================================================
def mouse_quadrant():
    width, height = 640, 480
    img = 255 * np.ones((height, width, 3), dtype=np.uint8)

    def detect_quadrant(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if x > width / 2:
                if y > height / 2:
                    point_top_left = (int(width / 2), int(height / 2))
                    point_bottom_right = (width - 1, height - 1)
                else:
                    point_top_left = (int(width / 2), 0)
                    point_bottom_right = (width - 1, int(height / 2))
            else:
                if y > height / 2:
                    point_top_left = (0, int(height / 2))
                    point_bottom_right = (int(width / 2), height - 1)
                else:
                    point_top_left = (0, 0)
                    point_bottom_right = (int(width / 2), int(height / 2))

            cv2.rectangle(img, (0, 0), (width - 1, height - 1),
                          (255, 255, 255), -1)
            cv2.rectangle(img, point_top_left, point_bottom_right,
                          (0, 100, 0), -1)

    cv2.namedWindow('Input window')
    cv2.setMouseCallback('Input window', detect_quadrant)

    while True:
        cv2.imshow('Input window', img)
        c = cv2.waitKey(10)
        if c == 27:
            break

    cv2.destroyAllWindows()


# ============================================================
# 4. Mouse Rectangle Drawing on Live Webcam
#    Click & drag to draw an inverted-color rectangle.
# ============================================================
def mouse_draw_rectangle():
    drawing = False
    top_left_pt, bottom_right_pt = (-1, -1), (-1, -1)
    x_init, y_init = -1, -1

    def draw_rectangle(event, x, y, flags, params):
        nonlocal x_init, y_init, drawing, top_left_pt, bottom_right_pt

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            x_init, y_init = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                top_left_pt = (min(x_init, x), min(y_init, y))
                bottom_right_pt = (max(x_init, x), max(y_init, y))

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            top_left_pt = (min(x_init, x), min(y_init, y))
            bottom_right_pt = (max(x_init, x), max(y_init, y))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    cv2.namedWindow('Webcam - Draw Rectangle')
    cv2.setMouseCallback('Webcam - Draw Rectangle', draw_rectangle)

    while True:
        ret, frame = cap.read()
        img = cv2.resize(frame, None, fx=0.5, fy=0.5,
                         interpolation=cv2.INTER_AREA)

        (x0, y0), (x1, y1) = top_left_pt, bottom_right_pt
        if x0 > -1 and x1 > -1:
            img[y0:y1, x0:x1] = 255 - img[y0:y1, x0:x1]

        cv2.imshow('Webcam - Draw Rectangle', img)
        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# 5. Cartoonize Effect
#    s = sketch mode | c = full cartoon | any other = normal
# ============================================================
def cartoonize_image(img, ds_factor=4, sketch_mode=False):
    # Convert image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply median filter to the grayscale image
    img_gray = cv2.medianBlur(img_gray, 7)
    # Detect edges in the image and threshold it
    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=5)
    ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)

    # 'mask' is the sketch of the image
    if sketch_mode:
        img_sketch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        kernel = np.ones((3, 3), np.uint8)
        img_eroded = cv2.erode(img_sketch, kernel, iterations=1)
        return cv2.medianBlur(img_eroded, 5)

    # Resize the image to a smaller size for faster computation
    img_small = cv2.resize(img, None,
                           fx=1.0 / ds_factor, fy=1.0 / ds_factor,
                           interpolation=cv2.INTER_AREA)

    num_repetitions = 10
    sigma_color = 5
    sigma_space = 7
    size = 5

    # Apply bilateral filter the image multiple times
    for i in range(num_repetitions):
        img_small = cv2.bilateralFilter(img_small, size,
                                        sigma_color, sigma_space)

    img_output = cv2.resize(img_small, None,
                            fx=ds_factor, fy=ds_factor,
                            interpolation=cv2.INTER_LINEAR)

    # Add the thick boundary lines to the image using 'AND' operator
    dst = cv2.bitwise_and(img_output, img_output, mask=mask)
    return dst


def cartoonize_webcam():
    cap = cv2.VideoCapture(0)
    cur_char = -1
    prev_char = -1

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5,
                           interpolation=cv2.INTER_AREA)

        c = cv2.waitKey(1)
        if c == 27:
            break

        if c > -1 and c != prev_char:
            cur_char = c
        prev_char = c

        if cur_char == ord('s'):
            cv2.imshow('Cartoonize',
                       cartoonize_image(frame, sketch_mode=True))
        elif cur_char == ord('c'):
            cv2.imshow('Cartoonize',
                       cartoonize_image(frame, sketch_mode=False))
        else:
            cv2.imshow('Cartoonize', frame)

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# 6. Median Blur Filter (on static image)
# ============================================================
def median_blur_demo():
    img = cv2.imread('input.png')
    if img is None:
        print("WARNING: input.png not found, skipping median blur demo.")
        return

    output = cv2.medianBlur(img, 7)
    cv2.imshow('Input', img)
    cv2.imshow('Median filter', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# 7. Gaussian vs Bilateral Filter Comparison
# ============================================================
def gaussian_bilateral_demo():
    img = cv2.imread('input.jpg')
    if img is None:
        print("WARNING: input.jpg not found, skipping gaussian/bilateral demo.")
        return

    img_gaussian = cv2.GaussianBlur(img, (13, 13), 0)
    img_bilateral = cv2.bilateralFilter(img, 13, 70, 50)

    cv2.imshow('Input', img)
    cv2.imshow('Gaussian filter', img_gaussian)
    cv2.imshow('Bilateral filter', img_bilateral)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# MAIN — run all demos sequentially
# ============================================================
if __name__ == '__main__':
    print("=" * 55)
    print("  Chapter: Cartoonizing an Image & Webcam Interaction")
    print("=" * 55)
    print("  Press ESC to move to the next demo.")
    print("=" * 55)

    print("\n[1/7] Basic Webcam Capture — press ESC to continue")
    basic_webcam()

    print("\n[2/7] Color Space Conversion — g=gray, y=YUV, h=HSV, ESC=next")
    color_space_webcam()

    print("\n[3/7] Mouse Quadrant Detection — click the window, ESC=next")
    mouse_quadrant()

    print("\n[4/7] Mouse Rectangle Drawing — click & drag on webcam, ESC=next")
    mouse_draw_rectangle()

    print("\n[5/7] Cartoonize Effect — s=sketch, c=cartoon, ESC=next")
    cartoonize_webcam()

    print("\n[6/7] Median Blur Filter — press any key to continue")
    median_blur_demo()

    print("\n[7/7] Gaussian vs Bilateral Filter — press any key to exit")
    gaussian_bilateral_demo()

    print("\nAll demos finished.")
