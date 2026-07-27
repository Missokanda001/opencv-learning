import cv2
import numpy as np


def dodgeNaive(image, mask):
    width, height = image.shape[:2]
    blend = np.zeros((width, height), np.uint8)
    for col in range(width):
        for row in range(height):
            tmp = (image[col, row] << 8) / (255. - mask[col, row])
            if tmp > 255:
                tmp = 255
            blend[col, row] = tmp
    return blend


def dodgeV2(image, mask):
    return cv2.divide(image, 255 - mask, scale=256)


def burnV2(image, mask):
    return 255 - cv2.divide(255 - image, 255 - mask, scale=256)


class PencilSketch:
    def __init__(self, frame_size, bg_gray='pencilsketch_bg.jpg'):
        self.width, self.height = frame_size
        self.canvas = cv2.imread(bg_gray, cv2.CV_8UC1)
        if self.canvas is not None:
            self.canvas = cv2.resize(self.canvas, (self.width, self.height))

    def renderV2(self, img_rgb):
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_gray_inv = 255 - img_gray
        img_blur = cv2.GaussianBlur(img_gray_inv, (21, 21), 0, 0)
        img_blend = dodgeV2(img_gray, img_blur)

        if self.canvas is not None:
            img_blend = cv2.multiply(img_blend, self.canvas, scale=1. / 256)

        return cv2.cvtColor(img_blend, cv2.COLOR_GRAY2RGB)

    def render(self, img_rgb):
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (21, 21), 0, 0)
        img_blend = cv2.divide(img_gray, img_blur, scale=256)
        return img_blend

    def save_result(self, output_image, filename="sketch_output.jpg"):
        # Convert RGB to BGR before saving (fix color inversion)
        output_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, output_bgr)
        print(f"Saved sketch to: {filename}")