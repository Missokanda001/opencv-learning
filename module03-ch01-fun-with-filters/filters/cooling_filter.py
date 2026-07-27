import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline


class CoolingFilter:
    """Makes image look cooler: less red, more blue, lower saturation."""

    def __init__(self):
        # Create lookup tables for tone curves
        self.incr_ch_lut = self._create_LUT_8UC1([0, 128, 255], [0, 192, 255])
        self.decr_ch_lut = self._create_LUT_8UC1([0, 128, 255], [0, 64, 255])

    def _create_LUT_8UC1(self, x, y):
        """Create a 256-entry lookup table using spline interpolation."""
        spl = UnivariateSpline(x, y, k=2)
        return spl(range(256)).astype(np.uint8)

    def render(self, img_rgb):
        # Split into RGB channels
        c_r, c_g, c_b = cv2.split(img_rgb)

        # Cool the image: decrease red, increase blue
        c_r = cv2.LUT(c_r, self.decr_ch_lut).astype(np.uint8)
        c_b = cv2.LUT(c_b, self.incr_ch_lut).astype(np.uint8)
        img_rgb = cv2.merge((c_r, c_g, c_b))

        # Decrease color saturation (convert to HSV, lower S channel)
        c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV))
        c_s = cv2.LUT(c_s, self.decr_ch_lut).astype(np.uint8)

        return cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2RGB)

    def save_result(self, output_image, filename="cooling_output.jpg"):
        """Save filtered image to file, auto RGB -> BGR."""
        output_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, output_bgr)
        print(f"Saved cooling filter result to: {filename}")