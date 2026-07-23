import cv2
import numpy as np
import os

def load_image(path, flags=cv2.IMREAD_COLOR):
    """Helper function to check if image loads successfully"""
    if not os.path.exists(path):
        print(f"WARNING: File {path} NOT FOUND!")
        return None
    img = cv2.imread(path, flags)
    if img is None:
        print(f"WARNING: Failed to load {path}")
    return img

# ==============================================
# 1. Basic 2D Convolution Filters (Smooth, Identity)
# ==============================================
print("\n===== Section 1: 2D Image Convolution Filters =====")
img = load_image('input.jpg')
if img is not None:
    rows, cols = img.shape[:2]
    kernel_identity = np.array([[0,0,0], [0,1,0], [0,0,0]])
    kernel_3x3 = np.ones((3,3), np.float32) / 9.0
    kernel_5x5 = np.ones((5,5), np.float32) / 25.0

    cv2.imshow('Original', img)
    output = cv2.filter2D(img, -1, kernel_identity)
    cv2.imshow('Identity filter', output)
    output = cv2.filter2D(img, -1, kernel_3x3)
    cv2.imshow('3x3 Average Blur', output)
    output = cv2.filter2D(img, -1, kernel_5x5)
    cv2.imshow('5x5 Average Blur', output)
    print("Press any key to continue...")
    cv2.waitKey(0)
cv2.destroyAllWindows()

# ==============================================
# 2. Sobel Edge Detection
# ==============================================
print("\n===== Section 2: Sobel Edge Detection =====")
img_sobel = load_image('input_shapes.png', cv2.IMREAD_GRAYSCALE)
if img_sobel is not None:
    sobel_horizontal = cv2.Sobel(img_sobel, cv2.CV_64F, 1, 0, ksize=5)
    sobel_vertical = cv2.Sobel(img_sobel, cv2.CV_64F, 0, 1, ksize=5)
    # Convert float gradients to visible 8-bit image
    sobel_horizontal = cv2.normalize(np.abs(sobel_horizontal), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    sobel_vertical = cv2.normalize(np.abs(sobel_vertical), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    cv2.imshow('Sobel Original', img_sobel)
    cv2.imshow('Sobel Horizontal Edges', sobel_horizontal)
    cv2.imshow('Sobel Vertical Edges', sobel_vertical)
    print("Press any key to continue...")
    cv2.waitKey(0)
cv2.destroyAllWindows()

# ==============================================
# 3. Emboss Filter
# ==============================================
print("\n===== Section 3: Emboss Effect =====")
img_emboss_input = load_image('input.jpg')
if img_emboss_input is not None:
    kernel_emboss_1 = np.array([[0,-1,-1],[1,0,-1],[1,1,0]])
    kernel_emboss_2 = np.array([[-1,-1,0],[-1,0,1],[0,1,1]])
    kernel_emboss_3 = np.array([[1,0,0],[0,0,0],[0,0,-1]])

    gray_img = cv2.cvtColor(img_emboss_input,cv2.COLOR_BGR2GRAY)
    output_1 = cv2.filter2D(gray_img, -1, kernel_emboss_1) + 128
    output_2 = cv2.filter2D(gray_img, -1, kernel_emboss_2) + 128
    output_3 = cv2.filter2D(gray_img, -1, kernel_emboss_3) + 128

    cv2.imshow('Input', img_emboss_input)
    cv2.imshow('Embossing - South West', output_1)
    cv2.imshow('Embossing - South East', output_2)
    cv2.imshow('Embossing - North West', output_3)
    print("Press any key to continue...")
    cv2.waitKey(0)
cv2.destroyAllWindows()

# ==============================================
# 4. Morphology: Erosion & Dilation
# ==============================================
print("\n===== Section 4: Erosion & Dilation =====")
img_morph = load_image('input.png', 0)
if img_morph is not None:
    kernel = np.ones((5,5), np.uint8)
    img_erosion = cv2.erode(img_morph, kernel, iterations=1)
    img_dilation = cv2.dilate(img_morph, kernel, iterations=1)

    cv2.imshow('Input', img_morph)
    cv2.imshow('Erosion', img_erosion)
    cv2.imshow('Dilation', img_dilation)
    print("Press any key to continue...")
    cv2.waitKey(0)
cv2.destroyAllWindows()

# ==============================================
# 5. Standard Vignette
# ==============================================
print("\n===== Section 5: Standard Vignette =====")
img_vignette = load_image('input.jpg')
if img_vignette is not None:
    rows_v, cols_v = img_vignette.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols_v,200)
    kernel_y = cv2.getGaussianKernel(rows_v,200)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    output_vignette = np.copy(img_vignette)
    for i in range(3):
        output_vignette[:,:,i] = output_vignette[:,:,i] * mask

    cv2.imshow('Original', img_vignette)
    cv2.imshow('Vignette', output_vignette)
    print("Press any key to continue...")
    cv2.waitKey(0)
cv2.destroyAllWindows()

# ==============================================
# 6. Shifted Focus Vignette
# ==============================================
print("\n===== Section 6: Shifted Focus Vignette =====")
img_vignette_shift = load_image('input.jpg')
if img_vignette_shift is not None:
    rows_vs, cols_vs = img_vignette_shift.shape[:2]
    kernel_x = cv2.getGaussianKernel(int(1.5*cols_vs),200)
    kernel_y = cv2.getGaussianKernel(int(1.5*rows_vs),200)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    mask = mask[int(0.5*rows_vs):, int(0.5*cols_vs):]
    output_vignette_shift = np.copy(img_vignette_shift)
    for i in range(3):
        output_vignette_shift[:,:,i] = output_vignette_shift[:,:,i] * mask

    cv2.imshow('Input', img_vignette_shift)
    cv2.imshow('Vignette with shifted focus', output_vignette_shift)
    print("Press any key to continue...")
    cv2.waitKey(0)
cv2.destroyAllWindows()

# ==============================================
# 7. Grayscale Histogram Equalization (Contrast Enhance)
# ==============================================
print("\n===== Section 7: Grayscale Histogram Equalization =====")
img_hist_gray = load_image('input.jpg', 0)
if img_hist_gray is not None:
    histeq = cv2.equalizeHist(img_hist_gray)
    cv2.imshow('Input', img_hist_gray)
    cv2.imshow('Histogram equalized', histeq)
    print("Press any key to continue...")
    cv2.waitKey(0)
cv2.destroyAllWindows()

# ==============================================
# 8. Color Histogram Equalization (YUV)
# ==============================================
print("\n===== Section 8: Color Histogram Equalization =====")
img_color_eq = load_image('input.jpg')
if img_color_eq is not None:
    img_yuv = cv2.cvtColor(img_color_eq, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    cv2.imshow('Color input image', img_color_eq)
    cv2.imshow('Histogram equalized', img_output)
    print("Press any key to exit program.")
    cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nProgram finished.")