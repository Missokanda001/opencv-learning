import sys
import cv2
import numpy as np

# ============================================================
# Draw vertical seam on top of the image
# ============================================================
def overlay_vertical_seam(img, seam):
    img_seam_overlay = np.copy(img)
    # Extract the list of points from the seam
    x_coords, y_coords = np.transpose([(i, int(j)) for i, j in enumerate(seam)])
    # Draw a green line on the image using the list of points
    img_seam_overlay[x_coords, y_coords] = (0, 255, 0)
    return img_seam_overlay

# ============================================================
# Compute the energy matrix from the input image
# ============================================================
def compute_energy_matrix(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Compute X derivative of the image
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    # Compute Y derivative of the image
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)
    # Return weighted summation of the two images i.e. 0.5*X + 0.5*Y
    return cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

# ============================================================
# Find vertical seam in the input image
# ============================================================
def find_vertical_seam(img, energy):
    rows, cols = img.shape[:2]
    # Initialize the seam vector with 0 for each element
    seam = np.zeros(img.shape[0])
    # Initialize distance and edge matrices
    dist_to = np.zeros(img.shape[:2]) + sys.maxsize
    dist_to[0, :] = np.zeros(img.shape[1])
    edge_to = np.zeros(img.shape[:2])

    # Dynamic programming; iterate using double loop and compute paths efficiently
    for row in range(rows - 1):
        for col in range(cols):
            if col != 0:
                if dist_to[row + 1, col - 1] > dist_to[row, col] + energy[row + 1, col - 1]:
                    dist_to[row + 1, col - 1] = dist_to[row, col] + energy[row + 1, col - 1]
                    edge_to[row + 1, col - 1] = 1
            if dist_to[row + 1, col] > dist_to[row, col] + energy[row + 1, col]:
                dist_to[row + 1, col] = dist_to[row, col] + energy[row + 1, col]
                edge_to[row + 1, col] = 0
            if col != cols - 1:
                if dist_to[row + 1, col + 1] > dist_to[row, col] + energy[row + 1, col + 1]:
                    dist_to[row + 1, col + 1] = dist_to[row, col] + energy[row + 1, col + 1]
                    edge_to[row + 1, col + 1] = -1

    # Retracing the path
    seam[rows - 1] = np.argmin(dist_to[rows - 1, :])
    for i in (x for x in reversed(range(rows)) if x > 0):
        seam[i - 1] = seam[i] + edge_to[i, int(seam[i])]
    return seam

# ============================================================
# Remove the input vertical seam from the image
# ============================================================
def remove_vertical_seam(img, seam):
    rows, cols = img.shape[:2]
    # To delete a point, move every point after it one step towards the left
    for row in range(rows):
        for col in range(int(seam[row]), cols - 1):
            img[row, col] = img[row, col + 1]
    # Discard the last column to create the final output image
    img = img[:, 0:cols - 1]
    return img

# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    # --- Handle command arguments (with fallback defaults) ---
    if len(sys.argv) < 3:
        print("[INFO] Missing arguments! Using defaults: img3.jpg , 20 seams")
        image_path = "D:\\project_envs\\endoscopy-pano\\opencv-learning\\module02-ch06-seam-carving\\img3.jpg"
        num_seams = 20
    else:
        image_path = sys.argv[1]
        num_seams = int(sys.argv[2])

    # --- Load image with error checking ---
    img_input = cv2.imread(image_path)
    if img_input is None:
        print("=" * 50)
        print("ERROR: Cannot load image file!")
        print("Tried to open:", image_path)
        print("")
        print("Fix: Put your image file in the SAME folder as this script,")
        print("     or change 'image_path' above to the correct path.")
        print("=" * 50)
        sys.exit(1)

    print(f"[OK] Image loaded: {img_input.shape[1]}x{img_input.shape[0]} pixels")
    print(f"[INFO] Removing {num_seams} vertical seams...")

    img = np.copy(img_input)
    img_overlay_seam = np.copy(img_input)
    energy = compute_energy_matrix(img)

    for i in range(num_seams):
        seam = find_vertical_seam(img, energy)
        img_overlay_seam = overlay_vertical_seam(img_overlay_seam, seam)
        img = remove_vertical_seam(img, seam)
        energy = compute_energy_matrix(img)
        print(f'Number of seams removed = {i + 1}')

    # --- Show results ---
    cv2.imshow('Input (Original)', img_input)
    cv2.imshow('Seams (green lines = removed)', img_overlay_seam)
    cv2.imshow('Output (Seam Carved)', img)
    
    print("\n[DONE] Press any key on the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()