import cv2
import numpy as np
import os

# Path to your test image (use raw string r"..." to avoid Windows backslash issues)
image_path = r"D:\project_envs\endoscopy-pano\opencv-learning\module02-ch08-object-tracking\blue_test_object.jpg"

# Load the image
frame = cv2.imread(image_path)

# Check if image loaded successfully
if frame is None:
    print(f"ERROR: Could not load image from {image_path}")
    print("Check that the file path is correct and the file exists.")
    exit()

print(f"Loaded image: {image_path}")
print(f"Image size: {frame.shape[1]} x {frame.shape[0]} pixels")

# Convert to HSV colorspace
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Define blue range in HSV colorspace
# Note: Your original range [60,100,100] - [180,255,255] captures green-cyan-blue
# For pure blue only, use the lower range below instead
lower_blue = np.array([60, 100, 100])
upper_blue = np.array([180, 255, 255])

# Threshold the HSV image to get only blue color
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(frame, frame, mask=mask)

# Apply median blur to smooth the result
res = cv2.medianBlur(res, 5)

# Count how many blue pixels were detected
blue_pixel_count = cv2.countNonZero(mask)
print(f"Detected blue pixels: {blue_pixel_count}")

# Save the result to your current working directory
output_filename = "blue_detection_result.png"
cv2.imwrite(output_filename, res)
print(f"Result saved to: {os.path.join(os.getcwd(), output_filename)}")

# Also save the mask (helpful for debugging)
mask_filename = "blue_detection_mask.png"
cv2.imwrite(mask_filename, mask)
print(f"Mask saved to: {os.path.join(os.getcwd(), mask_filename)}")

# Display the results
cv2.imshow("Original Image", frame)
cv2.imshow("Blue Mask", mask)
cv2.imshow("Color Detector Result", res)

# Wait for any key press, then close windows
print("\nPress any key to close the windows...")
cv2.waitKey(0)
cv2.destroyAllWindows()