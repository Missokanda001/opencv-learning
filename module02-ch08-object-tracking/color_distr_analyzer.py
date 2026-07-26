import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Use TkAgg backend for Windows
import matplotlib.pyplot as plt
import os

# Path to your test image
image_path = r"D:\project_envs\endoscopy-pano\opencv-learning\module02-ch08-object-tracking\blue_test_object.jpg"

# Load the image
img = cv2.imread(image_path)

if img is None:
    print(f"ERROR: Could not load image from {image_path}")
    exit()

print(f"Loaded image: {image_path}")
print(f"Size: {img.shape[1]} x {img.shape[0]}")

# Convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Split HSV channels
h, s, v = cv2.split(hsv)

# ============================================================
# 1. Calculate and print dominant color stats
# ============================================================
print("\n=== HSV Color Distribution Stats ===")
print(f"Hue (H)        -> min: {h.min()}, max: {h.max()}, mean: {h.mean():.1f}")
print(f"Saturation (S) -> min: {s.min()}, max: {s.max()}, mean: {s.mean():.1f}")
print(f"Value (V)      -> min: {v.min()}, max: {v.max()}, mean: {v.mean():.1f}")

# Find the most common hue range (helps tune your blue detector)
hue_hist = cv2.calcHist([h], [0], None, [180], [0, 180])
dominant_hue = np.argmax(hue_hist)
print(f"\nMost common Hue value: {dominant_hue}")
print(f"Suggested blue H range: around {max(0, dominant_hue - 20)} to {min(180, dominant_hue + 20)}")

# ============================================================
# 2. Plot histograms for all 3 HSV channels
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("HSV Color Distribution Analysis", fontsize=14, fontweight="bold")

# Original image (convert BGR -> RGB for matplotlib)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis("off")

# Hue histogram
axes[0, 1].plot(hue_hist, color="purple", linewidth=2)
axes[0, 1].set_title("Hue Distribution (0-180)")
axes[0, 1].set_xlabel("Hue value")
axes[0, 1].set_ylabel("Pixel count")
axes[0, 1].axvline(x=dominant_hue, color="red", linestyle="--",
                   label=f"Dominant: {dominant_hue}")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Saturation histogram
sat_hist = cv2.calcHist([s], [0], None, [256], [0, 256])
axes[1, 0].plot(sat_hist, color="green", linewidth=2)
axes[1, 0].set_title("Saturation Distribution (0-255)")
axes[1, 0].set_xlabel("Saturation value")
axes[1, 0].set_ylabel("Pixel count")
axes[1, 0].grid(True, alpha=0.3)

# Value (brightness) histogram
val_hist = cv2.calcHist([v], [0], None, [256], [0, 256])
axes[1, 1].plot(val_hist, color="orange", linewidth=2)
axes[1, 1].set_title("Value / Brightness Distribution (0-255)")
axes[1, 1].set_xlabel("Value (brightness)")
axes[1, 1].set_ylabel("Pixel count")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()

# Save the histogram chart to working directory
output_chart = "color_distribution_chart.png"
plt.savefig(output_chart, dpi=150)
print(f"\nHistogram chart saved to: {os.path.join(os.getcwd(), output_chart)}")

# Show the plot
plt.show()

# ============================================================
# 3. Also show the HSV channels as separate images
# ============================================================
cv2.imshow("Original Image", img)
cv2.imshow("Hue Channel (H)", h)
cv2.imshow("Saturation Channel (S)", s)
cv2.imshow("Value Channel (V)", v)

# Save individual HSV channel images
cv2.imwrite("hue_channel.png", h)
cv2.imwrite("saturation_channel.png", s)
cv2.imwrite("value_channel.png", v)
print("Hue / Saturation / Value channel images saved.")

print("\nPress any key in the OpenCV windows to close them.")
cv2.waitKey(0)
cv2.destroyAllWindows()