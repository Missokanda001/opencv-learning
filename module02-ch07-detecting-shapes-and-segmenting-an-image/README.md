# Chapter 07: Detecting Shapes and Segmenting an Image

## Overview
This chapter explores **contour detection**, **convexity defects**, and **contour approximation/smoothing** using OpenCV. The core example demonstrates how to automatically identify a "pizza with a missing slice" by detecting concave indentations in shape contours — without using template matching.

---

## Files in this Chapter

| File | Purpose |
|------|---------|
| `pizza_defect.py` | Basic convexity defect detection on raw contours |
| `smoothen_contour_pizza.py` | Contour smoothing via `approxPolyDP` + defect detection |
| `pizza_shapes.png` | Input test image (4 shapes: circle, ellipse, rounded rect, notched circle) |
| `pizza_result.png` | Output from basic defect detection |
| `smoothed_contour_result.png` | Output from smoothed contour detection |

---

## Key Concepts

### 1. Contour Detection
`cv2.findContours()` extracts the outline of every shape in a binary thresholded image.
- Requires grayscale → threshold conversion first
- `RETR_EXTERNAL` = only outermost contours
- `CHAIN_APPROX_SIMPLE` = compresses contour points for efficiency

### 2. Convex Hull & Convexity Defects
- **Convex Hull**: The smallest convex polygon that can fully enclose a contour
- **Convexity Defects**: Concave gaps between the original contour and its convex hull
- **Intuition**: A perfect circle has 0 defects; a circle with a slice removed has 1 defect
- This is how we automatically detect the "damaged pizza"

### 3. Contour Smoothing / Approximation
`cv2.approxPolyDP()` simplifies jagged, noisy contours using the **Douglas-Peucker algorithm**.
- `epsilon` controls smoothing strength
- Higher epsilon = more aggressive simplification
- Formula: `epsilon = 0.01 * cv2.arcLength(contour, True)`

---

## Installation
```bash
pip install opencv-python numpy
```

## How to Run
```bash
# Basic defect detection
python pizza_defect.py pizza_shapes.png

# Smoothed contour version
python smoothen_contour_pizza.py pizza_shapes.png
```

> Both scripts also support hardcoded `IMAGE_PATH` variable inside the file (no command-line args needed).

---

## Expected Output
- **Black thick lines**: Raw original contours
- **Red thin lines**: Smoothed approximated contours
- **Blue filled circle**: Marks the convexity defect (the pizza notch)

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| No contours drawn | Toggle `THRESH_BINARY` ↔ `THRESH_BINARY_INV` to match your image contrast |
| Too many false defects | Add `cv2.GaussianBlur()` before thresholding to reduce noise |
| `cannot unpack non-iterable numpy.int32` | Use safe indexing: `defects[i][0]` with shape validation, not `defects[i,0]` |
| Image won't load | Verify file path; use raw string `r"path"` on Windows |

---

## Why This Matters
Convexity defect detection is a shape-agnostic technique. Unlike template matching (which only works for known shapes), this method works on **any object with a concave indentation** — making it useful for industrial defect detection, gesture recognition, and object segmentation.
