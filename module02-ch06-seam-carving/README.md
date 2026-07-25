# Module 02 - Chapter 06: Seam Carving

Content-aware image resizing using seam carving algorithm.

## Overview

Seam carving is an algorithm for content-aware image resizing. Unlike normal scaling that squashes all pixels evenly, seam carving intelligently removes or adds pixel paths (called "seams") from the least important areas of an image. This preserves important objects (faces, people, foreground) while resizing the background.

A **vertical seam** is a connected path of pixels running from top to bottom of an image, shifting at most 1 pixel left or right per row.

## Files in This Chapter

| File | Purpose |
|---|---|
| `seam_carving_vertical.py` | Shrinks image width by removing low-energy vertical seams |
| `seam_carving_enlarge.py` | Enlarges image width by inserting new seams in low-energy areas |
| `seam_carving_remove_object.py` | Removes a selected object by forcing seams through a ROI, then restoring image size |

## Key Concepts

### 1. Energy Matrix
Computed using Sobel filters (X and Y derivatives). High energy = important detail (edges, textures). Low energy = smooth, boring background.

```
Energy = 0.5 * |Sobel_X| + 0.5 * |Sobel_Y|
```

### 2. Dynamic Programming (DP)
Finds the **minimum energy vertical seam** efficiently:
- `dist_to[row, col]` = minimum energy cost to reach pixel (row, col)
- `edge_to[row, col]` = direction of the path (-1, 0, +1)
- Time complexity: O(rows × cols)

### 3. Seam Removal
Delete the minimum energy seam column-by-column, shift pixels left. Image becomes 1 pixel narrower.

### 4. Seam Insertion (Enlargement)
Find the lowest energy seam, then **duplicate** it (insert averaged neighbor pixels). Image becomes 1 pixel wider.

### 5. Object Removal
- Set energy inside the selected ROI to **0** (forces seams to pass through it)
- Remove enough seams to delete the entire object
- Then add seams back to restore the original image dimensions
- Result: object disappears, background fills in naturally

## How to Run

### Prerequisites
- Python 3.x
- OpenCV (`cv2`)
- NumPy

Install dependencies:
```bash
pip install opencv-python numpy
```

### 1. Shrink Image (Remove Seams)
```bash
python seam_carving_vertical.py <image_path> <num_seams>
```
Example:
```bash
python seam_carving_vertical.py test.jpg 30
```

### 2. Enlarge Image (Add Seams)
```bash
python seam_carving_enlarge.py <image_path> <num_seams>
```
Example:
```bash
python seam_carving_enlarge.py test.jpg 30
```

### 3. Remove Object
```bash
python seam_carving_remove_object.py <image_path>
```
- Click and drag with mouse to draw a green rectangle around the object
- Release mouse button to start removal
- Press **ESC** to exit

## Tips for Best Results

| Tip | Reason |
|---|---|
| Use small images (640px wide max) | Original implementation is single-threaded; large images are very slow |
| Start with 10-30 seams | Too many seams cause visible distortion |
| For object removal: draw tight rectangles | Wider ROI = more iterations = much longer runtime |
| Works best on simple/consistent backgrounds | Complex backgrounds produce visible artifacts |

## Performance Note

The object removal script may appear to freeze ("Not Responding" in Windows) because it runs heavy DP loops in the main thread. The terminal prints progress — watch the `Number of seams removed = X` output to confirm it is still working.

For large images, expect several minutes of computation time.

## Real-World Applications

- **Photoshop Content-Aware Scale** — built on seam carving principles
- **Responsive web design** — adapt hero images to different screen sizes
- **Video retargeting** — convert widescreen video to vertical format for social media
- **Medical imaging** — resize endoscopic/panoramic images without distorting tissue features
- **Robotics vision** — compress camera feeds while preserving important regions

## Limitations

- Only works well when background has low energy (smooth, uniform)
- Cannot perfectly preserve objects when too many seams are removed
- Vertical-only implementation in this chapter (horizontal is also possible)
- Computationally expensive for large images without optimization

