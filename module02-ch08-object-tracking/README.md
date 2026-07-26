# Chapter 08: Object Tracking — OpenCV Learning Module


> This chapter covers classical computer vision object tracking algorithms using OpenCV.

---

## 📁 File Overview

| # | File Name | Algorithm | What It Does | Save Key |
|---|-----------|-----------|-------------|----------|
| 1 | `motion_tracking_frame_diff.py` | Triple-Frame Difference | Detects motion by comparing 3 consecutive frames using bitwise AND | `S` |
| 2 | `hsv_blue_color_detection.py` | HSV Color Segmentation | Detects blue-colored objects using HSV color space thresholding | `S` |
| 3 | `color_distribution_analyzer.py` | HSV Histogram Analysis | Analyzes HSV color distribution of a static image to help tune detection ranges | Auto-saves chart |
| 4 | `camshift_object_tracker.py` | CAMShift (Continuously Adaptive Mean Shift) | Mouse-select an object, then tracks it by color histogram back-projection | `S` |
| 5 | `optical_flow_tracking.py` | Lucas-Kanade Optical Flow | Tracks feature points (corners) across frames and draws motion trails | `S` |
| 6 | `background_subtraction.py` | MOG Background Subtraction | Separates moving foreground objects from static background | `S` |

---


### General Controls
- **`S` key** — Save a screenshot of the current output to your working directory
- **`ESC` key** — Exit the program

---

## 📝 Detailed File Descriptions

### 1. `motion_tracking_frame_diff.py`
**Algorithm:** Triple-frame differencing with thresholding

**How it works:**
- Computes absolute difference between `current ↔ next` and `previous ↔ current` frames
- Applies bitwise AND to both difference images
- Only pixels that changed in both intervals appear as motion
- Binary threshold removes camera sensor noise

**Best for:** Fast, transient movement detection
**Limitations:** Poor for slow movement; requires good lighting

**Key parameters to tune:**
- Threshold value in `cv2.threshold()` (default: 25)
- `scaling_factor` (default: 0.5)

---

### 2. `hsv_blue_color_detection.py`
**Algorithm:** HSV color space thresholding + bitwise masking

**How it works:**
- Converts BGR frame to HSV color space
- Applies `cv2.inRange()` to create a mask of pixels within the blue HSV range
- Applies mask to original frame with `cv2.bitwise_and()`
- Median blur smooths the result

**Best for:** Tracking objects of a known, distinct color
**Limitations:** Fails if background contains similar colors; sensitive to lighting changes



### 3. `color_distribution_analyzer.py`
**Algorithm:** HSV histogram analysis using matplotlib

**How it works:**
- Loads a static image (set `image_path` to your target object photo)
- Splits into H, S, V channels
- Computes and plots histograms for each channel
- Finds the dominant (most common) hue value
- Saves a 4-panel chart: original image + 3 histograms

**Best for:** Calibrating HSV ranges before building color trackers

**Output files generated:**
- `color_distribution_chart.png` — 4-panel histogram chart
- `hue_channel.png` — Hue channel grayscale image
- `saturation_channel.png` — Saturation channel grayscale image
- `value_channel.png` — Value (brightness) channel grayscale image

**How to use the results:**
Look at the **dominant hue** printed in console. Set your HSV range roughly ±20 around that value for best detection.

---

### 4. `camshift_object_tracker.py`
**Algorithm:** CAMShift (Continuously Adaptive Mean Shift)

**How it works:**
1. Click and drag with mouse to draw a rectangle around your target object
2. Code extracts the Hue histogram of the selected region
3. Uses `cv2.calcBackProject()` to find where similar colors exist in each frame
4. CAMShift algorithm iteratively adjusts the tracking window
5. Draws an ellipse around the tracked object

**Best for:** Tracking single colored objects with stable appearance
**Limitations:** Can drift if object color blends with background; struggles with occlusion

**Key parameters to tune:**
- Histogram bins (default: 16)
- Termination criteria `(10, 1)` — max iterations and epsilon

---

### 5. `optical_flow_tracking.py`
**Algorithm:** Lucas-Kanade Sparse Optical Flow (Pyramidal)

**How it works:**
- Detects corner features using `cv2.goodFeaturesToTrack()` (Shi-Tomasi)
- Tracks features across frames with `cv2.calcOpticalFlowPyrLK()`
- Uses forward-backward consistency check to filter unreliable points
- Draws green circles on tracked points + green trails showing movement history
- Old points fade out (FIFO queue of `num_frames_to_track`)

**Best for:** General motion analysis, tracking multiple feature points simultaneously
**Limitations:** Fails on textureless surfaces; points can drift over time


---

### 6. `mog_background_subtraction.py`
**Algorithm:** MOG (Mixture of Gaussians) Background Subtraction

**How it works:**
- Models each pixel as a mixture of multiple Gaussian distributions
- Learns the background over time (controlled by `history` parameter)
- Pixels that don't match the background model are classified as foreground
- Result is a binary mask: white = moving object, black = background

**Best for:** Detecting all moving objects in a scene (people, vehicles)
**Limitations:** Struggles with sudden lighting changes, moving shadows, camera shake



##  Algorithm Comparison Cheat Sheet

| Algorithm | Color-Based? | Needs Training? | Tracks Multiple? | Best For |
|-----------|-------------|-----------------|-----------------|----------|
| Frame Diff | ❌ No | ❌ No | ✅ Yes (motion regions) | Simple motion detection |
| HSV Color | ✅ Yes | ❌ No | ✅ Yes (all matching color) | Colored object detection |
| CAMShift | ✅ Yes | ✅ (select once) | ❌ No (single object) | Single colored object tracking |
| Optical Flow | ❌ No | ❌ No | ✅ Yes (many points) | Feature point motion analysis |
| MOG Background | ❌ No | ✅ (learns over time) | ✅ Yes (all moving objects) | Foreground/background separation |

---

## 🔧 Common Troubleshooting

### Camera not opening?
- Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` or `2`
- On Windows, ensure your webcam is not used by another application

### Detection is all black / no results?
1. Check lighting — dim light reduces contrast significantly
2. Lower the threshold values
3. For HSV: widen the `lower` and `upper` range
4. For frame diff: move objects faster across the frame

### Too much noise / false positives?
1. Increase threshold values
2. Add morphological operations:
   ```python
   kernel = np.ones((3, 3), np.uint8)
   mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
   mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
   ```
3. Filter contours by minimum area size

### `cv2.BackgroundSubtractorMOG` not found?
Install the contrib package:
```bash
pip install opencv-contrib-python
```
And use: `cv2.bgsegm.createBackgroundSubtractorMOG()`

---

## 📈 Next Steps to Build Real-World Projects

Once you understand all 6 algorithms, combine them into pipelines:

1. **People Counter** → MOG Background Subtraction + Contour Detection + Virtual Line Crossing
2. **Security Motion Alert** → Frame Difference + Background Subtraction + Auto-save on motion
3. **Colored Object Counter** → HSV Color Detection + Contours + Counting Logic
4. **Persistent Multi-Object Tracker** → Background Subtraction + Detection + CAMShift per object
5. **Speed Estimation** → Optical Flow + Pixel-to-real-world calibration


