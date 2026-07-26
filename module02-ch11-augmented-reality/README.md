# Augmented Reality Tracker

A simple OpenCV-based augmented reality demo that overlays a 3D pyramid on a tracked planar target using feature matching and pose estimation.

---

## Files

| File | Description |
|------|-------------|
| `pose_estimation.py` | Core module: `PoseEstimator`, `ROISelector`, `VideoHandler` |
| `ar_tracker.py` | Main AR script: loads camera/image and draws the 3D pyramid overlay |
| `test.jpg` | Test image (optional — for static image mode) |



## How to Run

### Mode 1: Webcam (default)

```bash
python ar_tracker.py
```

### Mode 2: Static Image

Open `ar_tracker.py` and change:
```python
USE_STATIC_IMAGE = True
```
Then run:
```bash
python ar_tracker.py
```

---

## How to Use

1. Run the program — a window titled "Augmented Reality" opens.
2. **Click and drag** with your mouse to draw a rectangle around a textured flat object (e.g., a book cover, printed photo, or poster).
3. Release the mouse — the tracker will lock onto the target and overlay a 3D pyramid on top of it.
4. Move the object (or your camera) around — the pyramid follows the target.
5. Press **S** to save a screenshot.
6. Press **ESC** to exit.

> 💡 **Tip**: The target needs texture (details, patterns, text). A blank white sheet won't work — there are no features to track.

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Pause / resume |
| `C` | Clear targets and select a new one |
| `S` | Save screenshot (saved to the same folder) |
| `ESC` | Exit |

---

## How It Works

1. **Feature Detection** — Finds unique keypoints (ORB features) in your selected region.
2. **Feature Matching** — Matches those keypoints in every new frame using FLANN.
3. **Homography** — Calculates how the target has moved/rotated using `cv2.findHomography`.
4. **Pose Estimation** — Uses `cv2.solvePnP` to find the 3D camera position relative to the target.
5. **Projection** — Projects a 3D pyramid onto the 2D image using `cv2.projectPoints`.
6. **Drawing** — Renders the pyramid on top of the real camera feed → augmented reality!

---

## Saved Screenshots

Press `S` at any time to save the current frame with the AR overlay.


