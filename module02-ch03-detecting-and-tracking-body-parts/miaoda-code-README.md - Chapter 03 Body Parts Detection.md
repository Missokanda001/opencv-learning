# Module 02 - Chapter 03: Detecting and Tracking Body Parts

OpenCV Python tutorial chapter covering face, eye, ear, mouth, nose, and pupil detection using Haar Cascades and contour analysis.


## Overview

This chapter explores **Haar Cascade Classifiers** for real-time body part detection on live webcam feed. Each script builds on the previous one, progressing from simple rectangle detection to creative image overlays (filters).

**Concepts covered:**
- Haar Cascade Classifier loading and usage
- Face detection
- Eye detection within face ROI (Region of Interest)
- Ear detection (left / right)
- Mouth detection
- Nose detection
- Pupil detection via contour analysis
- Image overlay / alpha blending with threshold masks
- Real-time webcam processing

---

## Prerequisites

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy

Install dependencies:
```bash
pip install opencv-python numpy
```

> **Note:** OpenCV 4.x is recommended. OpenCV 5.x pre-release builds may have API changes that break cascade detection.

---

## Project Structure

```
module02-ch03-detecting-and-tracking-body-parts/
│
├── cascades/                          # Haar cascade XML files
│   ├── haarcascade_frontalface_alt.xml
│   ├── haarcascade_eye.xml
│   ├── haarcascade_mcs_leftear.xml
│   ├── haarcascade_mcs_rightear.xml
│   ├── haarcascade_mcs_mouth.xml
│   └── haarcascade_mcs_nose.xml
│
├── mask_hannibal.png                  # Hannibal Lecter mask overlay
├── sunglasses.jpg                     # Sunglasses overlay image
├── moustache.png                      # Moustache overlay image
│
├── 01_face_detector.py                # Basic face detection + rectangles
├── 02_face_mask_overlay.py            # Face detection + Hannibal mask overlay
├── 03_eye_detector.py                 # Eye detection within face ROI
├── 04_sunglasses_overlay.py           # Eye detection + sunglasses filter
├── 05_ear_detector.py                 # Left/right ear detection
├── 06_mouth_detector.py               # Mouth detection + moustache overlay
├── 07_nose_detector.py                # Nose detection
└── 08_pupil_detector.py               # Pupil detection via contour analysis
```

---

## Scripts

### 01 - Face Detector
**File:** `01_face_detector.py`

Detects faces in live webcam feed and draws green rectangles around them.
- Uses `haarcascade_frontalface_alt.xml`
- Basic Haar cascade example

### 02 - Face Mask Overlay (Hannibal)
**File:** `02_face_mask_overlay.py`

Detects a face and overlays a Hannibal Lecter mask on it using threshold-based alpha blending.
- Requires: `mask_hannibal.png`
- Uses bitwise AND operations for mask blending

### 03 - Eye Detector
**File:** `03_eye_detector.py`

Detects face first, then finds eyes within the face region. Draws green circles around detected eyes.
- Uses `haarcascade_eye.xml`
- Demonstrates nested ROI detection

### 04 - Sunglasses Overlay
**File:** `04_sunglasses_overlay.py`

Detects both eyes and overlays a pair of sunglasses sized to match the distance between eyes.
- Requires: `sunglasses.jpg`
- Calculates sunglass width from inter-pupillary distance

### 05 - Ear Detector
**File:** `05_ear_detector.py`

Detects left and right ears separately.
- Uses `haarcascade_mcs_leftear.xml` and `haarcascade_mcs_rightear.xml`
- Green = left ear, Blue = right ear

### 06 - Mouth Detector + Moustache
**File:** `06_mouth_detector.py`

Detects mouth region and overlays a moustache image.
- Uses `haarcascade_mcs_mouth.xml`
- Requires: `moustache.png`

### 07 - Nose Detector
**File:** `07_nose_detector.py`

Detects nose using Haar cascade.
- Uses `haarcascade_mcs_nose.xml`

### 08 - Pupil Detector
**File:** `08_pupil_detector.py`

Detects pupils using contour analysis (not Haar cascade). Finds dark circular shapes matching pupil geometry (area, symmetry, fill ratio).
- Uses `cv2.findContours` and geometric filtering
- Works best with controlled lighting

---

## Keyboard Shortcuts

All webcam scripts share the same keyboard controls:

| Key | Action |
|-----|--------|
| `ESC` | Exit the program |
| `SPACE` | Pause / resume the webcam feed |
| `s` | Save current frame as a timestamped PNG snapshot |

Saved snapshots are stored in the same folder as the script with names like:
`face_snapshot_20260724_143025.png`

---

## Setup & Installation



1. **Install Python packages:**
   ```bash
   pip install opencv-python numpy
   ```

2. **Verify your camera index:**
   The default camera index is `1` (USB webcam). If you get a black screen or camera error, try changing `cv2.VideoCapture(1)` to `cv2.VideoCapture(0)` or `cv2.VideoCapture(2)`.

3. **Run any script:**
   ```bash
   python 01_face_detector.py
   ```

---

## Asset Files

### Cascade XML Files
The following cascades are **built into OpenCV** and loaded automatically via `cv2.data.haarcascades`:
- `haarcascade_frontalface_alt.xml`
- `haarcascade_eye.xml`

The following MCS cascades are **not included** in standard OpenCV and must be placed in the `cascades/` folder:
- `haarcascade_mcs_leftear.xml`
- `haarcascade_mcs_rightear.xml`
- `haarcascade_mcs_mouth.xml`
- `haarcascade_mcs_nose.xml`

> ⚠️ **Note:** The MCS (Multi-Camera System) cascade files are deprecated and removed from the official OpenCV repository. You may need to find them in older tutorial repositories or skip those exercises. Scripts with missing cascades will print a warning and run without that feature.

### Overlay Images
Place these images in the project root folder:
- `mask_hannibal.png` — Hannibal Lecter face mask (white background recommended)
- `sunglasses.jpg` — Sunglasses image (white background recommended)
- `moustache.png` — Moustache image (white background recommended)

> 💡 **Tip:** Overlay images with **plain white backgrounds** work best with the threshold-based blending technique used in these scripts.

---

## Troubleshooting

### Camera not opening / black screen
- Try changing the camera index: `cv2.VideoCapture(0)`, `cv2.VideoCapture(1)`, or `cv2.VideoCapture(2)`
- Make sure your USB webcam is properly connected

### "Unable to load cascade classifier" error
- Check that the XML file exists in the `cascades/` folder
- Verify the filename spelling matches exactly (case-sensitive on Linux)
- For face/eye cascades, use the built-in `cv2.data.haarcascades` path

### Overlay image not found
- Make sure the image file is in the **same folder** as the Python script
- Check for double extensions (e.g., `mask_hannibal.png.png` when Windows hides extensions)

### Detection is inaccurate / false positives
- Adjust the `scaleFactor` (1.3) and `minNeighbors` (5) parameters in `detectMultiScale()`
- Ensure proper lighting (well-lit face, even lighting)
- Face the camera directly for best Haar cascade results

### Pupil detection finds wrong objects
- The basic pupil script searches the entire frame — use the improved version that restricts search to eye ROI only
- Adjust area range and threshold values based on your lighting conditions


