# OpenCV Feature Detectors & Descriptors

A collection of working OpenCV feature detection and descriptor extraction algorithms — tested and updated for **OpenCV 5.x** compatibility. 


## 📁 Project Structure

```
.
├── README.md
├── sift.py              # SIFT detector + descriptor
├── surf.py              # SURF detector + descriptor (patented, see notes)
├── fast.py              # FAST corner detector (with/without non-max suppression)
├── brief.py             # FAST + BRIEF descriptor extractor
├── orb.py               # ORB detector + descriptor (recommended)
└── input.jpg            # Sample test image
```

---

##  Installation

### Prerequisites
- Python 3.8+
- OpenCV 4.x / 5.x
- NumPy

### Setup
```bash
# Install base OpenCV
pip install opencv-python

# Install contrib modules (required for BRIEF, SIFT legacy, SURF)
pip install opencv-contrib-python

# Install NumPy
pip install numpy
```

---

## 🧠 Algorithms Overview

### 1. SIFT (Scale-Invariant Feature Transform)
- **File**: `sift.py`
- **Type**: Detector + Descriptor (128D float)
- **Properties**: Scale-invariant, rotation-invariant, highly robust
- **Use case**: High-accuracy feature matching, panorama stitching
- **Status**: Patent expired (2020), works on modern OpenCV

```bash
python sift.py
```

### 2. SURF (Speeded Up Robust Features)
- **File**: `surf.py`
- **Type**: Detector + Descriptor (64D float)
- **Properties**: Faster than SIFT, scale/rotation invariant
- **Use case**: Real-time feature matching
- **Status**: ⚠️ **PATENTED — will NOT run on standard pip installs**

> ⚠️ **SURF Limitation**: Prebuilt `opencv-contrib-python` binaries disable SURF by default due to patent restrictions. To execute SURF, you must compile OpenCV from source with `OPENCV_ENABLE_NONFREE=ON`.
>
> For immediate cross-platform execution, use **ORB** as a patent-free alternative.

```bash
# Will fail on standard install — see note above
python surf.py
```

### 3. FAST (Features from Accelerated Segment Test)
- **File**: `fast.py`
- **Type**: Detector only (no built-in descriptor)
- **Properties**: Extremely fast corner detection
- **Use case**: Real-time video keypoint detection
- **Status**: Works immediately, no patent issues

```bash
python fast.py
```

The script demonstrates both modes:
- **With non-max suppression** (default): Cleaner, sparse keypoints
- **Without non-max suppression**: Dense, clustered keypoints

### 4. FAST + BRIEF (Binary Robust Independent Elementary Features)
- **File**: `brief.py`
- **Type**: FAST detector + BRIEF binary descriptor
- **Properties**: Very fast, compact binary descriptors
- **Limitation**: NOT rotation-invariant
- **Use case**: Lightweight real-time matching under stable camera orientation
- **Status**: Works immediately with `opencv-contrib-python`

```bash
python brief.py
```

### 5. ORB (Oriented FAST and Rotated BRIEF)
- **File**: `orb.py`
- **Type**: Detector + Descriptor (256-bit binary)
- **Properties**: Free, open-source, rotation-invariant, scale-invariant
- **Use case**: ✅ **Recommended default** for panorama stitching, SLAM, feature matching
- **Status**: Works immediately, no patent issues

```bash
python orb.py
```

---

## 📊 Algorithm Comparison

| Algorithm | Detector | Descriptor | Rotation Invariant | Scale Invariant | Speed | Patent Status |
|-----------|----------|------------|--------------------|-----------------|-------|---------------|
| SIFT | ✅ | 128D float | ✅ | ✅ | Slow | Expired (2020) |
| SURF | ✅ | 64D float | ✅ | ✅ | Medium | ⚠️ Patented |
| FAST | ✅ | None | ❌ | ❌ | Very Fast | Free |
| FAST+BRIEF | ✅ | Binary | ❌ | ❌ | Very Fast | Free |
| ORB | ✅ | Binary | ✅ | ✅ | Fast | Free |



## 🛠️ OpenCV 5.x Migration Notes

These scripts have been updated for OpenCV 4.x / 5.x API changes:

| Old (Deprecated) | New (Working) |
|------------------|---------------|
| `cv2.SIFT()` | `cv2.SIFT_create()` |
| `cv2.SURF()` | `cv2.xfeatures2d.SURF_create()` |
| `cv2.ORB()` | `cv2.ORB_create()` |
| `cv2.FastFeatureDetector()` | `cv2.FastFeatureDetector_create()` |
| `cv2.DescriptorExtractor_create("BRIEF")` | `cv2.xfeatures2d.BriefDescriptorExtractor_create()` |
| `fast.setBool('nonmaxSuppression', False)` | `cv2.FastFeatureDetector_create(nonmaxSuppression=False)` |
| `cv2.drawKeypoints(img, kp, color=...)` | `cv2.drawKeypoints(img, kp, None, color=...)` |

---

## 📝 License

This code is provided for educational and research purposes.

- **SURF**: Patented algorithm — ensure compliance with applicable patent laws before commercial use.
- **ORB, FAST, BRIEF, SIFT**: Free for research and commercial use (verify SIFT patent status in your jurisdiction).

---

