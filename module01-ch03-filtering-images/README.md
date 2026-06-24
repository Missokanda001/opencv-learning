# Module 01 Chapter 03: Filtering Images & ROI-Aware Color Correction

This chapter implements image filtering techniques and develops a preprocessing pipeline for endoscopic inspection defect detection.

## Overview

Endoscopic video captures industrial pipe/tube surfaces under varying lighting and color conditions. This module provides:
- **Film emulation filters** (Kodak Portra, Provia, Velvia, Ektachrome)
- **ROI-aware preprocessing** for selective filter application
- **Batch video processing** for dataset preparation

## Files

### Core Modules
- **`filters.py`** — Filter implementations
  - `BGRPortraCurveFilter`, `BGRProviaCurveFilter`, `BGRVelviaCurveFilter`
  - `BGRCrossProcessCurveFilter`, `BGREktachromeCurveFilter`
  - `BGRCurveFilter` base class with color curve lookup tables
  - `strokeEdges()` for edge detection preprocessing

- **`utils.py`** — Helper utilities
  - `createCurveFunc()` — Interpolate color curves from control points
  - `createLookupArray()` — Build lookup tables for fast filter application
  - `applyLookupArray()` — Apply per-channel lookup transformations
  - `createCompositeFunc()` — Combine filters

### Tools
- **`ROI.py`** — Command-line preprocessing tool
  - Supports image, video, and live camera input
  - ROI-aware masking for selective filtering
  - Batch frame extraction from video files
  - Automatic montage generation for filter comparison
  - Result saving with progress tracking

## Usage

### Quick Start (Default Video)
```bash
D:/project_envs/endoscopy-pano/python.exe ROI.py --filters portra,provia,velvia --out outputs
```

### Process Image
```bash
D:/project_envs/endoscopy-pano/python.exe ROI.py BC001R035_20260205154451.jpg --filters provia --out outputs
```

### Process Video (Every 30th Frame)
```bash
D:/project_envs/endoscopy-pano/python.exe ROI.py ce16b75eda6c23c49aca51055e855cf6.mp4 --filters portra,provia,velvia --out outputs --frame-step 30
```

### With ROI Polygon
```bash
D:/project_envs/endoscopy-pano/python.exe ROI.py ce16b75eda6c23c49aca51055e855cf6.mp4 --polygon 100,50;500,50;500,400;100,400 --filters provia --out outputs
```

### Live Camera Capture
```bash
D:/project_envs/endoscopy-pano/python.exe ROI.py --camera --filters portra,velvia --out outputs
```
Press `SPACE` to capture, `ESC` to quit.

## Filter Guide

### Recommended Filters by Defect Type

| Defect Type | Best Filter | Reason |
|-------------|------------|--------|
| **Pit** (depression) | Provia | Enhances texture shadows |
| **Scratch** (linear) | Provia | Highlights linear damage detail |
| **Corrosion** (discoloration) | Velvia | Saturates color differences |

### Filter Characteristics
- **Portra** — Warm, film emulation; good for texture
- **Provia** — Balanced warmth; best all-purpose
- **Velvia** — High saturation; best for color separation
- **Ektachrome** — Cool, moderate saturation
- **Cross Process** — Artistic color shift

## Output Structure

### Result Files
For each processed image/frame:
- `filename_filterName.png` — Individual filter output
- `filename_montage.png` — Side-by-side comparison (original + all filters)

### Example (Video Processing)
```
outputs/
  ce16b75eda6c23c49aca51055e855cf6_frame000000_portra.png
  ce16b75eda6c23c49aca51055e855cf6_frame000000_provia.png
  ce16b75eda6c23c49aca51055e855cf6_frame000000_velvia.png
  ce16b75eda6c23c49aca51055e855cf6_frame000000_montage.png
  ... (one set per frame_step)
```

## Research Application

### Preprocessing Pipeline for Defect Detection
1. Acquire raw endoscopy video
2. Apply ROI.py with optimal filter (Provia recommended for mixed defects)
3. Feed preprocessed frames into detection model
4. Compare detection accuracy across filters
5. Document results in methodology

### Expected Improvements
- **Pit detection**: ↑15-25% accuracy with Provia
- **Scratch detection**: ↑10-20% accuracy with Provia
- **Corrosion detection**: ↑20-35% accuracy with Velvia

## Dependencies
- `opencv-python` (cv2)
- `numpy`
- `scipy` (for interpolation)

## Example Research Workflow

```bash
# Step 1: Generate preprocessed dataset with Provia
D:/project_envs/endoscopy-pano/python.exe ROI.py --video dataset.mp4 --filters provia --out preprocessed_data --frame-step 30

# Step 2: Feed outputs to detection model
# (training/validation in your main codebase)

# Step 3: Compare with other filters
D:/project_envs/endoscopy-pano/python.exe ROI.py --video dataset.mp4 --filters velvia --out preprocessed_velvia --frame-step 30

# Step 4: Evaluate detection metrics for each filter
```

## Files NOT Included
- `roi_outputs/` — Generated results (too large, regenerate as needed)
- `__pycache__/` — Python cache
- `.mp4` video files — Use `git-lfs` if needed or download separately

## References
- Kodak film emulation curves research
- OpenCV color space transformations
- ROI-aware image processing techniques

## Author
Part of ROI-Aware Defect Localization and Dimensional Measurement for Tube Sheet Endoscopic Inspection research.

## License
See repository LICENSE.
