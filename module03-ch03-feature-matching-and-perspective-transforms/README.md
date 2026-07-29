# ORB Feature Matching & Perspective Warp

Real-time template detection using ORB features, homography, and perspective transformation with wxPython GUI.

## Files
- `chapter3.py` — Main entry point, webcam + wxPython GUI
- `feature_matching.py` — ORB detector, BFMatcher, RANSAC homography, perspective warp
- `salinger.jpg` — Template image (required)

## Install
```bash
pip install opencv-python numpy wxpython
```

## Run
```bash
python chapter3.py
```
Point webcam at `salinger.jpg`. Successful detection:
- Green bounding box drawn on target
- Warped (straightened) view shown in GUI
- Frame auto-saved as `detected_*.png`

## Key Fixes Applied
| Issue | Fix |
|---|---|
| `IndexError: tuple index out of range` | Boundary-validate match indices before accessing keypoints |
| Homography returns `None` | Increased RANSAC threshold (10.0), relaxed match ratio to 0.78 |
| Detection always rejected | Expanded out-of-bounds tolerance; area filter disabled for testing |
| No saved images | `SAVE_RESULTS = True` by default; saves on successful detection |

## Tuning (inside `feature_matching.py`)
- `nfeatures=4000` — more keypoints = slower but more matches
- Lowe ratio `0.78` — lower = stricter, fewer false positives
- RANSAC `10.0` — higher = tolerates more noise
- `SAVE_RESULTS` — toggle auto-save

## Notes
- Uses ORB (patent-free) instead of SURF — works with standard OpenCV 4/5
- `salinger.jpg` must be in the same folder as the scripts
- Saved images appear in your working directory
