# Chapter 2 — Kinect Hand Gesture Recognition

Python 3 implementation of hand gesture recognition using Kinect depth data and convex hull defect analysis.


## How to Run

### Quick test (webcam + OpenCV only)
```bash
python test_gesture_simple.py
```

### Full GUI (wxPython + Kinect)
```bash
python chapter2.py
```
Falls back to webcam if Kinect/freenect is not available.

## Controls

| Key | Action |
|-----|--------|
| `ESC` | Quit |
| `S`   | Save screenshot |

## Dependencies

```bash
pip install numpy opencv-python wxPython
```

For Kinect depth: install `libfreenect` + Python bindings.

## Algorithm

1. **Segment hand** — depth median thresholding + flood fill
2. **Find contours** — largest contour + convex hull
3. **Count fingers** — convexity defects with angle < 80° = finger gaps

## Windows MINGW64 Note


