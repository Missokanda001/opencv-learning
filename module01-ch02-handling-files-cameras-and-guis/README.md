# Module 01 - Chapter 02: Handling files, cameras and GUIs

This chapter demonstrates OpenCV basics for working with files, capturing camera input, and building a simple GUI preview app.

## What’s included

- `managers.py`
  - live camera preview with `WindowManager` and `CaptureManager`
  - screenshot capture using `SPACE` or `S`
  - video recording start/stop using `TAB` or `R`
  - quit with `ESC` or `Q`
  - on-screen status overlay for feedback

- `test_opencv.py`
  - image loading and display examples
  - grayscale conversion
  - random image buffer creation
  - video file read/write example
  - simple camera capture helper

## What I learned

- how to open and display images with `cv2.imread` and `cv2.imshow`
- how to read/write video frames with `cv2.VideoCapture` and `cv2.VideoWriter`
- how to handle keyboard input using `cv2.waitKey`
- how to store screenshots and screencasts on demand
- how to structure reusable helper classes for camera and window management

## How to run the app

From this folder, run:

```bash
python managers.py
```

Then use:

- `SPACE` or `S` to take a screenshot
- `TAB` or `R` to start/stop recording
- `ESC` or `Q` to quit


