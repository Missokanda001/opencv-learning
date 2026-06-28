# Detecting Foreground/Background Regions and Depth

This module demonstrates a small OpenCV application that combines:
- face tracking with Haar cascades
- foreground/background rectangle swapping
- depth masking using OpenNI-compatible depth cameras
- image filtering and visual debugging

## Goal
This folder supports the learning objectives for Module 1, Chapters 02 through 05:
- Chapter 02: Handling files, cameras, and GUIs
- Chapter 03: Filtering images and applying basic image transformations
- Chapter 04: Tracking faces with Haar cascades
- Chapter 05: Detecting foreground/background regions and using depth information

The project is designed to build practical experience with OpenCV, NumPy, and object-oriented Python code.

## Files
- `cameo.py` - main application and camera modes
- `depth.py` - depth-camera utilities and mask creation
- `rects.py` - rectangle copy and swap helpers
- `filters.py` - image filter implementations
- `managers.py` - capture and window management helpers
- `trackers.py` - face tracking utilities
- `utils.py` - utility functions used by the module
- `cascades/` - Haar cascade XML files for face detection

## Usage
Run one of the application modes from this directory:

```bash
python cameo.py
```

By default `cameo.py` starts in depth mode. You can switch to other modes by editing the bottom of `cameo.py`:

- `Cameo().run()` - single RGB camera mode
- `CameoDouble().run()` - dual RGB camera mode
- `CameoDepth().run()` - depth camera mode

## Depth Camera Note
The depth mode requires an OpenNI-compatible depth sensor such as:
- Microsoft Kinect v1
- Asus Xtion Pro Live
- other OpenNI-compatible depth cameras

If no depth camera is connected, the program will print:
`Error: Depth camera could not be opened.`

## What this module teaches
- working with camera input and camera channels
- detecting and tracking faces in live video
- copying and swapping image regions
- applying depth-based masks
- integrating OpenCV and NumPy in GUI applications

## Tips
- Use `space` to save a screenshot if the application supports it
- Use `tab` to start/stop recording if video writing is available
- Use `x` to toggle debug rectangles
- Use `Esc` or `q` to quit
