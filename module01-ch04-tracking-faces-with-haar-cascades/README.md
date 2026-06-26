# OpenCV Chapter 4 – Tracking Faces with Haar Cascades

## Overview

This project implements a real-time face and eye tracking application using OpenCV's Haar Cascade classifiers. The application captures live video from a webcam, detects faces and eyes in real time, applies a portrait-style filter, and supports screenshot capture and video recording through keyboard shortcuts.

This chapter demonstrates the fundamentals of classical computer vision, including object detection, image processing, and interactive camera applications.

---

## Features

* Real-time webcam video capture
* Face detection using Haar Cascade classifiers
* Eye detection within detected faces
* Portrait-style image filtering
* Mirror preview
* Screenshot capture
* Video recording
* Debug mode for visualizing detection rectangles
* Simple keyboard controls

---

## Project Structure

```text
module01-ch04-tracking-faces-with-haar-cascades/
│
├── cameo.py          # Main application
├── managers.py       # Camera, window, screenshot and video management
├── trackers.py       # Face and eye tracking logic
├── filters.py        # Portrait filter implementation
├── rects.py          # Rectangle manipulation utilities
├── utils.py          # Helper utility functions
├── cascades/
│   ├── haarcascade_frontalface_default.xml
│   └── haarcascade_eye.xml
└── README.md
```

---

## Keyboard Controls

| Key              | Action                     |
| ---------------- | -------------------------- |
| **Space**        | Capture a screenshot       |
| **Tab**          | Start/Stop video recording |
| **X**            | Toggle debug rectangles    |
| **Esc** or **Q** | Quit the application       |

---

## How It Works

The application performs the following pipeline:

1. Capture frames from the webcam.
2. Apply a portrait-style image filter.
3. Detect faces using a Haar Cascade classifier.
4. Detect eyes inside each detected face.
5. Draw detection rectangles (optional debug mode).
6. Display the processed video stream.
7. Save screenshots or record video when requested.

---

## Technologies Used

* Python 3
* OpenCV
* NumPy

---

## Learning Outcomes

Through this project I learned how to:

* Capture live video streams using OpenCV
* Manage webcam input efficiently
* Detect faces and eyes using Haar Cascade classifiers
* Apply real-time image filters
* Draw bounding boxes around detected objects
* Capture screenshots programmatically
* Record live video streams
* Organize a computer vision project into reusable modules

---

## Example Output

The application can:

* Detect faces in real time
* Detect eyes within the face region
* Apply a portrait/cartoon filter
* Save screenshots
* Record webcam videos


