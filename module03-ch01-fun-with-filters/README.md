# Fun with Filters — OpenCV Real-Time Filter GUI

Real-time webcam filter application built with **OpenCV** + **wxPython**.
Apply four artistic filters to live camera feed with a single click.

---

## ✨ Features

| Filter | Effect |
|--------|--------|
| **Warming Filter** | Boosts red tones + saturation for a warm, sunny look |
| **Cooling Filter** | Boosts blue tones for a cool, moody look |
| **Pencil Sketch** | Converts image to hand-drawn pencil sketch with paper texture |
| **Cartoonizer** | Bilateral smoothing + edge detection for cartoon/comic style |

---


## 🎮 How to Use

1. A window opens with live webcam feed
2. Click any radio button at the bottom to switch filters
3. Close the window to exit

> Webcam index defaults to **`1`**. If camera doesn't open, change to `0` in `chapter1.py` → `cv2.VideoCapture(1)`.

---

## 🧠 Key Concepts

- **Lookup Tables (LUT)**: Fast tone-curve mapping via `cv2.LUT` for color filters
- **Bilateral Filter**: Smooths colors while preserving edges (cartoon effect)
- **Adaptive Thresholding**: Detects edges for sketch / cartoon outlines
- **Gaussian Pyramid**: `pyrDown` / `pyrUp` for fast down/up sampling
- **wx.Timer**: Drives the ~30 FPS live frame refresh loop

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| Black screen / no video | Try `VideoCapture(0)` or `VideoCapture(2)` |
| `ModuleNotFoundError: No module named 'wx'` | `pip install wxpython` |
| Pencil sketch looks flat | Confirm `pencilsketch_bg.jpg` path is correct |
| `spline fit error (m>k)` | Uses `k=2` (quadratic) instead of default cubic |

---

## 📌 Required Files

- `pencilsketch_bg.jpg` — Paper texture overlay for Pencil Sketch filter
