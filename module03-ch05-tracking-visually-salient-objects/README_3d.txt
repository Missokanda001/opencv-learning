# 3D Scene Reconstruction from Motion

Two-view stereo 3D reconstruction using SIFT features and epipolar geometry.

## Quick Start

```bash
pip install opencv-python numpy matplotlib
python main.py
```

## Files

| File | Description |
|------|-------------|
| `main.py` | Run this to start |
| `scene3D.py` | Reconstruction engine |
| `img1.jpg` / `img2.jpg` | Input image pair |
| `calibration.npz` | Optional camera calibration |

## Output

- **3D Point Cloud** — interactive plot (drag to rotate)
- **Camera motion** — rotation R + translation T printed in terminal
- **~60 sparse 3D points** (typical for endoscopy images)

## Notes

- No calibration file required (falls back to approximate values)
- Scale is ambiguous — no real-world units
- Both images should be same resolution for epipolar lines to work

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Sparse point cloud | Normal for low-texture surfaces |
| Epipolar lines fail | Resize both images to same dimensions |
| No points found | Ensure images overlap and have texture |
