# Module 02 - Chapter 05: Creating a Panoramic Image

This chapter covers **feature-based image stitching** — the core technique for building panoramic images from multiple overlapping frames.

---

## 📁 Chapter Structure

```
module02-ch05-creating-a-panoramic-image/
├── README.md                    # This file
├── feature_match.py             # ORB feature matching with Lowe's ratio test
├── stitch.py                    # Full SIFT + FLANN + RANSAC panorama stitcher
└── images/
    ├── img1.jpg                 # Query / first image
    └── img2.jpg                 # Train / second image
```



## 🧪 Scripts

### 1. `feature_match.py` — ORB Feature Matching
Demonstrates brute-force ORB feature matching between two images, with **Lowe's ratio test** to filter out false positive matches.

#### Important keypoints:
- ORB keypoint detection and descriptor extraction
- Brute-force matcher with Hamming distance (for binary descriptors)
- k-NN matching + Lowe's ratio test to remove ambiguous correspondences
- Visualizing matched keypoints between image pairs

#### Run:
```bash
python feature_match.py
```
> Edit the image paths inside the script (`images/img1.jpg`, `images/img2.jpg`) before running.

---

### 2. `stitch.py` — Full Panorama Stitching (SIFT + FLANN + RANSAC)
Complete image stitching pipeline that warps and merges two overlapping images into a single panorama.

#### Pipeline steps:
1. **Detect SIFT keypoints** on both images
2. **Extract SIFT descriptors** (128D floating-point)
3. **FLANN-based k-NN matching** — fast approximate nearest neighbor search
4. **Lowe's ratio test** — filter unreliable matches
5. **RANSAC Homography estimation** — find the perspective transformation matrix, reject outliers
6. **Warp & blend** — warp the second image onto the first and produce the final stitched panorama

#### Run with command line arguments:
```bash
python stitch.py --query-image images/img1.jpg --train-image images/img2.jpg
```

#### Optional parameter:
```bash
python stitch.py --query-image images/img1.jpg --train-image images/img2.jpg --min-match-count 15
```
- `--min-match-count`: Minimum number of good matches required before attempting stitching (default: 10)

---

## 🧠 Key Concepts Covered

### Homography Matrix
A 3×3 transformation matrix that maps points from one image plane to another. It describes how a scene viewed from one camera perspective appears from another viewpoint.

```
[x']   [ h11 h12 h13 ] [x]
[y'] = [ h21 h22 h23 ] [y]
[w']   [ h31 h32 h33 ] [1]
```

### RANSAC (Random Sample Consensus)
An iterative outlier rejection algorithm:
1. Randomly select 4 matching point pairs
2. Compute a candidate homography
3. Count how many matches fit (inliers)
4. Repeat — keep the homography with the most inliers

This is critical for robust stitching when many feature matches are incorrect.

### FLANN Matcher
Fast Library for Approximate Nearest Neighbors — much faster than brute-force matching when working with large numbers of SIFT descriptors. Uses KD-tree index structure.

### Lowe's Ratio Test
For each descriptor, find the top-2 closest matches. Keep the match only if the best match is significantly better than the second-best.
- Ratio < 0.7 → good match (unambiguous)
- Ratio ≥ 0.7 → ambiguous, discard

---

## ⚠️ Common Issues & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `error: the following arguments are required` | Running `stitch.py` without arguments | Pass `--query-image` and `--train-image` paths, or hardcode paths |
| `'NoneType' object has no attribute 'shape'` | Image files not found | Verify file paths; images must be in `images/` folder |
| Not enough matches message | Images have little overlap or few features | Increase image overlap, lower `min_match_count`, or use different images |
| `cv2.SIFT()` AttributeError | Old OpenCV API | Use `cv2.SIFT_create()` (fixed in this chapter's code) |

---

## 📊 Algorithm Comparison for Stitching

| Detector | Descriptor Type | Matcher | Speed | Accuracy | Patent Status |
|----------|----------------|---------|-------|----------|---------------|
| SIFT | 128D float | FLANN | Slow | High | Expired (2020) |
| ORB | 256-bit binary | Brute Force (Hamming) | Fast | Good | Free |
| SURF | 64D float | FLANN | Medium | High | ⚠️ Patented |

> For your real-time endoscopic stitching, consider **ORB + Brute Force** as a faster patent-free alternative to SIFT+FLANN.

