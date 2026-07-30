# Chapter 6: Multi-Class Traffic Sign Classification with SVM

> Python 3 + Modern OpenCV implementation of traffic sign classification using
> Support Vector Machines (SVM) with multiple feature extraction methods.

---

## 📖 What This Project Does

This is a complete **traffic sign classification** system. It takes images of
road signs and automatically identifies what kind of sign it is (stop sign,
speed limit, yield, warning, etc.).

### The Big Picture: How It Works

```
Camera Image → Crop Sign → Extract Features → SVM Classifier → Predicted Sign
```

1. **Input**: An image containing a traffic sign
2. **Feature Extraction**: Convert the raw image into a numerical "feature
   vector" that captures the important visual patterns (edges, shapes, colors)
3. **SVM Classification**: A trained Support Vector Machine looks at the
   features and decides which class (sign type) it belongs to
4. **Output**: The predicted sign class

### Real-World Connection: How This Relates to Actual Traffic Infrastructure

**Important clarification**: This code is about **traffic sign recognition**
(reading road signs with a camera), **not** traffic light control.

#### Where you find this in real life:
- **Self-driving cars / ADAS**: A camera on the windshield reads speed limit
  signs, stop signs, and warning signs to help the car understand the road.
- **Driver assistance**: The car can warn you if you're speeding or missed a
  stop sign.
- **Road inventory**: Governments use camera-equipped vans to automatically
  catalog all road signs.

#### Does this code control physical traffic lights?
**No.** This is a *perception* system — it *sees* signs. To actually control
traffic lights, you would need a completely different system:
1. Cameras mounted on traffic light poles watching the road
2. Vehicle detection and counting
3. A separate control system that sends electrical signals to the light
   hardware (relays, controllers)
4. Strict safety certification

If you wanted to connect *this* classifier to physical hardware (e.g., a demo
with an LED), the chain would be:
```
Webcam → Object Detection → Sign Crop → Feature Extraction → SVM Predict
                                                              ↓
                                                Send signal to LED/actuator
```

---

## 📁 File Structure



After running the generator, you'll also get:
```
datasets/
├── 00000/                           # Class 0: Red circle (prohibition)
│   ├── GT-00000.csv                 # Annotations
│   └── 00000_00000.ppm ...          # Images
├── 00001/                           # Class 1: Red triangle (warning)
├── 00002/                           # Class 2: Blue circle (mandatory)
├── 00003/                           # Class 3: Red octagon (stop)
└── 00004/                           # Class 4: Yellow diamond (priority)
```


### Optional: SURF features
SURF is patented and lives in `opencv-contrib-python`:
```bash
pip install opencv-contrib-python
```
If you don't install it, the code will automatically fall back to HOG features
when SURF is request. 

This creates 5 classes × 50 images = 250 synthetic traffic sign images in
`datasets/`.


This tests all feature types (GRAY, RGB, HSV, HOG) with both SVM strategies
(one-vs-all, one-vs-one) and shows a comparison bar chart.


### Train with specific configuration
```bash
python chapter6.py --mode train --feature hog --svm-mode one-vs-all --dataset datasets
```

### Use your own dataset
Place your GTSRB-formatted data in a directory and point to it:
```bash
python chapter6.py --dataset /path/to/your/GTSRB
```

---

## ⌨️ Keyboard Shortcuts

All plot windows support these shortcuts:

| Key | Action |
|-----|--------|
| `ESC` | Close the window |
| `S` | Save the figure as a PNG file |

---

##  Architecture

### 1. `Classifier` — Abstract Base Class
Defines the interface that all classifiers must follow:
- `fit(X_train, y_train)` — Train on labeled data
- `evaluate(X_test, y_test, visualize)` — Test and return metrics

This is the **Strategy Pattern**: you can swap in any classifier (SVM, KNN,
CNN, etc.) and the rest of the code doesn't need to change.

### 2. `MultiClassSVM` — The Heart of the System

SVMs are **binary classifiers** — they can only say "this is class A or class
B." To handle multiple sign types, we use one of two strategies:

#### One-vs-All (OvA)
- Train **K** SVMs (one per class)
- Each SVM learns: "is this Class C, or is it something else?"
- At test time: pick the class whose SVM gives the strongest "yes"
- **Pros**: Fewer classifiers, simpler
- **Cons**: Imbalanced training data

#### One-vs-One (OvO)
- Train **K×(K-1)/2** SVMs (one per pair of classes)
- Each SVM learns: "is this Class A or Class B?"
- At test time: every SVM votes, majority wins
- **Pros**: Often more accurate, balanced training
- **Cons**: More classifiers, slower with many classes

### 3. Feature Extractors

| Feature | What It Captures | Best For |
|---------|-----------------|----------|
| `raw` | Raw pixel values | Baseline comparison |
| `gray` | Grayscale pixels | Shape + brightness |
| `rgb` | Color pixel values | Color + shape |
| `hsv` | Hue, Saturation, Value | Color robustness |
| `hog` | Histogram of Oriented Gradients | **Edge shapes (best overall)** |
| `surf` | Speeded Up Robust Features | Interest point descriptors |

**HOG is usually the best** for traffic signs because it captures shape and
edge information, which is exactly what distinguishes different signs.

### 4. Data Loader (`load_data`)
Reads images in GTSRB format:
- Iterates over numbered class directories
- Reads CSV annotation files (`GT-xxxxx.csv`)
- Crops to Region of Interest (ROI)
- Resizes to fixed dimensions
- Extracts features
- Splits into train/test sets

---



### CSV format (semicolon-separated)
```
Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId
00000_00000.ppm;64;64;6;6;58;58;0
```

Fields:
- **Filename**: Image file name
- **Width/Height**: Image dimensions
- **Roi.X1, Roi.Y1**: Top-left corner of the sign bounding box
- **Roi.X2, Roi.Y2**: Bottom-right corner of the sign bounding box
- **ClassId**: The class label (integer)

---

## 🧠 How Training and Evaluation Work

### Training Phase (`fit`)
1. Load training images and their labels
2. Extract features from each image
3. For each binary SVM:
   - Create binary labels (positive = target class, negative = everything else)
   - Train the SVM to find the best separating hyperplane
4. Store all trained SVMs

### Evaluation Phase (`evaluate`)
1. Load test images (never seen during training!)
2. Extract features
3. Run all SVMs and aggregate predictions (voting)
4. Compare predictions to true labels
5. Compute metrics:
   - **Accuracy**: Overall fraction correct
   - **Precision**: Of all "Class C" predictions, how many were actually Class C?
   - **Recall**: Of all true Class C samples, how many did we find?
   - **Confusion Matrix**: Shows which classes get confused with which

---

## 🔧 Using Real GTSRB Data

When you're ready for the real dataset:

1. Download GTSRB from:
   https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/

2. Extract the training set into a directory
3. Point the script at it:
   ```bash
   python chapter6.py --dataset /path/to/GTSRB/Final_Training/Images
   ```

The full GTSRB has **43 classes** and ~39,000 training images.

---


### SVM training is slow
- Reduce the number of samples
- Use smaller images
- Use `one-vs-all` instead of `one-vs-one` (fewer classifiers)
- Use a simpler feature (e.g., `gray` instead of `hog`)

### Low accuracy with synthetic data
That's expected! The synthetic dataset is simple and designed for testing the
*pipeline*, not achieving state-of-the-art accuracy. Real GTSRB data with HOG
features typically achieves 90%+ accuracy with SVM.

---

## 📚 Learning Path for Beginners

1. **Start here**: Run the synthetic dataset demo — understand the pipeline
2. **Experiment**: Try different features and SVM modes, see how accuracy changes
3. **Go deeper**: Read the code for `MultiClassSVM` — understand one-vs-all vs one-vs-one
4. **Scale up**: Download real GTSRB data and train on that
5. **Next steps**: Try a CNN (Convolutional Neural Network) — it will beat SVM on this task

---

## 📝 Notes

- This is a **modern Python 3 + OpenCV 4.x** rewrite of the classic
  OpenCV-Python book's Chapter 6 code.
- The original code used Python 2 syntax (`xrange`, `print` statement,
  `cv2.SVM()`, etc.) which no longer works.
- All interactive windows have `ESC` to close and `S` to save, per standard
  project conventions.
- Output images (confusion matrix, comparison charts) are auto-saved when you
  press `S` in the plot window.
