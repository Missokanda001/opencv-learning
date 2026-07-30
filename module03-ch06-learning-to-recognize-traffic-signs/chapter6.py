"""
Chapter 6: Multi-Class Traffic Sign Classification with SVM

This module implements a complete traffic sign classification pipeline:
  - Abstract Classifier base class
  - MultiClassSVM (one-vs-one and one-vs-all strategies)
  - Multiple feature extractors (HOG, SURF, raw pixels, grayscale, RGB, HSV)
  - GTSRB-compatible data loader
  - Training, evaluation, and visualization

Python 3 + Modern OpenCV compatible.

Usage:
    python chapter6.py                      # Run full demo with synthetic data
    python chapter6.py --dataset path/to/datasets  # Use your own dataset

Keyboard shortcuts in plot windows:
    ESC  - Close window
    S    - Save current figure as PNG
"""

import os
import csv
import argparse
import numpy as np
import cv2
from abc import ABC, abstractmethod
from matplotlib import cm
from matplotlib import pyplot as plt


# ============================================================================
# Feature Extraction Functions
# ============================================================================

def extract_hog_features(img, cell_size=8, block_size=2, nbins=9):
    """
    Extract Histogram of Oriented Gradients (HOG) features from an image.

    Args:
        img: Input image (H, W, 3) or (H, W)
        cell_size: Size of each cell in pixels
        block_size: Number of cells per block
        nbins: Number of orientation bins

    Returns:
        1D feature vector
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Ensure image is compatible with HOG block layout
    h, w = gray.shape
    win_size = (w // cell_size * cell_size, h // cell_size * cell_size)
    gray = cv2.resize(gray, win_size)

    hog = cv2.HOGDescriptor(
        _winSize=win_size,
        _blockSize=(block_size * cell_size, block_size * cell_size),
        _blockStride=(cell_size, cell_size),
        _cellSize=(cell_size, cell_size),
        _nbins=nbins
    )
    features = hog.compute(gray)
    return features.flatten()


# Module-level flag to avoid repeated SURF warnings
_surf_warned = False


def extract_surf_features(img, hessian_threshold=400):
    """
    Extract SURF (Speeded Up Robust Features) descriptors.
    Returns a fixed-length feature vector using a bag-of-words approach
    with dense sampling for classification.

    Args:
        img: Input image
        hessian_threshold: SURF hessian threshold

    Returns:
        1D feature vector (histogram of visual words)
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    try:
        surf = cv2.xfeatures2d.SURF_create(hessian_threshold)
    except AttributeError:
        # Fallback if opencv-contrib-python is not installed
        global _surf_warned
        if not _surf_warned:
            print("Warning: SURF requires opencv-contrib-python. Using HOG fallback.")
            _surf_warned = True
        return extract_hog_features(img)

    # Dense keypoint sampling (grid) for classification
    h, w = gray.shape
    step = 8
    keypoints = []
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            keypoints.append(cv2.KeyPoint(float(x), float(y), float(step)))

    _, descriptors = surf.compute(gray, keypoints)

    if descriptors is None or len(descriptors) == 0:
        return np.zeros(64, dtype=np.float32)

    # Flatten and take a fixed-size representation
    # For simplicity, use mean pooling of descriptors
    feat = np.mean(descriptors, axis=0)
    return feat.astype(np.float32)


def extract_raw_features(img):
    """Flatten raw pixel values as feature vector."""
    return img.flatten().astype(np.float32)


def extract_gray_features(img):
    """Convert to grayscale and flatten."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return gray.flatten().astype(np.float32)


def extract_rgb_features(img):
    """Flatten RGB (BGR in OpenCV) pixel values."""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.flatten().astype(np.float32)


def extract_hsv_features(img):
    """Convert to HSV color space and flatten."""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv.flatten().astype(np.float32)


# Map feature names to extraction functions
FEATURE_EXTRACTORS = {
    "raw": extract_raw_features,
    "gray": extract_gray_features,
    "rgb": extract_rgb_features,
    "hsv": extract_hsv_features,
    "surf": extract_surf_features,
    "hog": extract_hog_features,
}


# ============================================================================
# Abstract Classifier Base Class
# ============================================================================

class Classifier(ABC):
    """
    Abstract base class for all classifiers.

    Every classifier must implement:
        fit(X_train, y_train)  - Train on labeled data
        evaluate(X_test, y_test, visualize) - Test and return metrics
    """

    @abstractmethod
    def fit(self, X_train, y_train):
        """
        Train the classifier.

        Args:
            X_train: Training feature matrix (n_samples, n_features)
            y_train: Training labels (n_samples,)
        """
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test, visualize=False):
        """
        Evaluate the classifier on test data.

        Args:
            X_test: Test feature matrix (n_samples, n_features)
            y_test: True labels (n_samples,)
            visualize: If True, show confusion matrix plot

        Returns:
            dict with keys: accuracy, precision, recall, confusion_matrix
        """
        pass


# ============================================================================
# Multi-Class SVM Classifier
# ============================================================================

class MultiClassSVM(Classifier):
    """
    Multi-class classification using Support Vector Machines (SVMs).

    Supports two strategies:
        "one-vs-all"  : Train K binary SVMs, each class vs rest
        "one-vs-one"  : Train K*(K-1)/2 pairwise SVMs, majority vote

    Args:
        num_classes: Number of classes (K)
        mode: "one-vs-all" or "one-vs-one"
        params: Dictionary of SVM parameters (kernel, C, gamma, etc.)
    """

    def __init__(self, num_classes, mode="one-vs-all", params=None):
        self.num_classes = num_classes
        self.mode = mode
        self.params = params or dict()

        # Default SVM parameters
        self._default_params = {
            "kernel": cv2.ml.SVM_LINEAR,
            "C": 1.0,
            "gamma": 1.0,
            "type": cv2.ml.SVM_C_SVC,
        }
        self._default_params.update(self.params)

        # Initialize correct number of binary SVM classifiers
        self.classifiers = []
        if mode == "one-vs-one":
            # K classes: need K*(K-1)/2 classifiers
            n = num_classes * (num_classes - 1) // 2
            for _ in range(n):
                self.classifiers.append(self._create_svm())
        elif mode == "one-vs-all":
            # K classes: need K classifiers
            for _ in range(num_classes):
                self.classifiers.append(self._create_svm())
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'one-vs-all' or 'one-vs-one'.")

    def _create_svm(self):
        """Create a single binary SVM classifier with current parameters."""
        svm = cv2.ml.SVM_create()
        svm.setType(self._default_params["type"])
        svm.setKernel(self._default_params["kernel"])
        svm.setC(self._default_params["C"])
        svm.setGamma(self._default_params["gamma"])
        svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-6))
        return svm

    def fit(self, X_train, y_train):
        """
        Train all binary SVM classifiers.

        Args:
            X_train: Training data (n_samples, n_features), float32
            y_train: Training labels (n_samples,), int
        """
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int32)

        if self.mode == "one-vs-all":
            self._fit_one_vs_all(X_train, y_train)
        elif self.mode == "one-vs-one":
            self._fit_one_vs_one(X_train, y_train)

    def _fit_one_vs_all(self, X_train, y_train):
        """Train K classifiers: each class vs all others."""
        for c in range(self.num_classes):
            # Binary labels: 1 for class c, -1 for all others
            y_binary = np.where(y_train == c, 1, -1).astype(np.int32)
            self.classifiers[c].train(X_train, cv2.ml.ROW_SAMPLE, y_binary)

    def _fit_one_vs_one(self, X_train, y_train):
        """Train K*(K-1)/2 pairwise classifiers."""
        svm_idx = 0
        for c1 in range(self.num_classes):
            for c2 in range(c1 + 1, self.num_classes):
                # Select samples from class c1 and c2 only
                mask = (y_train == c1) | (y_train == c2)
                X_pair = X_train[mask]
                y_pair = y_train[mask]

                # Relabel: 1 for c1, -1 for c2
                y_binary = np.where(y_pair == c1, 1, -1).astype(np.int32)

                if len(np.unique(y_binary)) < 2:
                    # Skip if only one class present in training data
                    svm_idx += 1
                    continue

                self.classifiers[svm_idx].train(X_pair, cv2.ml.ROW_SAMPLE, y_binary)
                svm_idx += 1

    def predict(self, X_test):
        """
        Predict class labels for test samples.

        Args:
            X_test: Test data (n_samples, n_features), float32

        Returns:
            y_pred: Predicted labels (n_samples,)
        """
        X_test = np.array(X_test, dtype=np.float32)
        n_samples = X_test.shape[0]

        if self.mode == "one-vs-all":
            return self._predict_one_vs_all(X_test, n_samples)
        elif self.mode == "one-vs-one":
            return self._predict_one_vs_one(X_test, n_samples)

    def _predict_one_vs_all(self, X_test, n_samples):
        """
        Predict using one-vs-all strategy.

        Note on OpenCV SVM sign convention:
            Raw output > 0  →  predicted class label = -1 (not this class)
            Raw output < 0  →  predicted class label = +1 (is this class)
        So we NEGATE the raw values: higher (more positive) = more confident
        that this sample belongs to class c.
        """
        decisions = np.zeros((n_samples, self.num_classes), dtype=np.float32)

        for c in range(self.num_classes):
            svm = self.classifiers[c]
            # Get raw decision values (signed distance from hyperplane)
            _, results = svm.predict(X_test, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
            # Negate because OpenCV's sign convention is reversed
            decisions[:, c] = -results.flatten()

        # Class with highest (most positive) decision value wins
        return np.argmax(decisions, axis=1)

    def _predict_one_vs_one(self, X_test, n_samples):
        """Predict using one-vs-one strategy (majority voting)."""
        # Vote matrix: n_samples x num_classes
        votes = np.zeros((n_samples, self.num_classes), dtype=np.int32)

        svm_idx = 0
        for c1 in range(self.num_classes):
            for c2 in range(c1 + 1, self.num_classes):
                svm = self.classifiers[svm_idx]
                _, results = svm.predict(X_test)
                preds = results.flatten().astype(np.int32)

                # Vote for c1 if prediction is 1, vote for c2 if -1
                votes[preds == 1, c1] += 1
                votes[preds == -1, c2] += 1

                svm_idx += 1

        # Class with most votes wins
        return np.argmax(votes, axis=1)

    def evaluate(self, X_test, y_test, visualize=False):
        """
        Evaluate classifier performance.

        Args:
            X_test: Test data
            y_test: True labels
            visualize: If True, display confusion matrix

        Returns:
            dict with accuracy, precision, recall, confusion_matrix
        """
        y_test = np.array(y_test, dtype=np.int32)
        y_pred = self.predict(X_test)

        # Confusion matrix
        n_classes = self.num_classes
        cm = np.zeros((n_classes, n_classes), dtype=np.int32)
        for true, pred in zip(y_test, y_pred):
            cm[true, pred] += 1

        # Accuracy
        accuracy = np.sum(y_pred == y_test) / len(y_test)

        # Per-class precision and recall
        precision = np.zeros(n_classes)
        recall = np.zeros(n_classes)
        for c in range(n_classes):
            tp = cm[c, c]
            fp = np.sum(cm[:, c]) - tp
            fn = np.sum(cm[c, :]) - tp
            precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall[c] = tp / (fn + tp) if (fn + tp) > 0 else 0.0

        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": cm,
            "y_pred": y_pred,
        }

        if visualize:
            self._plot_confusion_matrix(cm)

        return results

    def _plot_confusion_matrix(self, cm):
        """Plot confusion matrix as a heatmap."""
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(self.num_classes),
            yticks=np.arange(self.num_classes),
            xlabel="Predicted label",
            ylabel="True label",
            title="Confusion Matrix",
        )

        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                ax.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black"
                )

        fig.tight_layout()
        self._show_plot_with_shortcuts(fig, "confusion_matrix.png")

    def _show_plot_with_shortcuts(self, fig, save_name):
        """
        Show a matplotlib figure with keyboard shortcuts.
        ESC to close, S to save as PNG.
        """
        def on_key(event):
            if event.key == "escape":
                plt.close(fig)
            elif event.key.lower() == "s":
                fig.savefig(save_name, dpi=150, bbox_inches="tight")
                print(f"Figure saved to {save_name}")

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()


# ============================================================================
# Data Loading (GTSRB-compatible format)
# ============================================================================

def load_data(
    rootpath="datasets",
    feature="hog",
    cut_roi=True,
    test_split=0.2,
    plot_samples=False,
    seed=113,
    img_size=(32, 32),
):
    """
    Load traffic sign dataset in GTSRB format.

    Expected directory structure:
        datasets/
            00000/
                GT-00000.csv
                00000_00000.ppm
                00000_00001.ppm
                ...
            00001/
                GT-00001.csv
                ...

    CSV format (semicolon-separated):
        Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId

    Args:
        rootpath: Path to dataset root directory
        feature: Feature type ("raw", "gray", "rgb", "hsv", "surf", "hog")
        cut_roi: If True, crop to Region of Interest from CSV
        test_split: Fraction of data to use for testing
        plot_samples: If True, show sample images
        seed: Random seed for reproducibility
        img_size: Target image size (width, height)

    Returns:
        (X_train, y_train), (X_test, y_test), class_names
    """
    # Get feature extractor
    if feature not in FEATURE_EXTRACTORS:
        raise ValueError(f"Unknown feature: {feature}. Choose from {list(FEATURE_EXTRACTORS.keys())}")
    extractor = FEATURE_EXTRACTORS[feature]

    # Discover class directories
    class_dirs = sorted([
        d for d in os.listdir(rootpath)
        if os.path.isdir(os.path.join(rootpath, d)) and d.isdigit()
    ])

    if not class_dirs:
        raise FileNotFoundError(
            f"No class directories found in '{rootpath}'. "
            f"Expected numbered directories like 00000/, 00001/, etc."
        )

    classes = [int(d) for d in class_dirs]
    num_classes = len(classes)
    class_map = {orig: idx for idx, orig in enumerate(classes)}
    class_names = [f"Class {c}" for c in classes]

    X = []
    labels = []

    for class_idx, class_dir in enumerate(class_dirs):
        prefix = os.path.join(rootpath, class_dir) + os.sep
        gt_file_path = os.path.join(prefix, f"GT-{class_dir}.csv")

        if not os.path.exists(gt_file_path):
            print(f"Warning: No GT file found at {gt_file_path}, skipping class {class_dir}")
            continue

        with open(gt_file_path, "r") as gt_file:
            gt_reader = csv.reader(gt_file, delimiter=";")
            next(gt_reader)  # Skip header row

            for row in gt_reader:
                if len(row) < 8:
                    continue

                filename = row[0]
                img_path = prefix + filename

                if not os.path.exists(img_path):
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    continue

                # Crop to ROI if requested
                if cut_roi and len(row) >= 7:
                    try:
                        x1 = int(row[3])
                        y1 = int(row[4])
                        x2 = int(row[5])
                        y2 = int(row[6])
                        img = img[y1:y2, x1:x2, :]
                    except (ValueError, IndexError):
                        pass

                # Resize to fixed size
                img = cv2.resize(img, img_size)

                # Extract features
                feat = extractor(img)
                X.append(feat)
                labels.append(class_idx)

    if len(X) == 0:
        raise RuntimeError("No data loaded. Check dataset path and format.")

    # Normalize features
    X = np.array(X, dtype=np.float32)

    # Per-sample mean subtraction (like the original code)
    X = np.array([x - np.mean(x) for x in X], dtype=np.float32)

    labels = np.array(labels, dtype=np.int32)

    # Shuffle with fixed seed
    np.random.seed(seed)
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    labels = labels[shuffle_idx]

    # Train/test split
    split_idx = int(len(X) * (1 - test_split))
    X_train = X[:split_idx]
    y_train = labels[:split_idx]
    X_test = X[split_idx:]
    y_test = labels[split_idx:]

    # Plot sample images if requested
    if plot_samples:
        _plot_sample_images(rootpath, class_dirs, class_map, img_size)

    return (X_train, y_train), (X_test, y_test), class_names


def _plot_sample_images(rootpath, class_dirs, class_map, img_size):
    """Plot a grid of sample images from each class."""
    n_classes = len(class_dirs)
    samples_per_class = 5
    fig, axes = plt.subplots(n_classes, samples_per_class, figsize=(12, 2.5 * n_classes))
    if n_classes == 1:
        axes = axes.reshape(1, -1)

    for row, class_dir in enumerate(class_dirs):
        prefix = os.path.join(rootpath, class_dir) + os.sep
        gt_file_path = os.path.join(prefix, f"GT-{class_dir}.csv")

        images = []
        if os.path.exists(gt_file_path):
            with open(gt_file_path, "r") as f:
                reader = csv.reader(f, delimiter=";")
                next(reader)  # skip header
                for i, gt_row in enumerate(reader):
                    if i >= samples_per_class:
                        break
                    img_path = prefix + gt_row[0]
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, img_size)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images.append(img)

        for col in range(samples_per_class):
            ax = axes[row, col]
            if col < len(images):
                ax.imshow(images[col])
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(f"Class {class_dir}", rotation=0, labelpad=40, va="center")

    fig.suptitle("Sample Images from Each Class", fontsize=14)
    fig.tight_layout()

    def on_key(event):
        if event.key == "escape":
            plt.close(fig)
        elif event.key.lower() == "s":
            fig.savefig("sample_images.png", dpi=150, bbox_inches="tight")
            print("Sample images saved to sample_images.png")

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


# ============================================================================
# Main Demo: Compare features and strategies
# ============================================================================

def run_comparison(dataset_path="datasets", features=None, modes=None):
    """
    Run full comparison of feature types and SVM strategies.

    Args:
        dataset_path: Path to dataset
        features: List of feature types to compare
        modes: List of SVM modes to compare
    """
    if features is None:
        features = ["gray", "rgb", "hsv", "hog"]
    if modes is None:
        modes = ["one-vs-all", "one-vs-one"]

    results = {}
    class_names = None

    for feat_name in features:
        print(f"\n{'='*60}")
        print(f"Feature: {feat_name.upper()}")
        print(f"{'='*60}")

        try:
            (X_train, y_train), (X_test, y_test), class_names = load_data(
                rootpath=dataset_path,
                feature=feat_name,
                cut_roi=True,
                test_split=0.2,
                plot_samples=False,
            )
        except Exception as e:
            print(f"  Skipping {feat_name}: {e}")
            continue

        num_classes = len(np.unique(y_train))
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Number of classes: {num_classes}")
        print(f"  Feature dimension: {X_train.shape[1]}")

        for mode in modes:
            print(f"\n  Mode: {mode}")
            clf = MultiClassSVM(num_classes=num_classes, mode=mode)
            clf.fit(X_train, y_train)
            res = clf.evaluate(X_test, y_test, visualize=False)

            key = f"{feat_name}_{mode}"
            results[key] = {
                "accuracy": res["accuracy"],
                "precision": res["precision"],
                "recall": res["recall"],
                "feature": feat_name,
                "mode": mode,
            }

            print(f"    Accuracy:  {res['accuracy']:.4f}")
            print(f"    Precision (mean): {np.mean(res['precision']):.4f}")
            print(f"    Recall (mean):    {np.mean(res['recall']):.4f}")

    # Plot comparison bar chart
    if results:
        _plot_comparison(results, features, modes)

    return results


def _plot_comparison(results, features, modes):
    """Plot grouped bar chart comparing all feature/mode combinations."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(features))
    width = 0.35

    for i, mode in enumerate(modes):
        accuracies = []
        for feat in features:
            key = f"{feat}_{mode}"
            if key in results:
                accuracies.append(results[key]["accuracy"])
            else:
                accuracies.append(0.0)

        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, accuracies, width, label=mode)

        # Add value labels on top of bars
        for bar, acc in zip(bars, accuracies):
            if acc > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    bar.get_height() + 0.01,
                    f"{acc:.2f}",
                    ha="center", va="bottom", fontsize=9,
                )

    ax.set_xlabel("Feature Type")
    ax.set_ylabel("Accuracy")
    ax.set_title("SVM Classification Accuracy by Feature Type and Strategy")
    ax.set_xticks(x)
    ax.set_xticklabels([f.upper() for f in features])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    def on_key(event):
        if event.key == "escape":
            plt.close(fig)
        elif event.key.lower() == "s":
            fig.savefig("feature_comparison.png", dpi=150, bbox_inches="tight")
            print("Comparison chart saved to feature_comparison.png")

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


# ============================================================================
# Quick Single-Class Demo (uses synthetic data if no dataset)
# ============================================================================

def quick_demo(dataset_path="datasets"):
    """
    Run a quick demo with the best-performing configuration (HOG + one-vs-all).
    Falls back to synthetic data if no real dataset found.
    """
    print("=" * 60)
    print("Traffic Sign Classification - Quick Demo")
    print("=" * 60)

    # Try loading real data first
    try:
        (X_train, y_train), (X_test, y_test), class_names = load_data(
            rootpath=dataset_path,
            feature="hog",
            cut_roi=True,
            test_split=0.2,
            plot_samples=True,
        )
        print(f"\nLoaded real dataset from '{dataset_path}'")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\nNo real dataset found at '{dataset_path}'.")
        print("Run 'python generate_synthetic_dataset.py' first to create test data.")
        print(f"Error: {e}")
        return

    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Feature dimension: {X_train.shape[1]}")

    # Train and evaluate
    print("\nTraining MultiClass SVM (one-vs-all, HOG features)...")
    clf = MultiClassSVM(num_classes=num_classes, mode="one-vs-all")
    clf.fit(X_train, y_train)

    results = clf.evaluate(X_test, y_test, visualize=True)

    print(f"\nResults:")
    print(f"  Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.1f}%)")
    print(f"  Precision (per class): {[f'{p:.2f}' for p in results['precision']]}")
    print(f"  Recall (per class):    {[f'{r:.2f}' for r in results['recall']]}")

    return clf, results


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Chapter 6: Multi-Class Traffic Sign Classification with SVM"
    )
    parser.add_argument(
        "--dataset", "-d",
        default="datasets",
        help="Path to dataset directory (default: datasets/)",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["demo", "compare", "train"],
        default="demo",
        help="Run mode: demo (quick test), compare (full comparison), train (single model)",
    )
    parser.add_argument(
        "--feature", "-f",
        default="hog",
        choices=list(FEATURE_EXTRACTORS.keys()),
        help="Feature type for 'train' mode (default: hog)",
    )
    parser.add_argument(
        "--svm-mode",
        default="one-vs-all",
        choices=["one-vs-all", "one-vs-one"],
        help="SVM strategy for 'train' mode (default: one-vs-all)",
    )

    args = parser.parse_args()

    if args.mode == "demo":
        quick_demo(args.dataset)
    elif args.mode == "compare":
        run_comparison(args.dataset)
    elif args.mode == "train":
        (X_train, y_train), (X_test, y_test), class_names = load_data(
            rootpath=args.dataset,
            feature=args.feature,
            cut_roi=True,
            test_split=0.2,
            plot_samples=True,
        )
        num_classes = len(class_names)
        clf = MultiClassSVM(num_classes=num_classes, mode=args.svm_mode)
        clf.fit(X_train, y_train)
        results = clf.evaluate(X_test, y_test, visualize=True)
        print(f"Accuracy: {results['accuracy']:.4f}")


if __name__ == "__main__":
    main()
