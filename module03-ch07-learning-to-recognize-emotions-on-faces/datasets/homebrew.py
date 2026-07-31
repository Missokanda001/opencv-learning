"""
Homebrew Dataset Module
Load and preprocess custom facial expression datasets.
Python 3 compatible version.
"""

import cv2
import numpy as np
import pickle
from os import path


def extract_features(X, V=None, m=None, num_components=None):
    """Extract features using PCA (Principal Component Analysis).

    Can either compute PCA from scratch (if V and m are None)
    or project data using pre-computed PCA basis vectors and mean.

    Args:
        X: list of flattened image samples (each is a 1D numpy array)
        V: pre-computed PCA basis vectors (eigenvectors), shape (n_components, n_features)
           If None, PCA is computed from scratch.
        m: pre-computed mean vector, shape (1, n_features)
           If None, mean is computed from scratch.
        num_components: number of principal components to keep.
                        Only used when computing PCA from scratch.
                        Defaults to 50 if None.

    Returns:
        tuple: (X_projected, V, m)
            - X_projected: list of projected feature vectors
            - V: PCA basis vectors (eigenvectors)
            - m: mean vector
    """
    if V is None or m is None:
        # Need to perform PCA from scratch
        if num_components is None:
            num_components = 50

        # Convert list to numpy array: rows are samples, cols are pixels/features
        Xarr = np.squeeze(np.array(X).astype(np.float32))
        if Xarr.ndim == 1:
            Xarr = Xarr.reshape(1, -1)

        # Perform PCA: returns mean and basis vectors (eigenvectors)
        m, V = cv2.PCACompute(Xarr, mean=None)

        # Keep only the first num_components principal components
        V = V[:num_components]

    # Project each sample onto the PCA basis
    # Projected = V @ (x - mean)
    mean_vec = m.flatten()  # shape (n_features,)
    for i in range(len(X)):
        X[i] = np.dot(V, X[i] - mean_vec)

    return X, V, m


def load_data(load_from_file, test_split=0.2, num_components=50,
              save_to_file=None, plot_samples=False, seed=113):
    """Load dataset from pickle file and perform PCA feature extraction.

    Args:
        load_from_file: path to the pickle file containing training samples
        test_split: fraction of data to use for testing (default 0.2)
        num_components: number of PCA components to keep (default 50)
        save_to_file: if provided, save preprocessed data to this file
        plot_samples: if True, plot some sample images (requires matplotlib)
        seed: random seed for reproducible train/test split

    Returns:
        tuple: ((X_train, y_train), (X_test, y_test), V, m)
            - X_train, y_train: training data and labels
            - X_test, y_test: test data and labels
            - V: PCA basis vectors
            - m: PCA mean vector
    """
    # Prepare lists for samples and labels
    X = []
    labels = []

    # Try to load the data file
    if not path.isfile(load_from_file):
        print("Could not find file", load_from_file)
        return (X, labels), (X, labels), None, None
    else:
        print("Loading data from", load_from_file)
        with open(load_from_file, 'rb') as f:
            samples = pickle.load(f)
            labels = pickle.load(f)
        print("Loaded", len(samples), "training samples")

    if len(samples) == 0:
        print("Warning: empty dataset")
        return (X, labels), (X, labels), None, None

    # Perform feature extraction (PCA)
    # Returns preprocessed samples, PCA basis vectors & mean
    X, V, m = extract_features(samples, num_components=num_components)

    # Convert labels to numpy array for consistent handling
    labels = np.array(labels)

    # Shuffle dataset (same seed for both X and labels)
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = [X[i] for i in indices]
    labels = labels[indices]

    # Split data according to test_split
    split_idx = int(len(X) * (1 - test_split))
    X_train = X[:split_idx]
    y_train = labels[:split_idx]
    X_test = X[split_idx:]
    y_test = labels[split_idx:]

    # Optionally save preprocessed data to file
    if save_to_file is not None:
        with open(save_to_file, 'wb') as f:
            pickle.dump(X_train, f)
            pickle.dump(y_train, f)
            pickle.dump(X_test, f)
            pickle.dump(y_test, f)
            pickle.dump(V, f)
            pickle.dump(m, f)
        print("Saved preprocessed data to", save_to_file)

    # Optional: plot some samples
    if plot_samples:
        try:
            from matplotlib import pyplot as plt
            from matplotlib import cm

            n_plot = min(10, len(X_train))
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            for i in range(n_plot):
                ax = axes[i // 5, i % 5]
                # Reconstruct a sample from PCA space for visualization
                # (approximate - just for sanity check)
                ax.set_title(str(y_train[i]))
                ax.axis('off')
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib not available, skipping plot")

    return (X_train, y_train), (X_test, y_test), V, m


def load_preprocessed(load_from_file):
    """Load preprocessed dataset (PCA features + labels + PCA params).

    Args:
        load_from_file: path to preprocessed pickle file

    Returns:
        tuple: ((X_train, y_train), (X_test, y_test), V, m)
    """
    if not path.isfile(load_from_file):
        print("Could not find preprocessed data file", load_from_file)
        return (None, None), (None, None), None, None

    with open(load_from_file, 'rb') as f:
        X_train = pickle.load(f)
        y_train = pickle.load(f)
        X_test = pickle.load(f)
        y_test = pickle.load(f)
        V = pickle.load(f)
        m = pickle.load(f)

    print("Loaded preprocessed data from", load_from_file)
    return (X_train, y_train), (X_test, y_test), V, m
