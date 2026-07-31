#!/usr/bin/env python3
"""
Train and Test MLP Script for Facial Expression Recognition
Python 3 compatible version.

Trains multiple MLP configurations on a homebrew dataset and saves the best one.
Usage: python train_test_mlp.py
"""

import cv2
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import homebrew
from classifiers import MultiLayerPerceptron


def main():
    # --- Configuration ---
    data_file = "datasets/faces_training.pkl"
    preprocessed_file = "datasets/faces_preprocessed.pkl"
    save_file = "params/mlp.xml"
    num_components = 50
    test_split = 0.2
    seed = 42

    # --- Load training data ---
    # Training data can be recorded using main.py in training mode
    print("=" * 60)
    print("Facial Expression Recognition - MLP Training")
    print("=" * 60)

    (X_train, y_train), (X_test, y_test), V, m = homebrew.load_data(
        data_file,
        num_components=num_components,
        test_split=test_split,
        save_to_file=preprocessed_file,
        seed=seed
    )

    if len(X_train) == 0 or len(X_test) == 0:
        print("Error: Empty data - cannot train")
        print(f"Please make sure {data_file} exists and contains training samples.")
        print("Run main.py in Training mode to collect samples first.")
        sys.exit(1)

    # Convert to numpy arrays
    X_train = np.squeeze(np.array(X_train)).astype(np.float32)
    y_train = np.array(y_train)
    X_test = np.squeeze(np.array(X_test)).astype(np.float32)
    y_test = np.array(y_test)

    # Ensure 2D arrays
    if X_train.ndim == 1:
        X_train = X_train.reshape(1, -1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(1, -1)

    # Find all class labels
    labels = np.unique(np.hstack((y_train, y_test)))
    print(f"\nClasses found: {labels}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of PCA features: {X_train.shape[1]}")
    print()

    # --- Training parameters ---
    params = dict(
        term_crit=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 300, 0.01),
        bp_dw_scale=0.001,
        bp_moment_scale=0.9
    )

    # --- Find best MLP configuration ---
    # Try 1-hidden-layer networks with increasing hidden layer size
    num_features = X_train.shape[1]
    num_classes = len(labels)

    print("1-hidden layer networks")
    print("-" * 40)

    best_acc = 0.0
    best_layer_sizes = None

    for l1 in range(10):
        # Gradually increase the hidden-layer size
        hidden_size = (l1 + 1) * num_features // 5
        if hidden_size < 2:
            hidden_size = 2
        layer_sizes = np.int32([num_features, hidden_size, num_classes])

        print(f"  Hidden size: {hidden_size}")

        # Create and train MLP
        MLP = MultiLayerPerceptron(layer_sizes, labels)
        MLP.fit(X_train, y_train, params=params)

        # Evaluate on training set
        acc_train, prec_train, rec_train = MLP.evaluate(X_train, y_train)
        print(f"    Train acc = {acc_train:.4f}, prec = {prec_train:.4f}, rec = {rec_train:.4f}")

        # Evaluate on test set
        acc_test, prec_test, rec_test = MLP.evaluate(X_test, y_test)
        print(f"    Test acc  = {acc_test:.4f}, prec = {prec_test:.4f}, rec = {rec_test:.4f}")

        # Save best model
        if acc_test > best_acc:
            best_acc = acc_test
            best_layer_sizes = layer_sizes.copy()
            # Ensure params directory exists
            os.makedirs(os.path.dirname(save_file) if os.path.dirname(save_file) else '.', exist_ok=True)
            MLP.save(save_file)
            print(f"    -> New best model saved to {save_file}")

        print()

    # --- Summary ---
    print("=" * 60)
    print("Training complete!")
    print(f"Best test accuracy: {best_acc:.4f}")
    print(f"Best layer sizes: {best_layer_sizes}")
    print(f"Model saved to: {save_file}")
    print(f"Preprocessed data saved to: {preprocessed_file}")
    print("=" * 60)


if __name__ == '__main__':
    main()
