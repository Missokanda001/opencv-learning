"""
Generate a synthetic traffic sign dataset for testing the classification pipeline.

Creates 5 classes of synthetic "traffic sign-like" images in GTSRB-compatible format,
so you can test chapter6.py immediately without downloading the full GTSRB dataset.

Classes generated:
    00000 - Red circle (prohibition style)
    00001 - Red triangle (warning style)
    00002 - Blue circle (mandatory style)
    00003 - Red octagon (stop style)
    00004 - Yellow diamond (priority style)

Usage:
    python generate_synthetic_dataset.py
    python generate_synthetic_dataset.py --output datasets --samples 50 --size 64
"""

import os
import csv
import argparse
import numpy as np
import cv2


# Color definitions (BGR format for OpenCV)
COLORS = {
    "red": (0, 0, 255),
    "dark_red": (0, 0, 180),
    "blue": (255, 0, 0),
    "dark_blue": (180, 0, 0),
    "yellow": (0, 200, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "gray": (180, 180, 180),
    "green": (0, 180, 0),
}


def draw_red_circle(img_size, variation=0):
    """Draw a red circular prohibition-style sign."""
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240  # light gray bg

    center = (img_size // 2, img_size // 2)
    radius = int(img_size * 0.35)

    # Add slight position variation
    dx = int(np.random.uniform(-3, 3))
    dy = int(np.random.uniform(-3, 3))
    center = (center[0] + dx, center[1] + dy)

    # Vary radius slightly
    radius = int(radius * np.random.uniform(0.9, 1.1))

    # White inner circle
    cv2.circle(img, center, radius, COLORS["white"], -1)
    # Red border
    cv2.circle(img, center, radius, COLORS["red"], max(2, img_size // 16))

    # Add a diagonal red line (like "no entry")
    if variation % 3 == 0:
        pt1 = (center[0] - radius + 5, center[1] - radius + 5)
        pt2 = (center[0] + radius - 5, center[1] + radius - 5)
        cv2.line(img, pt1, pt2, COLORS["red"], max(2, img_size // 20))

    return img


def draw_red_triangle(img_size, variation=0):
    """Draw a red triangular warning-style sign."""
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240

    center = (img_size // 2, img_size // 2)
    size = int(img_size * 0.38)

    # Triangle vertices (pointing up)
    pts = np.array([
        [center[0], center[1] - size],
        [center[0] - size, center[1] + int(size * 0.7)],
        [center[0] + size, center[1] + int(size * 0.7)],
    ], np.int32)

    # Add slight position variation
    dx = int(np.random.uniform(-3, 3))
    dy = int(np.random.uniform(-3, 3))
    pts[:, 0] += dx
    pts[:, 1] += dy

    # White fill
    cv2.fillPoly(img, [pts], COLORS["white"])
    # Red border
    cv2.polylines(img, [pts], True, COLORS["red"], max(2, img_size // 16))

    # Add a small black exclamation mark in center
    if variation % 2 == 0:
        cx, cy = center[0] + dx, center[1] + dy
        cv2.putText(img, "!", (cx - 5, cy + 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, COLORS["black"], 2)

    return img


def draw_blue_circle(img_size, variation=0):
    """Draw a blue circular mandatory-style sign."""
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240

    center = (img_size // 2, img_size // 2)
    radius = int(img_size * 0.35)

    dx = int(np.random.uniform(-3, 3))
    dy = int(np.random.uniform(-3, 3))
    center = (center[0] + dx, center[1] + dy)
    radius = int(radius * np.random.uniform(0.9, 1.1))

    # Blue fill
    cv2.circle(img, center, radius, COLORS["blue"], -1)
    # Dark blue border
    cv2.circle(img, center, radius, COLORS["dark_blue"], max(2, img_size // 20))

    # White arrow pointing up (like "go straight")
    if variation % 3 == 0:
        cx, cy = center
        arrow_pts = np.array([
            [cx, cy - radius // 2],
            [cx - radius // 3, cy],
            [cx - radius // 6, cy],
            [cx - radius // 6, cy + radius // 2],
            [cx + radius // 6, cy + radius // 2],
            [cx + radius // 6, cy],
            [cx + radius // 3, cy],
        ], np.int32)
        cv2.fillPoly(img, [arrow_pts], COLORS["white"])

    return img


def draw_red_octagon(img_size, variation=0):
    """Draw a red octagonal stop-style sign."""
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240

    center = (img_size // 2, img_size // 2)
    size = int(img_size * 0.35)

    # Octagon vertices
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
    pts = np.array([
        [int(center[0] + size * np.cos(a)),
         int(center[1] + size * np.sin(a))]
        for a in angles
    ], np.int32)

    dx = int(np.random.uniform(-2, 2))
    dy = int(np.random.uniform(-2, 2))
    pts[:, 0] += dx
    pts[:, 1] += dy

    # Red fill
    cv2.fillPoly(img, [pts], COLORS["dark_red"])
    # Red border
    cv2.polylines(img, [pts], True, COLORS["red"], max(2, img_size // 20))

    # White "STOP" text (simplified)
    if variation % 2 == 0:
        cx, cy = center[0] + dx, center[1] + dy
        cv2.putText(img, "STOP", (cx - 20, cy + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, COLORS["white"], 2)

    return img


def draw_yellow_diamond(img_size, variation=0):
    """Draw a yellow diamond priority-style sign."""
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240

    center = (img_size // 2, img_size // 2)
    size = int(img_size * 0.38)

    # Diamond (rotated square) vertices
    pts = np.array([
        [center[0], center[1] - size],
        [center[0] + size, center[1]],
        [center[0], center[1] + size],
        [center[0] - size, center[1]],
    ], np.int32)

    dx = int(np.random.uniform(-3, 3))
    dy = int(np.random.uniform(-3, 3))
    pts[:, 0] += dx
    pts[:, 1] += dy

    # Yellow fill
    cv2.fillPoly(img, [pts], COLORS["yellow"])
    # Black border
    cv2.polylines(img, [pts], True, COLORS["black"], max(2, img_size // 20))

    # Small black dot in center
    if variation % 2 == 0:
        cx, cy = center[0] + dx, center[1] + dy
        cv2.circle(img, (cx, cy), max(2, size // 8), COLORS["black"], -1)

    return img


# Map class indices to drawing functions and names
CLASS_GENERATORS = [
    (draw_red_circle, "Red Circle (Prohibition)"),
    (draw_red_triangle, "Red Triangle (Warning)"),
    (draw_blue_circle, "Blue Circle (Mandatory)"),
    (draw_red_octagon, "Red Octagon (Stop)"),
    (draw_yellow_diamond, "Yellow Diamond (Priority)"),
]


def add_noise(img, level=5):
    """Add small Gaussian noise to make images more realistic."""
    noise = np.random.normal(0, level, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def generate_dataset(output_dir="datasets", samples_per_class=50, img_size=64, seed=42):
    """
    Generate a synthetic traffic sign dataset in GTSRB format.

    Args:
        output_dir: Output directory path
        samples_per_class: Number of images per class
        img_size: Image size (width and height in pixels)
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    total_generated = 0
    class_info = []

    for class_idx, (generator, class_name) in enumerate(CLASS_GENERATORS):
        class_dir = os.path.join(output_dir, f"{class_idx:05d}")
        os.makedirs(class_dir, exist_ok=True)

        gt_csv_path = os.path.join(class_dir, f"GT-{class_idx:05d}.csv")

        with open(gt_csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            # Header: Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId
            writer.writerow([
                "Filename", "Width", "Height",
                "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2",
                "ClassId"
            ])

            for sample_idx in range(samples_per_class):
                filename = f"{class_idx:05d}_{sample_idx:05d}.ppm"
                filepath = os.path.join(class_dir, filename)

                # Generate image with variation
                img = generator(img_size, variation=sample_idx)
                img = add_noise(img, level=3)

                # Save as PPM (GTSRB format)
                cv2.imwrite(filepath, img)

                # ROI: the sign occupies most of the image
                margin = int(img_size * 0.1)
                roi_x1 = margin
                roi_y1 = margin
                roi_x2 = img_size - margin
                roi_y2 = img_size - margin

                writer.writerow([
                    filename,
                    img_size, img_size,
                    roi_x1, roi_y1, roi_x2, roi_y2,
                    class_idx,
                ])

                total_generated += 1

        class_info.append((class_idx, class_name, samples_per_class))

    # Print summary
    print(f"\n{'='*60}")
    print("Synthetic Dataset Generated Successfully!")
    print(f"{'='*60}")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print(f"Total images: {total_generated}")
    print(f"Image size: {img_size}x{img_size}")
    print(f"\nClasses:")
    for idx, name, count in class_info:
        print(f"  Class {idx:05d} ({name}): {count} samples")

    print(f"\nNext steps:")
    print(f"  1. Run the quick demo:")
    print(f"     python chapter6.py --dataset {output_dir}")
    print(f"  2. Run full feature comparison:")
    print(f"     python chapter6.py --mode compare --dataset {output_dir}")
    print(f"  3. Train with specific features:")
    print(f"     python chapter6.py --mode train --feature hog --svm-mode one-vs-all --dataset {output_dir}")
    print()

    return class_info


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic traffic sign dataset for testing"
    )
    parser.add_argument(
        "--output", "-o",
        default="datasets",
        help="Output directory (default: datasets/)",
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=50,
        help="Number of samples per class (default: 50)",
    )
    parser.add_argument(
        "--size", "-s",
        type=int,
        default=64,
        help="Image size in pixels (default: 64)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()
    generate_dataset(
        output_dir=args.output,
        samples_per_class=args.samples,
        img_size=args.size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
