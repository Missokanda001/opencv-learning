#!/usr/bin/env python3
"""
Facial Expression Recognition - Main GUI Application
Python 3 compatible - FULLY CORRECTED VERSION
Camera index: 1 (external webcam)
"""

import cv2
import numpy as np
import wx
import time
import sys
import os
from os import path
import pickle

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detectors import FaceDetector
from classifiers import MultiLayerPerceptron
from datasets import homebrew
from gui import BaseLayout


class FaceLayout(BaseLayout):
    """Facial Expression Recognition GUI layout."""

    def __init__(self, parent, id, title, capture,
                 data_file='datasets/faces_training.pkl',
                 load_preprocessed_data='datasets/faces_preprocessed.pkl',
                 load_mlp='params/mlp.xml'):
        # Call parent constructor first (creates video_panel + control_panel)
        super(FaceLayout, self).__init__(parent, id, title, capture, fps=15)

        self.data_file = data_file
        self.load_preprocessed_data = load_preprocessed_data
        self.load_mlp = load_mlp

        # Training data storage
        self.samples = []
        self.labels = []

        # Face detector
        self.faces = None
        self.head = None

        # MLP classifier and PCA params (for testing)
        self.MLP = None
        self.pca_V = None
        self.pca_m = None
        self.all_labels = None

        # Initialize all UI controls on the RIGHT panel
        self._init_controls()

        # Initialize face detector and (optionally) MLP
        self.init_algorithm()

    # ============================================================
    # UI CONTROLS - all widgets use self.control_panel as parent!
    # ============================================================
    def _init_controls(self):
        """Build all UI controls on the right-side control panel."""

        # --- Mode section ---
        mode_title = wx.StaticText(self.control_panel, -1, "Mode")
        mode_title.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))

        self.radio_train = wx.RadioButton(self.control_panel, -1, "Training", style=wx.RB_GROUP)
        self.radio_test = wx.RadioButton(self.control_panel, -1, "Testing")
        self.radio_train.SetValue(True)

        self.Bind(wx.EVT_RADIOBUTTON, self._on_training, self.radio_train)
        self.Bind(wx.EVT_RADIOBUTTON, self._on_testing, self.radio_test)

        # --- Expression section ---
        expr_title = wx.StaticText(self.control_panel, -1, "Expression")
        expr_title.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))

        self.radio_neutral = wx.RadioButton(self.control_panel, -1, "Neutral", style=wx.RB_GROUP)
        self.radio_happy = wx.RadioButton(self.control_panel, -1, "Happy")
        self.radio_sad = wx.RadioButton(self.control_panel, -1, "Sad")
        self.radio_surprised = wx.RadioButton(self.control_panel, -1, "Surprised")
        self.radio_angry = wx.RadioButton(self.control_panel, -1, "Angry")
        self.radio_disgusted = wx.RadioButton(self.control_panel, -1, "Disgusted")
        self.radio_neutral.SetValue(True)

        # --- Snapshot button ---
        self.btn_snapshot = wx.Button(self.control_panel, -1, "Take Snapshot")
        self.btn_snapshot.Bind(wx.EVT_BUTTON, self._on_snapshot)

        # --- Status text ---
        self.status_text = wx.StaticText(self.control_panel, -1, "Ready - Training Mode")
        self.status_text.SetFont(wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))

        self.sample_count = wx.StaticText(self.control_panel, -1, "Samples collected: 0")

        # --- Add everything to control_sizer ---
        self.control_sizer.Add(mode_title, 0, wx.ALL, 5)
        self.control_sizer.Add(self.radio_train, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 3)
        self.control_sizer.Add(self.radio_test, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        self.control_sizer.Add(wx.StaticLine(self.control_panel, -1), 0, wx.EXPAND | wx.ALL, 3)

        self.control_sizer.Add(expr_title, 0, wx.ALL, 5)
        self.control_sizer.Add(self.radio_neutral, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 3)
        self.control_sizer.Add(self.radio_happy, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 3)
        self.control_sizer.Add(self.radio_sad, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 3)
        self.control_sizer.Add(self.radio_surprised, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 3)
        self.control_sizer.Add(self.radio_angry, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 3)
        self.control_sizer.Add(self.radio_disgusted, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        self.control_sizer.Add(wx.StaticLine(self.control_panel, -1), 0, wx.EXPAND | wx.ALL, 3)

        self.control_sizer.Add(self.btn_snapshot, 0, wx.EXPAND | wx.ALL, 8)
        self.control_sizer.Add(self.status_text, 0, wx.ALL, 5)
        self.control_sizer.Add(self.sample_count, 0, wx.ALL, 5)

        # Refresh layout
        self.control_panel.Layout()
        self.panel.Layout()

    # ============================================================
    # ALGORITHM INITIALIZATION
    # ============================================================
    def init_algorithm(self):
        """Initialize face detector and load pre-trained MLP if available."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        face_casc_path = os.path.join(script_dir, "params", "haarcascade_frontalface_default.xml")
        left_eye_path = os.path.join(script_dir, "params", "haarcascade_lefteye_2splits.xml")
        right_eye_path = os.path.join(script_dir, "params", "haarcascade_righteye_2splits.xml")

        try:
            self.faces = FaceDetector(
                face_casc=face_casc_path,
                left_eye_casc=left_eye_path,
                right_eye_casc=right_eye_path,
                scale_factor=4
            )
            print("[OK] Face detector loaded")
        except Exception as e:
            print(f"[ERROR] Face detector failed: {e}")
            self.faces = None
            wx.MessageBox(
                "Could not load Haar cascade files.\n"
                "Make sure all 3 XML files are in the params/ folder.",
                "Error", wx.OK | wx.ICON_ERROR
            )

        # Load preprocessed data + MLP for testing mode
        prep_path = os.path.join(script_dir, self.load_preprocessed_data)
        mlp_path = os.path.join(script_dir, self.load_mlp)

        if path.isfile(prep_path):
            (_, y_train), (_, y_test), self.pca_V, self.pca_m = homebrew.load_preprocessed(prep_path)
            if y_train is not None and y_test is not None:
                self.all_labels = np.unique(np.hstack((y_train, y_test)))
                if path.isfile(mlp_path):
                    try:
                        num_features = self.pca_V.shape[0]
                        num_classes = len(self.all_labels)
                        layer_sizes = np.int32([num_features, num_classes])
                        self.MLP = MultiLayerPerceptron(layer_sizes, self.all_labels)
                        self.MLP.load(mlp_path)
                        print(f"[OK] MLP loaded: {mlp_path}")
                    except Exception as e:
                        print(f"[WARNING] MLP load failed: {e}")
                        self.MLP = None
                else:
                    print("[INFO] No trained MLP found - testing disabled")
        else:
            print("[INFO] No preprocessed data - testing disabled")

        # Disable testing radio if no MLP available
        if self.MLP is None:
            self.radio_test.Disable()
            self.status_text.SetLabel("Ready - Training Mode\n(Testing unavailable)")

    # ============================================================
    # EVENT HANDLERS - all defined, no AttributeError!
    # ============================================================
    def _on_training(self, event):
        """Switch to training mode - enable all training buttons."""
        self.radio_neutral.Enable()
        self.radio_happy.Enable()
        self.radio_sad.Enable()
        self.radio_surprised.Enable()
        self.radio_angry.Enable()
        self.radio_disgusted.Enable()
        self.btn_snapshot.Enable()
        self.status_text.SetLabel("Training Mode - collect samples")
        self.status_text.SetForegroundColour(wx.Colour(0, 0, 0))

    def _on_testing(self, event):
        """Switch to testing mode - disable all training buttons."""
        self.radio_neutral.Disable()
        self.radio_happy.Disable()
        self.radio_sad.Disable()
        self.radio_surprised.Disable()
        self.radio_angry.Disable()
        self.radio_disgusted.Disable()
        self.btn_snapshot.Disable()
        self.status_text.SetLabel("Testing Mode - real-time recognition")
        self.status_text.SetForegroundColour(wx.Colour(0, 128, 0))

    def _on_snapshot(self, event):
        """Capture current face and add to training set with selected label."""
        # Determine selected expression label
        if self.radio_neutral.GetValue():
            label = 'neutral'
        elif self.radio_happy.GetValue():
            label = 'happy'
        elif self.radio_sad.GetValue():
            label = 'sad'
        elif self.radio_surprised.GetValue():
            label = 'surprised'
        elif self.radio_angry.GetValue():
            label = 'angry'
        elif self.radio_disgusted.GetValue():
            label = 'disgusted'
        else:
            label = 'neutral'

        if self.head is None:
            print("No face detected!")
            self.status_text.SetLabel("No face - position your face")
            self.status_text.SetForegroundColour(wx.Colour(255, 0, 0))
            return

        if self.faces is None:
            return

        # Align the detected face
        success, aligned_head = self.faces.align_head(self.head)

        if success:
            self.samples.append(aligned_head.flatten())
            self.labels.append(label)
            print(f"Added sample: {label} (total: {len(self.samples)})")
            self.status_text.SetLabel(f"Added: {label}")
            self.status_text.SetForegroundColour(wx.Colour(0, 128, 0))
            self.sample_count.SetLabel(f"Samples collected: {len(self.samples)}")
        else:
            print("Alignment failed (eye detection)")
            self.status_text.SetLabel("Eye detection failed - try again")
            self.status_text.SetForegroundColour(wx.Colour(255, 128, 0))

    # ============================================================
    # FRAME PROCESSING
    # ============================================================
    def _process_frame(self, frame):
        """Detect face and predict expression in testing mode."""
        if self.faces is None:
            return frame

        # Detect face
        success, frame, self.head = self.faces.detect(frame)

        # In testing mode: predict expression
        if success and self.radio_test.GetValue() and self.MLP is not None and self.pca_V is not None:
            align_success, head_aligned = self.faces.align_head(self.head)
            if align_success:
                # Extract PCA features
                X, _, _ = homebrew.extract_features(
                    [head_aligned.flatten()], V=self.pca_V, m=self.pca_m
                )
                X_arr = np.array(X, dtype=np.float32)
                if X_arr.ndim == 1:
                    X_arr = X_arr.reshape(1, -1)

                # Predict label
                label = self.MLP.predict(X_arr)[0]

                # Draw label on frame (re-detect to get face position)
                frame_small = cv2.cvtColor(
                    cv2.resize(frame, (0, 0), fx=0.25, fy=0.25), cv2.COLOR_BGR2GRAY
                )
                faces = self.faces.face_casc.detectMultiScale(frame_small, 1.1, 3)
                faces = self.faces._find_biggest(faces)
                if len(faces) > 0:
                    x, y, w, h = faces[0] * 4
                    cv2.putText(frame, str(label), (int(x), int(y) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame

    # ============================================================
    # EXIT / SAVE
    # ============================================================
    def _on_exit(self, event):
        """Save training data to file before closing."""
        if len(self.samples) > 0:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(script_dir, self.data_file)

            # Don't overwrite existing files - auto-number
            if path.isfile(data_path):
                filename, fileext = path.splitext(data_path)
                offset = 0
                while True:
                    candidate = f"{filename}-{offset}{fileext}"
                    if path.isfile(candidate):
                        offset += 1
                    else:
                        break
                data_path = candidate

            # Ensure directory exists
            os.makedirs(os.path.dirname(data_path), exist_ok=True)

            # Save with pickle
            with open(data_path, 'wb') as f:
                pickle.dump(self.samples, f)
                pickle.dump(self.labels, f)
            print(f"\n[SAVED] {len(self.samples)} samples -> {data_path}")

        # Call parent cleanup (stops timer, releases camera)
        super(FaceLayout, self)._on_exit(event)


# ============================================================
# MAIN ENTRY POINT
# ============================================================
def main():
    print("=" * 55)
    print("  Facial Expression Recognition  (Python 3)")
    print("=" * 55)

    # CAMERA INDEX 1 (external webcam) - falls back to 0 if not found
    capture = cv2.VideoCapture(1)
    if not capture.isOpened():
        print("[WARN] Camera 1 not found, trying camera 0...")
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            print("[ERROR] No camera found! Check your webcam connection.")
            sys.exit(1)
        else:
            print("[INFO] Using camera 0 (fallback)")
    else:
        print("[INFO] Using camera 1")

    # Set resolution
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Start GUI
    app = wx.App()
    layout = FaceLayout(None, -1, 'Facial Expression Recognition', capture)
    layout.Show(True)

    print("\nKeyboard shortcuts:")
    print("  ESC  =  Exit application")
    print("  S    =  Save screenshot")
    print("\nHow to collect training data:")
    print("  1. Select 'Training' mode")
    print("  2. Pick an expression (Neutral, Happy, etc.)")
    print("  3. Click 'Take Snapshot' repeatedly")
    print("  4. Switch expression and repeat")
    print("  5. Close window to auto-save training data")
    print("-" * 55)

    app.MainLoop()


if __name__ == '__main__':
    main()
