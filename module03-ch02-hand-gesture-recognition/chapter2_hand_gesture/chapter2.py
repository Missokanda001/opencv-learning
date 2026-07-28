import sys
import os
import importlib.util

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# No subfolders anymore
ui_file = os.path.join(SCRIPT_FOLDER, "base_layout.py")
_ui_mod = _load_module("base_layout", ui_file)
BaseLayout = _ui_mod.BaseLayout

gest_file = os.path.join(SCRIPT_FOLDER, "hand_gesture.py")
_gest_mod = _load_module("hand_gesture", gest_file)
HandGestureRecognition = _gest_mod.HandGestureRecognition

import cv2
import numpy as np
import wx

# ==================== REST OF YOUR CODE (KinectLayout class, main()) BELOW ====================

# Direct file import (bypasses module system)
import importlib.util

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_ui_mod = _load_module("base_layout", os.path.join(SCRIPT_FOLDER, "ui", "base_layout.py"))
BaseLayout = _ui_mod.BaseLayout

_gest_mod = _load_module("hand_gesture", os.path.join(SCRIPT_FOLDER, "gestures", "hand_gesture.py"))
HandGestureRecognition = _gest_mod.HandGestureRecognition


class KinectLayout(BaseLayout):
    def _init_custom_layout(self):
        self.hand_gestures = HandGestureRecognition()

    def _create_custom_layout(self):
        info_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.finger_count_label = wx.StaticText(self, -1, "Fingers: --")
        font = wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        self.finger_count_label.SetFont(font)
        self.finger_count_label.SetForegroundColour(wx.Colour(0, 153, 76))
        hint_label = wx.StaticText(self, -1, "  |  ESC to quit, S to save screenshot")
        hint_label.SetForegroundColour(wx.Colour(128, 128, 128))
        info_sizer.Add(self.finger_count_label, 0, wx.ALL, 5)
        info_sizer.Add(hint_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.main_sizer.Add(info_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 5)

    def _capture_frame(self):
        try:
            import freenect
            frame, timestamp = freenect.sync_get_depth()
            if frame is not None:
                return True, frame
            return False, None
        except ImportError:
            if hasattr(self.capture, 'read'):
                return self.capture.read()
            return False, None

    def _process_frame(self, frame):
        if frame.dtype != np.uint8:
            np.clip(frame, 0, 2**10 - 1, frame)
            frame >>= 2
            frame = frame.astype(np.uint8)
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        height, width = frame.shape[:2]
        img_draw = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        cv2.circle(img_draw, (int(width / 2), int(height / 2)), 3, [255, 102, 0], 2)
        cv2.rectangle(img_draw, (int(width / 3), int(height / 3)),
                      (int(width * 2 / 3), int(height * 2 / 3)), [255, 102, 0], 2)

        num_fingers, img_draw = self.hand_gestures.recognize(frame)

        cv2.putText(img_draw, str(num_fingers), (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if hasattr(self, 'finger_count_label'):
            wx.CallAfter(self.finger_count_label.SetLabel, f"Fingers: {num_fingers}")

        return img_draw


def main():
    use_kinect = True
    try:
        import freenect
        frame, _ = freenect.sync_get_depth()
        if frame is None:
            print("Warning: No Kinect depth frame. Falling back to webcam.")
            use_kinect = False
    except ImportError:
        print("Warning: freenect not installed. Falling back to webcam.")
        use_kinect = False

    capture = None
    if not use_kinect:
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            print("Error: Could not open any video source.")
            return
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    app = wx.App()
    frame = wx.Frame(None, -1, "Kinect Hand Gesture Recognition", size=(680, 560))
    layout = KinectLayout(frame, -1, "Kinect Hand Gesture Recognition", capture)
    layout.Show(True)
    frame.Show(True)
    layout.SetFocus()
    app.MainLoop()

    if capture is not None:
        capture.release()


if __name__ == '__main__':
    main()