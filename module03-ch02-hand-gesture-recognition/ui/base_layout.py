"""
Base Layout class for wxPython-based computer vision applications.

Provides reusable base panel with live video display, frame capture timer,
and keyboard shortcuts (ESC to quit, S to save screenshot).

Subclasses override:
    _init_custom_layout()
    _create_custom_layout()
    _process_frame(frame)
"""

import os
import time
from datetime import datetime

import cv2
import numpy as np
import wx


class BaseLayout(wx.Panel):
    """Base panel for camera-based computer vision GUI apps."""

    def __init__(self, parent, id, title, capture):
        super(BaseLayout, self).__init__(parent, id)

        self.capture = capture
        self.frame_count = 0
        self._last_time = time.time()

        # Main sizer
        self.main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Initialize custom components (hook)
        self._init_custom_layout()

        # Create video display
        self._create_video_display()

        # Create custom layout (hook)
        self._create_custom_layout()

        self.SetSizer(self.main_sizer)
        self.Layout()

        # Frame timer (~30 FPS)
        self.frame_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_frame_timer, self.frame_timer)
        self.frame_timer.Start(33)

        # Keyboard shortcuts
        self.Bind(wx.EVT_KEY_DOWN, self._on_key_down)

    def _init_custom_layout(self):
        """Override: initialize custom components."""
        pass

    def _create_custom_layout(self):
        """Override: add custom UI elements to self.main_sizer."""
        pass

    def _create_video_display(self):
        """Create the video bitmap widget."""
        self.video_bitmap = wx.StaticBitmap(
            self, -1, wx.Bitmap(640, 480)
        )
        self.main_sizer.Add(self.video_bitmap, 1, wx.EXPAND | wx.ALL, 5)

    def _on_frame_timer(self, event):
        """Capture and display a new frame."""
        success, frame = self._capture_frame()

        if not success or frame is None:
            return

        self.frame_count += 1

        # Process frame (hook)
        display_frame = self._process_frame(frame)

        if display_frame is not None:
            self._display_frame(display_frame)

    def _capture_frame(self):
        """Capture a frame. Override for custom sources (e.g., Kinect)."""
        if hasattr(self.capture, 'read'):
            return self.capture.read()
        return False, None

    def _process_frame(self, frame):
        """Override: process frame before display."""
        return frame

    def _display_frame(self, frame):
        """Display a frame in the video bitmap."""
        # Convert to RGB
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        height, width = frame.shape[:2]

        wx_image = wx.Image(width, height, frame.tobytes())
        bitmap = wx.Bitmap(wx_image)

        self.video_bitmap.SetBitmap(bitmap)
        self.video_bitmap.SetSize(width, height)
        self.Layout()

    def _on_key_down(self, event):
        """Handle keyboard shortcuts: ESC=quit, S=save screenshot."""
        key_code = event.GetKeyCode()

        if key_code == wx.WXK_ESCAPE:
            self._on_quit()
        elif key_code == ord('S') or key_code == ord('s'):
            self._save_screenshot()
        else:
            event.Skip()

    def _save_screenshot(self):
        """Save current frame as PNG with timestamp."""
        bitmap = self.video_bitmap.GetBitmap()
        if bitmap.IsOk():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            filepath = os.path.join(os.getcwd(), filename)

            image = bitmap.ConvertToImage()
            image.SaveFile(filepath, wx.BITMAP_TYPE_PNG)
            print(f"Screenshot saved: {filepath}")

    def _on_quit(self):
        """Stop timer and close window."""
        self.frame_timer.Stop()
        top_window = self.GetTopLevelParent()
        if top_window:
            top_window.Close()

    def __del__(self):
        """Cleanup."""
        try:
            if hasattr(self, 'frame_timer') and self.frame_timer.IsRunning():
                self.frame_timer.Stop()
        except Exception:
            pass