"""
Base GUI Layout for Computer Vision Applications with wxPython.
Python 3 compatible.
"""

import cv2
import wx


class BaseLayout(wx.Frame):
    """
    Base class for camera-based GUI applications.
    Handles video capture, frame display, and provides hooks for
    subclasses to add custom UI elements and frame processing.
    """

    def __init__(self, parent, id, title, capture):
        wx.Frame.__init__(self, parent, id, title, size=(800, 600))
        self.capture = capture

        ret, frame = self.capture.read()
        if ret:
            self.imgHeight, self.imgWidth = frame.shape[:2]
        else:
            self.imgHeight, self.imgWidth = 480, 640

        self.panels_vertical = wx.BoxSizer(wx.VERTICAL)

        self.video_panel = wx.Panel(self, -1)
        self.bmp = wx.Bitmap.FromBuffer(self.imgWidth, self.imgHeight,
                                        frame if ret else
                                        bytearray(self.imgHeight * self.imgWidth * 3))
        self.video_ctrl = wx.StaticBitmap(self.video_panel, -1, self.bmp)
        video_sizer = wx.BoxSizer(wx.VERTICAL)
        video_sizer.Add(self.video_ctrl, 1, wx.EXPAND)
        self.video_panel.SetSizer(video_sizer)

        self._create_custom_layout()
        self._init_custom_layout()

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(self.panels_vertical, 0, wx.EXPAND | wx.ALL, border=5)
        main_sizer.Add(self.video_panel, 1, wx.EXPAND | wx.ALL, border=5)
        self.SetSizer(main_sizer)
        self.Fit()

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_next_frame, self.timer)
        self.timer.Start(30)

        self.Bind(wx.EVT_CHAR_HOOK, self._on_key)

    def _create_custom_layout(self):
        pass

    def _init_custom_layout(self):
        pass

    def _process_frame(self, frame):
        return frame

    def _on_next_frame(self, event):
        ret, frame = self.capture.read()
        if not ret:
            return
        frame = self._process_frame(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.bmp.CopyFromBuffer(frame_rgb)
        self.video_ctrl.SetBitmap(self.bmp)
        self.video_panel.Refresh()

    def _on_key(self, event):
        key_code = event.GetKeyCode()
        if key_code == wx.WXK_ESCAPE:
            self.Close()
        elif key_code == ord('S') or key_code == ord('s'):
            self._save_screenshot()
        else:
            event.Skip()

    def _save_screenshot(self):
        ret, frame = self.capture.read()
        if ret:
            processed = self._process_frame(frame)
            filename = "screenshot_{}.png".format(
                cv2.getTickCount() % 1000000)
            cv2.imwrite(filename, processed)
            print("Screenshot saved as: {}".format(filename))

    def Close(self, force=False):
        self.timer.Stop()
        self.capture.release()
        super().Close(force)