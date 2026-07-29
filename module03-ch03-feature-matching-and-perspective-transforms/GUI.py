import cv2
import wx
import numpy as np

class BaseLayout(wx.Frame):
    def __init__(self, parent, id, title, capture):
        wx.Frame.__init__(self, parent, id, title, size=(680, 540))
        self.capture = capture
        self.Bind(wx.EVT_CLOSE, self._on_close)
        self.panel = wx.Panel(self)
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.display = wx.StaticBitmap(self.panel)
        self.vbox.Add(self.display, flag=wx.EXPAND | wx.ALL, border=10)
        self.panel.SetSizer(self.vbox)

        self._init_custom_layout()
        self._create_custom_layout()

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_frame, self.timer)
        self.timer.Start(30)

    def _init_custom_layout(self):
        pass

    def _create_custom_layout(self):
        pass

    def _process_frame(self, frame):
        return frame

    def _on_frame(self, event):
        ret, frame = self.capture.read()
        if not ret:
            return
        processed_frame = self._process_frame(frame)
        rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        bitmap = wx.Bitmap.FromBuffer(w, h, rgb)
        self.display.SetBitmap(bitmap)
        self.Refresh()

    def _on_close(self, event):
        self.timer.Stop()
        self.capture.release()
        self.Destroy()