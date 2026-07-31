import cv2
import wx
import time


class BaseLayout(wx.Frame):
    def __init__(self, parent, id, title, capture, fps=15):
        ret, frame = capture.read()
        if not ret or frame is None:
            frame_width, frame_height = 640, 480
        else:
            frame_height, frame_width = frame.shape[:2]

        style = wx.DEFAULT_FRAME_STYLE | wx.NO_FULL_REPAINT_ON_RESIZE
        super(BaseLayout, self).__init__(
            parent, id, title,
            size=(frame_width + 320, frame_height + 80),
            style=style
        )
        self.capture = capture
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.frame = None
        self.bmp = None

        self._init_ui()
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_next_frame, self.timer)
        self.timer.Start(int(1000.0 / self.fps))
        self.Bind(wx.EVT_CLOSE, self._on_exit)
        self.Bind(wx.EVT_CHAR_HOOK, self._on_key)

    def _init_ui(self):
        self.panel = wx.Panel(self, -1)

        # Video panel on LEFT
        self.video_panel = wx.Panel(self.panel, -1, size=(self.frame_width, self.frame_height))
        self.video_panel.Bind(wx.EVT_PAINT, self._on_paint)

        # Control panel on RIGHT (for radio buttons/buttons)
        self.control_panel = wx.Panel(self.panel, -1)
        self.control_sizer = wx.BoxSizer(wx.VERTICAL)
        self.control_panel.SetSizer(self.control_sizer)

        # Horizontal main layout: Video | Controls
        self.main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.main_sizer.Add(self.video_panel, 1, wx.EXPAND | wx.ALL, 5)
        self.main_sizer.Add(self.control_panel, 0, wx.EXPAND | wx.ALL, 5)

        self.panel.SetSizer(self.main_sizer)
        self.main_sizer.Fit(self.panel)
        self.panel.Layout()

    def _on_next_frame(self, event):
        ret, frame = self.capture.read()
        if not ret or frame is None:
            return
        frame = self._process_frame(frame)
        self.frame = frame
        self.bmp = self._cv2_to_wxbitmap(frame)
        self.video_panel.Refresh()

    def _process_frame(self, frame):
        return frame

    def _cv2_to_wxbitmap(self, frame):
        if frame is None:
            return wx.Bitmap(self.frame_width, self.frame_height)
        if len(frame.shape) == 2:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame_rgb.shape[:2]
        image = wx.Image(width, height, frame_rgb.tobytes())
        return wx.Bitmap(image)

    def _on_paint(self, event):
        dc = wx.PaintDC(self.video_panel)
        if self.bmp is not None:
            dc.DrawBitmap(self.bmp, 0, 0, True)

    def _on_key(self, event):
        key_code = event.GetKeyCode()
        if key_code == wx.WXK_ESCAPE:
            self.Close()
            return
        elif key_code == ord('S') or key_code == ord('s'):
            self._save_screenshot()
            return
        event.Skip()

    def _save_screenshot(self):
        if self.frame is None:
            print("No frame to save")
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        cv2.imwrite(filename, self.frame)
        print(f"Screenshot saved: {filename}")

    def _on_exit(self, event):
        self.timer.Stop()
        if self.capture is not None:
            self.capture.release()
        self.Destroy()