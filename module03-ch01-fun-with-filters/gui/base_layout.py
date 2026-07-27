import wx
import cv2

class BaseLayout(wx.Frame):
    def __init__(self, parent, window_id, title, capture, fps=10):
        self.capture = capture
        self.fps = fps
        self.bmp = None

        ret, frame = self.capture.read()
        while not ret:
            ret, frame = self.capture.read()
        self.imgHeight, self.imgWidth = frame.shape[:2]

        wx.Frame.__init__(self, parent, window_id, title,
                          size=(self.imgWidth, self.imgHeight + 70))

        # CORRECT ORDER
        self._init_base_layout()
        self._create_base_layout()
        self._init_custom_layout()
        self._create_custom_layout()
        self.Layout()

    def _init_base_layout(self):
        self.timer = wx.Timer(self)
        self.timer.Start(int(1000.0 / self.fps))
        self.Bind(wx.EVT_TIMER, self._on_next_frame)
        self.Bind(wx.EVT_PAINT, self._on_paint)

    def _create_base_layout(self):
        self.panels_vertical = wx.BoxSizer(wx.VERTICAL)
        self.pnl = wx.Panel(self, size=(self.imgWidth, self.imgHeight))
        self.pnl.SetBackgroundColour(wx.BLACK)
        self.panels_vertical.Add(self.pnl, proportion=1, flag=wx.EXPAND)
        self.SetSizer(self.panels_vertical)
        self.Centre()

    def _on_next_frame(self, event):
        ret, frame_bgr = self.capture.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            processed_rgb = self._process_frame(frame_rgb)
            safe_frame = processed_rgb.copy()
            self.bmp = wx.Bitmap.FromBuffer(self.imgWidth, self.imgHeight, safe_frame)
            self.Refresh()

    def _on_paint(self, event):
        dc = wx.PaintDC(self.pnl)
        if self.bmp:
            dc.DrawBitmap(self.bmp, 0, 0)

    def _init_custom_layout(self):
        raise NotImplementedError()
    def _create_custom_layout(self):
        raise NotImplementedError()
    def _process_frame(self, frame_rgb):
        raise NotImplementedError()