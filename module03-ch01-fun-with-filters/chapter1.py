import wx
import cv2
import numpy as np

# ---------------------- FILTER CLASSES ----------------------
class PencilSketch:
    def __init__(self, size, bg_gray):
        self.w, self.h = size
        self.bg = cv2.imread(bg_gray, cv2.IMREAD_GRAYSCALE)
        if self.bg is not None:
            self.bg = cv2.resize(self.bg, (self.w, self.h))

    def renderV2(self, img_rgb):
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_inv = 255 - img_gray
        blur = cv2.GaussianBlur(img_inv, (21, 21), 0)
        inv_blur = 255 - blur
        sketch = cv2.divide(img_gray, inv_blur, scale=256)
        if self.bg is not None:
            sketch = cv2.multiply(sketch, self.bg, scale=1 / 255.0)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)


class WarmingFilter:
    def _create_LUT_8UC1(self, x, y):
        from scipy.interpolate import UnivariateSpline
        spl = UnivariateSpline(x, y, k=2)
        return spl(np.arange(256)).astype(np.uint8)

    def __init__(self):
        self.inc = self._create_LUT_8UC1([0, 128, 255], [0, 192, 255])
        self.dec = self._create_LUT_8UC1([0, 128, 255], [0, 64, 255])

    def render(self, img_rgb):
        r, g, b = cv2.split(img_rgb)
        r = cv2.LUT(r, self.inc)
        b = cv2.LUT(b, self.dec)
        img = cv2.merge((r, g, b))
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.LUT(s, self.inc)
        return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB)


class CoolingFilter:
    def _create_LUT_8UC1(self, x, y):
        from scipy.interpolate import UnivariateSpline
        spl = UnivariateSpline(x, y, k=2)
        return spl(np.arange(256)).astype(np.uint8)

    def __init__(self):
        self.inc = self._create_LUT_8UC1([0, 128, 255], [0, 192, 255])
        self.dec = self._create_LUT_8UC1([0, 128, 255], [0, 64, 255])

    def render(self, img_rgb):
        r, g, b = cv2.split(img_rgb)
        r = cv2.LUT(r, self.dec)
        b = cv2.LUT(b, self.inc)
        img = cv2.merge((r, g, b))
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.LUT(s, self.inc)
        return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB)


class Cartoonizer:
    def render(self, img_rgb):
        numDownSamples = 2
        numBilateralFilters = 7
        img_color = img_rgb.copy()
        for _ in range(numDownSamples):
            img_color = cv2.pyrDown(img_color)
        for _ in range(numBilateralFilters):
            img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
        for _ in range(numDownSamples):
            img_color = cv2.pyrUp(img_color)

        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(img_gray, 7)
        img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 9, 2)
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
        return cv2.bitwise_and(img_color, img_edge)

# ---------------------- MAIN WINDOW ----------------------
class MainWindow(wx.Frame):
    def __init__(self, parent, title):
        super().__init__(parent, title=title, size=(640, 520))
        self.capture = None
        self.bmp = None

        # Filter texture path (ONLY pencilsketch_bg.jpg required)
        texture_path = r"D:\project_envs\endoscopy-pano\opencv-learning\module03-ch01-fun-with-filters\pencilsketch_bg.jpg"
        self.pencil_sketch = PencilSketch((640, 480), texture_path)
        self.warm_filter = WarmingFilter()
        self.cool_filter = CoolingFilter()
        self.cartoonizer = Cartoonizer()

        # GUI layout
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        self.panel_display = wx.Panel(self, size=(640, 480))
        self.panel_display.SetBackgroundColour(wx.BLACK)
        self.Bind(wx.EVT_PAINT, self.OnPaint)

        # Radio buttons
        self.panel_controls = wx.Panel(self)
        self.radio_warm = wx.RadioButton(self.panel_controls, label="Warming Filter", style=wx.RB_GROUP)
        self.radio_warm.SetValue(True)
        self.radio_cool = wx.RadioButton(self.panel_controls, label="Cooling Filter")
        self.radio_sketch = wx.RadioButton(self.panel_controls, label="Pencil Sketch")
        self.radio_cartoon = wx.RadioButton(self.panel_controls, label="Cartoon")

        ctrl_sizer = wx.BoxSizer(wx.HORIZONTAL)
        ctrl_sizer.Add(self.radio_warm, 1, wx.ALL, 5)
        ctrl_sizer.Add(self.radio_cool, 1, wx.ALL, 5)
        ctrl_sizer.Add(self.radio_sketch, 1, wx.ALL, 5)
        ctrl_sizer.Add(self.radio_cartoon, 1, wx.ALL, 5)
        self.panel_controls.SetSizer(ctrl_sizer)

        main_sizer.Add(self.panel_display, proportion=1, flag=wx.EXPAND)
        main_sizer.Add(self.panel_controls, flag=wx.EXPAND)
        self.SetSizer(main_sizer)

        # Preload blank bitmap
        blank_np = np.zeros((480, 640, 3), dtype=np.uint8)
        self.bmp = wx.Bitmap.FromBuffer(640, 480, blank_np)

        # Timer for camera frames
        self.timer = wx.Timer(self)
        self.timer.Start(33)
        self.Bind(wx.EVT_TIMER, self.OnNextFrame)
        self.Bind(wx.EVT_CLOSE, self.OnClose)

        # ONLY WEBCAM — NO static test image fallback
        self.capture = cv2.VideoCapture(1)
        if not self.capture.isOpened():
            print("ERROR: Webcam index 1 cannot be opened!")

    def OnNextFrame(self, event):
        if self.capture is None or not self.capture.isOpened():
            return

        ret, frame_bgr = self.capture.read()
        if not ret:
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Choose active filter
        if self.radio_warm.GetValue():
            out = self.warm_filter.render(frame_rgb)
        elif self.radio_cool.GetValue():
            out = self.cool_filter.render(frame_rgb)
        elif self.radio_sketch.GetValue():
            out = self.pencil_sketch.renderV2(frame_rgb)
        elif self.radio_cartoon.GetValue():
            out = self.cartoonizer.render(frame_rgb)
        else:
            out = frame_rgb

        out = out.copy()
        self.bmp = wx.Bitmap.FromBuffer(640, 480, out)
        self.Refresh()

    def OnPaint(self, event):
        try:
            dc = wx.PaintDC(self.panel_display)
            if self.bmp:
                dc.DrawBitmap(self.bmp, 0, 0)
        except Exception:
            pass

    def OnClose(self, event):
        if self.capture:
            self.capture.release()
        self.timer.Stop()
        event.Skip()


def main():
    app = wx.App(False)
    win = MainWindow(None, "Fun with Filters")
    win.Show()
    app.MainLoop()

if __name__ == "__main__":
    main()