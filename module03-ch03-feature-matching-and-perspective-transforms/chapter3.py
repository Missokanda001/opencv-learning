import cv2
import wx
from GUI import BaseLayout
from feature_matching import FeatureMatching


class FeatureMatchingLayout(BaseLayout):
    def _init_custom_layout(self):
        # Initialize feature matcher with template image
        self.matching = FeatureMatching(train_image='salinger.jpg')

    def _process_frame(self, frame):
        success, output_frame = self.matching.match(frame)
        if success:
            return output_frame
        else:
            return frame


def main():
    capture = cv2.VideoCapture(1)
    if not capture.isOpened():
        capture.open(1)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    app = wx.App(False)
    layout = FeatureMatchingLayout(None, -1, 'Feature Matching Object Tracking', capture)
    layout.Show(True)
    app.MainLoop()


if __name__ == "__main__":
    main()