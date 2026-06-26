import cv2
import filters
from managers import CaptureManager, WindowManager
import rects
from trackers import FaceTracker


class Cameo(object):
    def __init__(self):
        self._windowManager = WindowManager('Cameo', self.onKeypress)

        # Use 0 or 1 depending on your working webcam index.
        self._captureManager = CaptureManager(
            cv2.VideoCapture(1),
            self._windowManager,
            True
        )

        self._faceTracker = FaceTracker()
        self._shouldDrawDebugRects = False
        self._curveFilter = filters.BGRPortraCurveFilter()

    def run(self):
        self._windowManager.createWindow()

        if not self._captureManager._capture.isOpened():
            print("Error: Camera could not be opened.")
            return

        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame

            if frame is None:
                print("No frame received from camera.")
                break

            self._faceTracker.update(frame)
            faces = self._faceTracker.faces

            if len(faces) >= 2:
                rects.swapRects(
                    frame,
                    frame,
                    [face.faceRect for face in faces]
                )

            filters.strokeEdges(frame, frame)
            self._curveFilter.apply(frame, frame)

            if self._shouldDrawDebugRects:
                self._faceTracker.drawDebugRects(frame)

            self._captureManager.exitFrame()
            self._windowManager.processEvents()

        self._captureManager.release()
        cv2.destroyAllWindows()

    def onKeypress(self, keycode):
        if keycode == 32:  # space
            self._captureManager.writeImage('screenshot.png')
            print("Screenshot saved.")

        elif keycode == 9:  # tab
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('screencast.avi')
                print("Recording started.")
            else:
                self._captureManager.stopWritingVideo()
                print("Recording stopped.")

        elif keycode in (ord('x'), ord('X')):
            self._shouldDrawDebugRects = not self._shouldDrawDebugRects
            print("Debug rectangles:", self._shouldDrawDebugRects)

        elif keycode in (27, ord('q'), ord('Q')):  # esc or q
            self._windowManager.destroyWindow()


class CameoDouble(Cameo):
    def __init__(self):
        Cameo.__init__(self)

        # Second camera. Change 0/1/2 depending on your available cameras.
        self._hiddenCaptureManager = CaptureManager(
            cv2.VideoCapture(0)
        )

    def run(self):
        self._windowManager.createWindow()

        if not self._captureManager._capture.isOpened():
            print("Error: Main camera could not be opened.")
            return

        if not self._hiddenCaptureManager._capture.isOpened():
            print("Error: Hidden camera could not be opened.")
            return

        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            self._hiddenCaptureManager.enterFrame()

            frame = self._captureManager.frame
            hiddenFrame = self._hiddenCaptureManager.frame

            if frame is None or hiddenFrame is None:
                print("No frame received from one camera.")
                break

            self._faceTracker.update(hiddenFrame)
            hiddenFaces = self._faceTracker.faces

            self._faceTracker.update(frame)
            faces = self._faceTracker.faces

            i = 0
            while i < len(faces) and i < len(hiddenFaces):
                rects.copyRect(
                    hiddenFrame,
                    frame,
                    hiddenFaces[i].faceRect,
                    faces[i].faceRect
                )
                i += 1

            filters.strokeEdges(frame, frame)
            self._curveFilter.apply(frame, frame)

            if self._shouldDrawDebugRects:
                self._faceTracker.drawDebugRects(frame)

            self._captureManager.exitFrame()
            self._hiddenCaptureManager.exitFrame()
            self._windowManager.processEvents()

        self._captureManager.release()
        self._hiddenCaptureManager.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    Cameo().run()

    # To use two cameras instead, comment the line above and use:
    # CameoDouble().run()

