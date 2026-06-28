import cv2
import numpy as np
import time
import os


class WindowManager(object):
    def __init__(self, windowName, keypressCallback=None):
        self.keypressCallback = keypressCallback
        self._windowName = windowName
        self._isWindowCreated = False

    @property
    def isWindowCreated(self):
        return self._isWindowCreated

    def createWindow(self):
        cv2.namedWindow(self._windowName)
        self._isWindowCreated = True

    def show(self, frame):
        cv2.imshow(self._windowName, frame)

    def destroyWindow(self):
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False

    def processEvents(self):
        keycode = cv2.waitKey(1)
        if self.keypressCallback is not None and keycode != -1:
            keycode &= 0xFF
            self.keypressCallback(keycode)


class CaptureManager(object):
    def __init__(self, capture, previewWindowManager=None,
                 shouldMirrorPreview=False):
        self.previewWindowManager = previewWindowManager
        self.shouldMirrorPreview = shouldMirrorPreview
        self._capture = capture
        self._enteredFrame = False
        self._frame = None
        self._imageFilename = None
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None
        self._startTime = None
        self._framesElapsed = 0
        self._fpsEstimate = None

    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            success, self._frame = self._capture.read()
            if not success:
                self._frame = None
        return self._frame

    @property
    def isWritingImage(self):
        return self._imageFilename is not None

    @property
    def isWritingVideo(self):
        return self._videoFilename is not None

    def enterFrame(self):
        assert not self._enteredFrame, (
            "previous enterFrame() had no matching exitFrame()"
        )
        self._enteredFrame = True

    def exitFrame(self):
        if not self._enteredFrame:
            return

        if self.frame is None:
            self._enteredFrame = False
            return

        if self._framesElapsed == 0:
            self._startTime = time.time()
        else:
            timeElapsed = time.time() - self._startTime
            if timeElapsed > 0:
                self._fpsEstimate = self._framesElapsed / timeElapsed

        self._framesElapsed += 1

        if self.previewWindowManager is not None:
            frameToShow = self._frame

            if self.shouldMirrorPreview:
                frameToShow = np.fliplr(self._frame).copy()

            self.previewWindowManager.show(frameToShow)

        if self.isWritingImage:
            cv2.imwrite(self._imageFilename, self._frame)
            print("Screenshot saved to:", self._imageFilename)
            self._imageFilename = None

        self._writeVideoFrame()

        self._frame = None
        self._enteredFrame = False

    def _writeVideoFrame(self):
        if not self.isWritingVideo:
            return

        if self._videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(*self._videoEncoding)

            fps = self._capture.get(cv2.CAP_PROP_FPS)

            if fps <= 0 and self._fpsEstimate is not None:
                fps = self._fpsEstimate

            if fps <= 0:
                fps = 30.0

            height, width = self._frame.shape[:2]

            self._videoWriter = cv2.VideoWriter(
                self._videoFilename,
                fourcc,
                fps,
                (width, height)
            )

            print("Video Writer:", self._videoFilename)
            print("Resolution:", width, "x", height)
            print("FPS:", fps)
            print("Opened:", self._videoWriter.isOpened())

            if not self._videoWriter.isOpened():
                raise IOError(
                    f"Could not open video writer for {self._videoFilename}"
                )

        self._videoWriter.write(self._frame)

    def writeImage(self, filename):
        folder = os.path.dirname(os.path.abspath(__file__))
        self._imageFilename = os.path.join(folder, os.path.basename(filename))

    def startWritingVideo(self, filename, encoding="XVID"):
        folder = os.path.dirname(os.path.abspath(__file__))
        self._videoFilename = os.path.join(folder, os.path.basename(filename))
        self._videoEncoding = encoding
        print("Recording to:", self._videoFilename)

    def stopWritingVideo(self):
        self._videoFilename = None

        if self._videoWriter is not None:
            self._videoWriter.release()
            self._videoWriter = None
            print("Recording saved successfully.")

    def release(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None

        self.stopWritingVideo()

