import cv2
import numpy as np
import time
from pathlib import Path


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
        self._overlayCallback = None
        self._capture = capture
        self._channel = 0
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
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if self._channel != value:
            self._channel = value
            self._frame = None

    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            success, self._frame = self._capture.retrieve(self.channel)
            if not success:
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
        assert not self._enteredFrame, 'previous enterFrame() had no matching exitFrame()'
        if self._capture is not None:
            self._enteredFrame = self._capture.grab()

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
                self._fpsEstimate = float(self._framesElapsed) / timeElapsed
        self._framesElapsed += 1

        if self.previewWindowManager is not None:
            frameToShow = self._frame
            if self.shouldMirrorPreview:
                frameToShow = np.fliplr(self._frame).copy()
            if self._overlayCallback is not None:
                frameToShow = frameToShow.copy()
                self._overlayCallback(frameToShow)
            self.previewWindowManager.show(frameToShow)

        if self.isWritingImage:
            cv2.imwrite(self._imageFilename, self._frame)
            self._imageFilename = None

        self._writeVideoFrame()
        self._frame = None
        self._enteredFrame = False

    def _writeVideoFrame(self):
        if not self.isWritingVideo:
            return

        if self._videoWriter is None:
            if self._videoEncoding is None:
                fourcc = cv2.VideoWriter_fourcc(*'I420')
            else:
                fourcc = cv2.VideoWriter_fourcc(*self._videoEncoding)

            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps <= 0 and self._fpsEstimate is not None:
                fps = self._fpsEstimate
            if fps <= 0:
                fps = 30.0

            height, width = self._frame.shape[:2]
            self._videoWriter = cv2.VideoWriter(self._videoFilename, fourcc, fps, (width, height))
            if not self._videoWriter.isOpened():
                raise IOError(f'Could not open video writer for {self._videoFilename}')

        self._videoWriter.write(self._frame)

    def writeImage(self, filename):
        self._imageFilename = filename

    def startWritingVideo(self, filename, encoding='I420'):
        self._videoFilename = filename
        self._videoEncoding = encoding

    def stopWritingVideo(self):
        self._videoFilename = None
        if self._videoWriter is not None:
            self._videoWriter.release()
            self._videoWriter = None

    def release(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self.stopWritingVideo()


class Cameo(object):
    def __init__(self):
        self._windowManager = WindowManager('Cameo', self.onKeypress)
        self._captureManager = CaptureManager(
            cv2.VideoCapture(0), self._windowManager, True)
        self._captureManager._overlayCallback = self._draw_status
        self._outputFolder = Path(__file__).resolve().parent
        self._status_text = ''
        self._status_hide_time = 0.0

    def _draw_status(self, frame):
        if self._status_text and time.time() < self._status_hide_time:
            text = self._status_text
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.7
            thickness = 2
            color = (0, 255, 255)
            margin = 10
            text_size, _ = cv2.getTextSize(text, font, scale, thickness)
            x = margin
            y = frame.shape[0] - margin
            cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
        elif self._status_text:
            self._status_text = ''

    def _set_status(self, text, duration=1.5):
        self._status_text = text
        self._status_hide_time = time.time() + duration

    def run(self):
        self._windowManager.createWindow()
        if not self._captureManager._capture.isOpened():
            print('Error: could not open camera.')
            return

        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            if self._captureManager.frame is None:
                print('Warning: no frame available from camera.')
                self._windowManager.processEvents()
                continue

            self._captureManager.exitFrame()
            self._windowManager.processEvents()

        self._captureManager.release()
        cv2.destroyAllWindows()

    def onKeypress(self, keycode):
        if keycode in (32, ord('s'), ord('S')):  # space or s
            filename = self._outputFolder / f'screenshot_{int(time.time())}.png'
            self._captureManager.writeImage(str(filename))
            self._set_status('Screenshot saved')
            print('screenshot saved as', filename.name)
        elif keycode in (9, ord('r'), ord('R')):  # tab or r
            if not self._captureManager.isWritingVideo:
                filename = self._outputFolder / f'screencast_{int(time.time())}.avi'
                self._captureManager.startWritingVideo(str(filename))
                self._set_status('Recording')
                print('started recording', filename.name)
            else:
                self._captureManager.stopWritingVideo()
                self._set_status('Recording stopped')
                print('stopped recording')
        elif keycode in (27, ord('q'), ord('Q')):  # escape or q
            self._windowManager.destroyWindow()


if __name__ == '__main__':
    print('Cameo camera app started.')
    print('SPACE = screenshot, TAB = record/stop, ESC = quit')
    Cameo().run()

