import cv2
from pathlib import Path
import rects
import utils


class Face(object):
    def __init__(self):
        self.faceRect = None
        self.leftEyeRect = None
        self.rightEyeRect = None
        self.noseRect = None
        self.mouthRect = None


class FaceTracker(object):
    def __init__(self, scaleFactor=1.2, minNeighbors=2,
                 flags=cv2.CASCADE_SCALE_IMAGE):
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.flags = flags
        self._faces = []

        module_dir = Path(__file__).resolve().parent
        cascade_dir = module_dir / "cascades"

        self._faceClassifier = cv2.CascadeClassifier(
            str(cascade_dir / "haarcascade_frontalface_alt.xml")
        )

        self._eyeClassifier = cv2.CascadeClassifier(
            str(cascade_dir / "haarcascade_eye.xml")
        )

        if self._faceClassifier.empty():
            raise IOError("Could not load face cascade XML file.")

        if self._eyeClassifier.empty():
            self._eyeClassifier = None

        self._noseClassifier = None
        self._mouthClassifier = None

    @property
    def faces(self):
        return self._faces

    def update(self, image):
        self._faces = []

        if image is None:
            return

        if utils.isGray(image):
            gray = cv2.equalizeHist(image)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

        minSize = utils.widthHeightDividedBy(gray, 8)

        faceRects = self._faceClassifier.detectMultiScale(
            gray,
            scaleFactor=self.scaleFactor,
            minNeighbors=self.minNeighbors,
            flags=self.flags,
            minSize=minSize
        )

        for faceRect in faceRects:
            face = Face()
            face.faceRect = faceRect

            x, y, w, h = faceRect

            searchRect = (
                int(x + w / 7),
                int(y),
                int(w * 2 / 7),
                int(h / 2)
            )
            face.leftEyeRect = self._detectOneObject(
                self._eyeClassifier, gray, searchRect, 64
            )

            searchRect = (
                int(x + w * 4 / 7),
                int(y),
                int(w * 2 / 7),
                int(h / 2)
            )
            face.rightEyeRect = self._detectOneObject(
                self._eyeClassifier, gray, searchRect, 64
            )

            self._faces.append(face)

    def _detectOneObject(self, classifier, image, rect,
                         imageSizeToMinSizeRatio):
        if classifier is None:
            return None

        x, y, w, h = [int(v) for v in rect]

        if w <= 0 or h <= 0:
            return None

        subImage = image[y:y + h, x:x + w]

        if subImage.size == 0:
            return None

        minSize = utils.widthHeightDividedBy(
            subImage, imageSizeToMinSizeRatio
        )

        subRects = classifier.detectMultiScale(
            subImage,
            scaleFactor=self.scaleFactor,
            minNeighbors=self.minNeighbors,
            flags=self.flags,
            minSize=minSize
        )

        if len(subRects) == 0:
            return None

        subX, subY, subW, subH = subRects[0]
        return (x + subX, y + subY, subW, subH)

    def drawDebugRects(self, image):
        if utils.isGray(image):
            faceColor = 255
            leftEyeColor = 255
            rightEyeColor = 255
        else:
            faceColor = (255, 255, 255)
            leftEyeColor = (0, 0, 255)
            rightEyeColor = (0, 255, 255)

        for face in self.faces:
            rects.outlineRect(image, face.faceRect, faceColor)
            rects.outlineRect(image, face.leftEyeRect, leftEyeColor)
            rects.outlineRect(image, face.rightEyeRect, rightEyeColor)