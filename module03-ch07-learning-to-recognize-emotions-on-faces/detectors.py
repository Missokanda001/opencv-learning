import cv2
import numpy as np


class FaceDetector:
    def __init__(self, face_casc='params/haarcascade_frontalface_default.xml',
                 left_eye_casc='params/haarcascade_lefteye_2splits.xml',
                 right_eye_casc='params/haarcascade_righteye_2splits.xml',
                 scale_factor=4):
        self.face_casc = cv2.CascadeClassifier(face_casc)
        if self.face_casc.empty():
            print("Warning: Could not load face cascade:", face_casc)
            raise SystemExit
        self.left_eye_casc = cv2.CascadeClassifier(left_eye_casc)
        if self.left_eye_casc.empty():
            print("Warning: Could not load left eye cascade:", left_eye_casc)
            raise SystemExit
        self.right_eye_casc = cv2.CascadeClassifier(right_eye_casc)
        if self.right_eye_casc.empty():
            print("Warning: Could not load right eye cascade:", right_eye_casc)
            raise SystemExit
        self.scale_factor = scale_factor

    def _find_biggest(self, rects):
        if len(rects) == 0:
            return rects
        areas = rects[:, 2] * rects[:, 3]
        return np.array([rects[np.argmax(areas)]])

    def detect(self, frame):
        frame_small = cv2.cvtColor(
            cv2.resize(frame, (0, 0), fx=1.0/self.scale_factor, fy=1.0/self.scale_factor),
            cv2.COLOR_BGR2GRAY
        )
        faces = self.face_casc.detectMultiScale(frame_small, 1.1, 3)
        faces = self._find_biggest(faces)
        if len(faces) == 0:
            return False, frame, None
        faces = faces * self.scale_factor
        x, y, w, h = faces[0]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 255, 0), 2)
        head = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        return True, frame, head

    def align_head(self, head):
        height, width = head.shape[:2]

        left_eye_region = head[int(0.2*height):int(0.5*height), int(0.1*width):int(0.5*width)]
        left_eyes = self.left_eye_casc.detectMultiScale(left_eye_region, 1.1, 3)
        left_eyes = self._find_biggest(left_eyes)
        left_eye_center = None
        for (xl, yl, wl, hl) in left_eyes:
            left_eye_center = np.array([0.1*width + xl + wl/2.0, 0.2*height + yl + hl/2.0])
            break

        right_eye_region = head[int(0.2*height):int(0.5*height), int(0.5*width):int(0.9*width)]
        right_eyes = self.right_eye_casc.detectMultiScale(right_eye_region, 1.1, 3)
        right_eyes = self._find_biggest(right_eyes)
        right_eye_center = None
        for (xr, yr, wr, hr) in right_eyes:
            right_eye_center = np.array([0.5*width + xr + wr/2.0, 0.2*height + yr + hr/2.0])
            break

        if left_eye_center is None or right_eye_center is None:
            return False, head

        desired_eye_x = 0.25
        desired_eye_y = 0.2
        desired_img_width = 200
        desired_img_height = desired_img_width

        eye_center = (left_eye_center + right_eye_center) / 2.0
        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        eye_angle_deg = np.arctan2(dy, dx) * 180.0 / np.pi

        eye_dist = np.sqrt(dx*dx + dy*dy)
        desired_eye_dist = (1.0 - 2.0 * desired_eye_x) * desired_img_width
        scale = desired_eye_dist / eye_dist

        rot_mat = cv2.getRotationMatrix2D(
            (float(eye_center[0]), float(eye_center[1])), eye_angle_deg, scale
        )
        rot_mat[0, 2] += desired_img_width * 0.5 - eye_center[0]
        rot_mat[1, 2] += desired_eye_y * desired_img_height - eye_center[1]

        aligned_head = cv2.warpAffine(
            head, rot_mat, (desired_img_width, desired_img_height),
            flags=cv2.INTER_CUBIC
        )
        return True, aligned_head
