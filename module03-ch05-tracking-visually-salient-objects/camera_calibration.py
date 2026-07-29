"""
Camera Calibration GUI using wxPython and OpenCV.
Python 3 compatible.

Controls:
  - Click "Calibrate Camera" to start
  - Move chessboard around to capture 20 frames
  - Press ESC to close, S to save screenshot
"""

import cv2
import numpy as np
import wx
import json

from gui import BaseLayout


class CameraCalibration(BaseLayout):
    """Live camera calibration using a 9x6 chessboard pattern."""

    def _create_custom_layout(self):
        pnl = wx.Panel(self, -1)
        self.button_calibrate = wx.Button(pnl, label='Calibrate Camera')
        self.button_calibrate.Bind(wx.EVT_BUTTON, self._on_button_calibrate)
        self.status_text = wx.StaticText(pnl, label="Ready. Click button to start calibration.")

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.button_calibrate, flag=wx.RIGHT, border=10)
        hbox.Add(self.status_text, flag=wx.ALIGN_CENTER_VERTICAL)
        pnl.SetSizer(hbox)
        self.panels_vertical.Add(pnl, flag=wx.EXPAND | wx.BOTTOM | wx.TOP, border=10)

    def _init_custom_layout(self):
        self.chessboard_size = (9, 6)
        self.objp = np.zeros((np.prod(self.chessboard_size), 3), dtype=np.float32)
        self.objp[:, :2] = np.mgrid[0:self.chessboard_size[0],
                                    0:self.chessboard_size[1]].T.reshape(-1, 2)

        self.recording = False
        self.record_min_num_frames = 20
        self._reset_recording()

        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_rms = None

    def _on_button_calibrate(self, event):
        self.button_calibrate.Disable()
        self.recording = True
        self._reset_recording()
        self.status_text.SetLabel(
            "Capturing: 0/{} frames. Move chessboard around!".format(
                self.record_min_num_frames))

    def _reset_recording(self):
        self.record_cnt = 0
        self.obj_points = []
        self.img_points = []

    def _process_frame(self, frame):
        if not self.recording:
            return frame

        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.uint8)

        if self.record_cnt < self.record_min_num_frames:
            ret, corners = cv2.findChessboardCorners(
                img_gray, self.chessboard_size, None)

            if ret:
                cv2.drawChessboardCorners(frame, self.chessboard_size, corners, ret)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
                cv2.cornerSubPix(img_gray, corners, (9, 9), (-1, -1), criteria)

                self.obj_points.append(self.objp)
                self.img_points.append(corners)
                self.record_cnt += 1
                self.status_text.SetLabel(
                    "Capturing: {}/{} frames".format(self.record_cnt, self.record_min_num_frames))
        else:
            print("Calibrating...")
            self.status_text.SetLabel("Calibrating... please wait.")

            ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                self.obj_points, self.img_points,
                (self.imgWidth, self.imgHeight), None, None)

            self.calibration_rms = ret
            self.camera_matrix = K
            self.dist_coeffs = dist

            print("\n=== CALIBRATION RESULTS ===")
            print("RMS reprojection error: {:.4f} pixels".format(ret))
            print("\nCamera Matrix (K):")
            print(K)
            print("\nDistortion Coefficients (dist):")
            print(dist)
            print("===========================\n")

            self._save_calibration()

            self.status_text.SetLabel(
                "Done! RMS error: {:.3f} px. Saved to calibration.json".format(ret))
            self.button_calibrate.SetLabel("Recalibrate")
            self.button_calibrate.Enable()
            self.recording = False

        return frame

    def _save_calibration(self):
        if self.camera_matrix is None:
            return

        calib_data = {
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coefficients": self.dist_coeffs.tolist(),
            "image_width": self.imgWidth,
            "image_height": self.imgHeight,
            "rms_error": float(self.calibration_rms),
            "chessboard_size": list(self.chessboard_size),
            "num_frames_used": self.record_cnt
        }

        with open("calibration.json", 'w') as f:
            json.dump(calib_data, f, indent=2)
        print("Calibration saved to calibration.json")

        np.savez("calibration.npz",
                 camera_matrix=self.camera_matrix,
                 dist_coeffs=self.dist_coeffs,
                 image_width=self.imgWidth,
                 image_height=self.imgHeight)
        print("Calibration also saved as calibration.npz")


def main():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        capture.open(0)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    app = wx.App()
    layout = CameraCalibration(None, -1, 'Camera Calibration', capture)
    layout.Show(True)
    app.MainLoop()


if __name__ == '__main__':
    main()