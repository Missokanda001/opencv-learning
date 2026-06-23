import cv2
import numpy as np
import os
from pathlib import Path


def main():
    script_folder = Path(__file__).resolve().parent
    image_path = script_folder / 'key_frame_17.jpg'
    video_path = script_folder / 'BC008R002_20260204134606.mp4'
    output_path = script_folder / 'MyOutputVid.avi'

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f'Could not open image: {image_path}')

    cv2.imshow('color image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    gray_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray_image is None:
        raise FileNotFoundError(f'Could not open image: {image_path}')

    cv2.imshow('gray image', gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    color_bytes = bytearray(image.tobytes())
    gray_bytes = bytearray(gray_image.tobytes())

    height, width = gray_image.shape
    grayimage = np.frombuffer(gray_bytes, dtype=np.uint8).reshape((height, width))

    height, width, channels = image.shape
    bgrimage = np.frombuffer(color_bytes, dtype=np.uint8).reshape((height, width, channels))

    random_bytes_gray = bytearray(os.urandom(120000))
    flat_nparray_gray = np.frombuffer(random_bytes_gray, dtype=np.uint8)
    grayimage = flat_nparray_gray.reshape((300, 400))

    cv2.imshow('RandomGray', grayimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(str(script_folder / 'RandomGray.png'), grayimage)

    random_bytes_color = bytearray(os.urandom(360000))
    flat_nparray_color = np.frombuffer(random_bytes_color, dtype=np.uint8)
    bgrimage = flat_nparray_color.reshape((300, 400, 3))

    cv2.imshow('RandomColor', bgrimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(str(script_folder / 'RandomColor.png'), bgrimage)

    video_capture = cv2.VideoCapture(str(video_path))
    if not video_capture.isOpened():
        raise FileNotFoundError(f'Could not open video: {video_path}')

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)

    if fps <= 0:
        fps = 30.0
        print(f'Warning: could not read FPS from video, using default {fps}')

    fourcc = cv2.VideoWriter_fourcc(*'I420')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, size)
    if not video_writer.isOpened():
        raise IOError(f'Could not open VideoWriter for {output_path}')

    frame_count = 0
    save_frames = {400, 500, 600, 700, 800}

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        frame_count += 1
        video_writer.write(frame)

        if frame_count in save_frames:
            frame_file = script_folder / f'frame_{frame_count:06d}.png'
            cv2.imwrite(str(frame_file), frame)
            print('saved frame', frame_count, 'to', frame_file.name)

        if frame_count % 100 == 0:
            print('written', frame_count, 'frames')

    video_capture.release()
    video_writer.release()

    print('done, frames written:', frame_count)
    print('output saved to', output_path)
    print('saved snapshot frames at', sorted(save_frames))


# Capturing camera frames is moved into a helper function below.

def capture_camera_frames(camera_index=0, output_filename='MyOutputVid.avi', duration_seconds=10):
    camera_capture = cv2.VideoCapture(camera_index)
    if not camera_capture.isOpened():
        raise IOError(f'Could not open camera index {camera_index}')

    fps = camera_capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
        print(f'Warning: could not read camera FPS, using default {fps}')

    width = int(camera_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*'I420')
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, size)
    if not video_writer.isOpened():
        camera_capture.release()
        raise IOError(f'Could not open VideoWriter for {output_filename}')

    frames_to_capture = int(duration_seconds * fps)
    frame_count = 0

    while frame_count < frames_to_capture:
        success, frame = camera_capture.read()
        if not success:
            break
        cv2.imshow('camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        video_writer.write(frame)
        frame_count += 1

    camera_capture.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f'Captured {frame_count} frames from camera to {output_filename}')


def preview_camera_window(camera_index=0):
    clicked = False

    def on_mouse(event, x, y, flags, param):
        nonlocal clicked
        if event == cv2.EVENT_LBUTTONUP:
            clicked = True

    camera_capture = cv2.VideoCapture(camera_index)
    if not camera_capture.isOpened():
        raise IOError(f'Could not open camera index {camera_index}')

    cv2.namedWindow('MyWindow')
    cv2.setMouseCallback('MyWindow', on_mouse)
    print('Showing camera feed. Click window or press any key to stop.')

    success, frame = camera_capture.read()
    while success and not clicked:
        cv2.imshow('MyWindow', frame)
        key = cv2.waitKey(1)
        if key != -1:
            break
        success, frame = camera_capture.read()

    camera_capture.release()
    cv2.destroyWindow('MyWindow')


if __name__ == '__main__':
    capture_camera_frames()
    # To use the click-to-stop preview instead, replace the line above with:
    # preview_camera_window()
    