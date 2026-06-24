"""ROI-aware filter application tool.

Usage examples:
    D:/project_envs/endoscopy-pano/python.exe module01-ch03-filtering-images/ROI.py BC001R035_20260205154451.jpg --filters portra,provia,velvia --out outputs
    D:/project_envs/endoscopy-pano/python.exe module01-ch03-filtering-images/ROI.py ce16b75eda6c23c49aca51055e855cf6.mp4 --filters portra,provia,velvia --out outputs --frame-step 60
    D:/project_envs/endoscopy-pano/python.exe module01-ch03-filtering-images/ROI.py --camera --filters portra,velvia --out outputs

If using camera, press SPACE to capture a frame and process it; ESC to quit.
"""

import os
import sys
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure filters module is importable (chapter 3 folder)
THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, '..'))
CH3_DIR = os.path.join(ROOT_DIR, 'module01-ch03-filtering-images')
if CH3_DIR not in sys.path:
    sys.path.insert(0, CH3_DIR)

import filters


def create_polygon_roi_mask(image_shape, polygon_points):
    """Return a boolean ROI mask for a polygon in an image."""
    if len(image_shape) == 3:
        height, width = image_shape[:2]
    else:
        height, width = image_shape

    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = np.array(polygon_points, dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 255)
    return mask.astype(bool)


def apply_roi_masked_filter(src, roi_mask, filter_obj):
    """Return a new image with filter applied inside the ROI mask.

    Args:
        src: input BGR image (numpy array)
        roi_mask: boolean 2D mask (height, width)
        filter_obj: object with method apply(src, dst)

    Returns:
        result image (numpy array)
    """
    if roi_mask.shape != src.shape[:2]:
        raise ValueError('ROI mask must match image height and width')

    temp = src.copy()
    filter_obj.apply(src.copy(), temp)

    out = src.copy()
    if src.ndim == 3:
        roi_mask_3c = np.repeat(roi_mask[:, :, None], 3, axis=2)
        out[roi_mask_3c] = temp[roi_mask_3c]
    else:
        out[roi_mask] = temp[roi_mask]
    return out


def build_filter(name):
    """Return a filter instance by short name."""
    name = name.lower()
    if name == 'portra':
        return filters.BGRPortraCurveFilter()
    if name == 'provia':
        return filters.BGRProviaCurveFilter()
    if name == 'velvia':
        return filters.BGRVelviaCurveFilter()
    if name == 'ektachrome':
        return filters.BGREktachromeCurveFilter()
    if name == 'cross':
        return filters.BGRCrossProcessCurveFilter()
    raise ValueError(f'Unknown filter: {name}')


def montage(images, labels=None, max_width=1200):
    """Create a horizontal montage of images (resize if too wide)."""
    if labels is None:
        labels = [''] * len(images)

    # Ensure same height
    heights = [img.shape[0] for img in images]
    h = min(heights)
    resized = [cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h)) if img.shape[0] != h else img for img in images]

    total_w = sum(img.shape[1] for img in resized)
    if total_w > max_width:
        scale = max_width / total_w
        resized = [cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale))) for img in resized]

    # add labels on top
    rows = []
    for img, lab in zip(resized, labels):
        img_lab = img.copy()
        if lab:
            cv2.putText(img_lab, lab, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        rows.append(img_lab)

    montage = np.hstack(rows)
    return montage


def process_image(image_path, polygon=None, filters_list=None, out_dir=None):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f'Could not open image: {image_path}')

    if polygon is None:
        # full image ROI
        roi_mask = np.ones(img.shape[:2], dtype=bool)
    else:
        roi_mask = create_polygon_roi_mask(img.shape, polygon)

    results = [img]
    labels = ['original']
    for name in filters_list:
        f = build_filter(name)
        out = apply_roi_masked_filter(img, roi_mask, f)
        results.append(out)
        labels.append(name)

        if out_dir is not None:
            out_file = Path(out_dir) / f'{Path(image_path).stem}_{name}.png'
            cv2.imwrite(str(out_file), out)
            print(f'Saved filtered image: {out_file}')

    mont = montage(results, labels)
    if out_dir is not None:
        mont_file = Path(out_dir) / f'{Path(image_path).stem}_montage.png'
        cv2.imwrite(str(mont_file), mont)
        print(f'Saved montage image: {mont_file}')

    winname = 'ROI Filters - press any key to close'
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.imshow(winname, mont)
    cv2.waitKey(0)
    cv2.destroyWindow(winname)


def capture_and_process(camera_index, polygon=None, filters_list=None, out_dir=None):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise IOError(f'Could not open camera {camera_index}')

    print('Press SPACE to capture frame, ESC to quit')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display = frame.copy()
        if polygon is not None:
            poly = np.array(polygon, dtype=np.int32)
            cv2.polylines(display, [poly], True, (0, 255, 0), 2)

        cv2.imshow('Camera - press SPACE to capture', display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == 32:
            # capture
            if filters_list:
                results = [frame]
                labels = ['original']
                for name in filters_list:
                    f = build_filter(name)
                    out = apply_roi_masked_filter(frame, create_polygon_roi_mask(frame.shape, polygon) if polygon else np.ones(frame.shape[:2], dtype=bool), f)
                    results.append(out)
                    labels.append(name)
                    if out_dir:
                        timestamp = int(time.time())
                        out_file = Path(out_dir) / f'capture_{name}_{timestamp}.png'
                        cv2.imwrite(str(out_file), out)
                        print(f'Saved camera capture: {out_file}')

                mont = montage(results, labels)
                if out_dir:
                    montage_file = Path(out_dir) / f'capture_montage_{timestamp}.png'
                    cv2.imwrite(str(montage_file), mont)
                    print(f'Saved camera montage: {montage_file}')
                cv2.namedWindow('Captured Results', cv2.WINDOW_NORMAL)
                cv2.imshow('Captured Results', mont)
                cv2.waitKey(0)
            else:
                print('No filters specified')

    cap.release()
    cv2.destroyAllWindows()


def process_video(video_path, polygon=None, filters_list=None, out_dir=None, frame_step=30):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f'Could not open video: {video_path}')

    frame_index = 0
    print('Processing video frames. Press Q in the display window to stop early.')
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_step == 0:
            if polygon is not None:
                roi_mask = create_polygon_roi_mask(frame.shape, polygon)
            else:
                roi_mask = np.ones(frame.shape[:2], dtype=bool)

            results = [frame]
            labels = ['original']
            for name in filters_list:
                f = build_filter(name)
                out = apply_roi_masked_filter(frame, roi_mask, f)
                results.append(out)
                labels.append(name)
                if out_dir:
                    out_file = Path(out_dir) / f'{Path(video_path).stem}_frame{frame_index:06d}_{name}.png'
                    cv2.imwrite(str(out_file), out)
                    print(f'Saved video frame: {out_file}')

            mont = montage(results, labels)
            if out_dir:
                mont_file = Path(out_dir) / f'{Path(video_path).stem}_frame{frame_index:06d}_montage.png'
                cv2.imwrite(str(mont_file), mont)
                print(f'Saved video montage: {mont_file}')
            cv2.namedWindow('Video Filter Results', cv2.WINDOW_NORMAL)
            cv2.imshow('Video Filter Results', mont)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()


def parse_polygon(s):
    # expects x1,y1;x2,y2;...
    pts = []
    for part in s.split(';'):
        x, y = part.split(',')
        pts.append((int(x), int(y)))
    return pts


def main():
    parser = argparse.ArgumentParser(description='Apply ROI-aware filters to an image or camera frame')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--image', '-i', help='Path to input image')
    group.add_argument('--video', '-v', help='Path to input video')
    group.add_argument('--camera', '-c', action='store_true', help='Use camera capture')
    parser.add_argument('input', nargs='?', help='Optional image/video filename in the module01-ch03-filtering-images folder')
    parser.add_argument('--polygon', '-p', help='Polygon ROI as x1,y1;x2,y2;...')
    parser.add_argument('--filters', '-f', default='portra,provia,velvia', help='Comma-separated filter names (portra, provia, velvia, ektachrome, cross)')
    parser.add_argument('--out', '-o', default='roi_outputs', help='Output directory to save results')
    parser.add_argument('--camera-index', type=int, default=0, help='Camera index for --camera')
    parser.add_argument('--frame-step', type=int, default=30, help='Process every Nth frame from a video')

    args = parser.parse_args()

    if args.input and not (args.image or args.video or args.camera):
        if args.input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            args.image = args.input
        elif args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            args.video = args.input
        else:
            parser.error('Input filename must end with an image or video extension, or use --image/--video explicitly')

    if not (args.image or args.video or args.camera or args.input):
        default_video = Path(THIS_DIR) / 'ce16b75eda6c23c49aca51055e855cf6.mp4'
        default_image = Path(THIS_DIR) / 'BC001R035_20260205154451.jpg'
        if default_video.exists():
            args.video = str(default_video)
            print(f'No input specified, defaulting to local video: {default_video.name}')
        elif default_image.exists():
            args.image = str(default_image)
            print(f'No input specified, defaulting to local image: {default_image.name}')

    if args.image and not Path(args.image).is_absolute():
        args.image = str(Path(THIS_DIR) / args.image)
    if args.video and not Path(args.video).is_absolute():
        args.video = str(Path(THIS_DIR) / args.video)

    if not (args.image or args.video or args.camera):
        parser.error('one of the arguments --image/-i --video/-v --camera/-c or a filename is required')

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    polygon = parse_polygon(args.polygon) if args.polygon else None
    filters_list = [f.strip() for f in args.filters.split(',') if f.strip()]

    if args.image:
        process_image(args.image, polygon, filters_list, out_dir)
    elif args.video:
        process_video(args.video, polygon, filters_list, out_dir, args.frame_step)
    else:
        capture_and_process(args.camera_index, polygon, filters_list, out_dir)


if __name__ == '__main__':
    main()
