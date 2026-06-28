import cv2
import numpy as np
import utils


def outlineRect(image, rect, color):
    if rect is None:
        return
    x, y, w, h = rect
    cv2.rectangle(image, (x, y), (x + w, y + h), color)


def copyRect(src, dst, srcRect, dstRect, mask=None,
             interpolation=cv2.INTER_LINEAR):
    """Copy part of the source to part of the destination."""
    x0, y0, w0, h0 = srcRect
    x1, y1, w1, h1 = dstRect

    src_region = src[y0:y0+h0, x0:x0+w0]
    dst_region = cv2.resize(src_region, (w1, h1),
                            interpolation=interpolation)

    if mask is None:
        dst[y1:y1+h1, x1:x1+w1] = dst_region
        return

    if not utils.isGray(src):
        mask = np.repeat(mask, 3).reshape(h0, w0, 3)

    mask_resized = cv2.resize(mask, (w1, h1),
                              interpolation=cv2.INTER_NEAREST)
    dst[y1:y1+h1, x1:x1+w1] = np.where(
        mask_resized,
        dst_region,
        dst[y1:y1+h1, x1:x1+w1]
    )


def swapRects(src, dst, rects, masks=None,
              interpolation=cv2.INTER_LINEAR):
    """Copy the source with two or more sub-rectangles swapped."""
    if dst is not src:
        dst[:] = src

    numRects = len(rects)
    if numRects < 2:
        return

    if masks is None:
        masks = [None] * numRects

    # Copy the contents of the last rectangle into temporary storage.
    x, y, w, h = rects[numRects - 1]
    temp = src[y:y+h, x:x+w].copy()

    # Copy the contents of each rectangle into the next.
    for i in range(numRects - 2, -1, -1):
        copyRect(src, dst, rects[i], rects[i+1], masks[i],
                 interpolation)

    # Copy the temporarily stored content into the first rectangle.
    copyRect(temp, dst, (0, 0, w, h), rects[0], masks[numRects - 1],
             interpolation)
# Convenience examples (leave commented or remove as needed)
# swapRects(image, image, rect0, rect1)
# swapRects(image, image, rect1, rect0)