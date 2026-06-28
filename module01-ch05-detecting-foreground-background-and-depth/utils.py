import numpy as np
import scipy.interpolate
import depth


def createCurveFunc(points):
    if points is None:
        return None

    numPoints = len(points)
    if numPoints < 2:
        return None

    xs, ys = zip(*points)
    kind = 'linear' if numPoints < 4 else 'cubic'

    return scipy.interpolate.interp1d(
        xs, ys, kind, bounds_error=False, fill_value=(ys[0], ys[-1])
    )


def createLookupArray(func, length=256):
    if func is None:
        return None

    lookupArray = np.empty(length, dtype=np.uint8)
    for i in range(length):
        func_i = func(i)
        lookupArray[i] = min(max(0, func_i), length - 1)

    return lookupArray


def applyLookupArray(lookupArray, src, dst):
    if lookupArray is None:
        return

    dst[:] = lookupArray[src]


def createCompositeFunc(func0, func1):
    if func0 is None:
        return func1
    if func1 is None:
        return func0
    return lambda x: func0(func1(x))


def flatView(array):
    flatView = array.view()
    flatView.shape = array.size
    return flatView


def createFlatView(array):
    return flatView(array)


def isGray(image):
    return image.ndim < 3


def widthHeightDividedBy(image, divisor):
    h, w = image.shape[:2]
    return (int(w / divisor), int(h / divisor))