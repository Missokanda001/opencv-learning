"""
Saliency Detection using Spectral Residual (Fourier Analysis).

This module implements the Saliency class that generates saliency maps
from RGB images using the spectral residual approach in the frequency domain.

Based on "OpenCV with Python By Example" - Chapter 5.
Python 3 compatible.

Usage:
    from saliency import Saliency
    sal = Saliency(img, use_numpy_fft=True, gauss_kernel=(5, 5))
    saliency_map = sal.get_saliency_map()
    proto_objects = sal.get_proto_objects_map(use_otsu=False)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


class Saliency:
    """
    Saliency detector using the Spectral Residual algorithm.

    Computes which regions of an image are visually salient (stand out
    from their surroundings) by analyzing the log spectrum in the
    Fourier domain.

    Attributes:
        use_numpy_fft: if True, use numpy.fft; else use cv2.dft
        gauss_kernel: kernel size for Gaussian blur (None to disable)
        frame_orig: original input image
        frame_small: downscaled version (64x64) for fast computation
        saliency_map: computed saliency map (normalized 0-1 float32)
        need_saliency_map: flag for lazy computation
    """

    def __init__(self, img, use_numpy_fft=True, gauss_kernel=(5, 5)):
        """
        Initialize saliency detector with an input image.

        Args:
            img: input BGR image (numpy array)
            use_numpy_fft: use NumPy FFT (True) or OpenCV DFT (False)
            gauss_kernel: (kx, ky) Gaussian kernel size, or None to skip blur
        """
        self.use_numpy_fft = use_numpy_fft
        self.gauss_kernel = gauss_kernel
        self.frame_orig = img

        # Downscale to 64x64 for fast Fourier computation
        self.small_shape = (64, 64)
        self.frame_small = cv2.resize(img, self.small_shape[1::-1])

        # Lazy computation flag
        self.need_saliency_map = True
        self.saliency_map = None

    def _get_channel_sal_magn(self, channel):
        """
        Compute spectral residual saliency for a single image channel.

        Core algorithm:
          1. Compute FFT of the channel
          2. Get log magnitude spectrum
          3. Compute spectral residual = log_mag - blurred(log_mag)
          4. Inverse FFT to get saliency in spatial domain

        Args:
            channel: single-channel grayscale image

        Returns:
            magnitude: saliency map in spatial domain (float32)
        """
        # Step 1: Compute DFT / FFT
        if self.use_numpy_fft:
            img_dft = np.fft.fft2(channel)
            magnitude, angle = cv2.cartToPolar(
                np.real(img_dft), np.imag(img_dft))
        else:
            img_dft = cv2.dft(np.float32(channel),
                              flags=cv2.DFT_COMPLEX_OUTPUT)
            magnitude, angle = cv2.cartToPolar(
                img_dft[:, :, 0], img_dft[:, :, 1])

        # Step 2: Log amplitude (log spectrum)
        log_ampl = np.log10(magnitude.clip(min=1e-9))

        # Step 3: Average (blurred) log spectrum
        log_ampl_blur = cv2.blur(log_ampl, (3, 3))

        # Step 4: Spectral residual
        residual = np.exp(log_ampl - log_ampl_blur)

        # Step 5: Inverse FFT / DFT back to spatial domain
        if self.use_numpy_fft:
            real_part, imag_part = cv2.polarToCart(residual, angle)
            img_combined = np.fft.ifft2(real_part + 1j * imag_part)
            magnitude, _ = cv2.cartToPolar(
                np.real(img_combined), np.imag(img_combined))
        else:
            img_dft[:, :, 0], img_dft[:, :, 1] = cv2.polarToCart(
                residual, angle)
            img_combined = cv2.idft(img_dft)
            magnitude, _ = cv2.cartToPolar(
                img_combined[:, :, 0], img_combined[:, :, 1])

        return magnitude

    def get_saliency_map(self):
        """
        Generate the full saliency map for the input image.

        For grayscale images: processes the single channel.
        For color images: processes each BGR channel independently.

        Returns:
            saliency_map: float32 image, normalized to [0, 1],
                          same size as original input image.
        """
        if self.need_saliency_map:
            if len(self.frame_orig.shape) == 2:
                # Single channel (grayscale)
                sal = self._get_channel_sal_magn(self.frame_small)
            else:
                # Multi-channel (color): process each channel independently
                sal = np.zeros_like(self.frame_small).astype(np.float32)
                for c in range(self.frame_small.shape[2]):
                    sal[:, :, c] = self._get_channel_sal_magn(
                        self.frame_small[:, :, c])

            # Apply Gaussian blur to smooth the saliency map
            if self.gauss_kernel is not None:
                sal = cv2.GaussianBlur(sal, self.gauss_kernel,
                                       sigmaX=8, sigmaY=0)

            # Square and normalize to [0, 1]
            sal = sal ** 2
            sal = np.float32(sal) / np.max(sal)

            # Resize back to original image size
            sal = cv2.resize(sal, self.frame_orig.shape[1::-1])

            self.saliency_map = sal
            self.need_saliency_map = False

        return self.saliency_map

    def get_proto_objects_map(self, use_otsu=False):
        """
        Convert saliency map to a binary mask of "proto-objects".

        Proto-objects are visually salient regions detected by thresholding.

        Args:
            use_otsu: if True, use Otsu's automatic thresholding;
                      if False, use mean-based threshold.

        Returns:
            img_objects: uint8 binary mask (0 or 255) of salient regions
        """
        saliency = self.get_saliency_map()

        if use_otsu:
            _, img_objects = cv2.threshold(
                np.uint8(saliency * 255), 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            thresh = np.mean(saliency) * 255
            _, img_objects = cv2.threshold(
                np.uint8(saliency * 255), thresh, 255,
                cv2.THRESH_BINARY)

        return img_objects

    def plot_power_density(self):
        """Display the 2D power density (Fourier magnitude spectrum)."""
        if len(self.frame_orig.shape) > 2:
            frame = cv2.cvtColor(self.frame_orig, cv2.COLOR_BGR2GRAY)
        else:
            frame = self.frame_orig

        rows, cols = frame.shape[:2]
        nrows = cv2.getOptimalDFTSize(rows)
        ncols = cv2.getOptimalDFTSize(cols)
        frame = cv2.copyMakeBorder(frame, 0, ncols - cols, 0,
                                   nrows - rows, cv2.BORDER_CONSTANT,
                                   value=0)

        img_dft = np.fft.fft2(frame)
        magn = np.abs(img_dft)
        log_magn = np.log10(magn)
        spectrum = np.fft.fftshift(log_magn)

        spectrum_display = cv2.normalize(spectrum, None, 0, 255,
                                         cv2.NORM_MINMAX, cv2.CV_8U)

        cv2.imshow('Power Density (2D Fourier Spectrum)', spectrum_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def plot_power_spectrum(self):
        """Display the radially averaged power spectrum (RAPS)."""
        if len(self.frame_orig.shape) > 2:
            frame = cv2.cvtColor(self.frame_orig, cv2.COLOR_BGR2GRAY)
        else:
            frame = self.frame_orig

        rows, cols = frame.shape
        nrows = cv2.getOptimalDFTSize(rows)
        ncols = cv2.getOptimalDFTSize(cols)
        frame = cv2.copyMakeBorder(frame, 0, ncols - cols, 0,
                                   nrows - rows, cv2.BORDER_CONSTANT,
                                   value=0)

        if self.use_numpy_fft:
            img_dft = np.fft.fft2(frame)
            spectrum = np.log10(np.real(np.abs(img_dft)) ** 2)
        else:
            img_dft = cv2.dft(np.float32(frame),
                              flags=cv2.DFT_COMPLEX_OUTPUT)
            spectrum = np.log10(img_dft[:, :, 0] ** 2 +
                                img_dft[:, :, 1] ** 2)

        L = max(frame.shape)
        freqs = np.fft.fftfreq(L)[:L // 2]
        dists = np.sqrt(
            np.fft.fftfreq(frame.shape[0])[:, np.newaxis] ** 2 +
            np.fft.fftfreq(frame.shape[1]) ** 2
        )

        dcount = np.histogram(dists.ravel(), bins=freqs)[0]
        histo, bins = np.histogram(dists.ravel(),
                                   bins=freqs,
                                   weights=spectrum.ravel())

        centers = (bins[:-1] + bins[1:]) / 2
        plt.figure(figsize=(8, 5))
        plt.plot(centers, histo / dcount)
        plt.xlabel('Spatial Frequency')
        plt.ylabel('Log Power Spectrum')
        plt.title('Radially Averaged Power Spectrum (RAPS)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('power_spectrum.png', dpi=150)
        print("Power spectrum plot saved to power_spectrum.png")
        plt.show()