"""
Optimized CPU Enhancer implementation.
"""

import logging

import cv2
import numpy as np
from scipy import ndimage, signal

from src.core.config import config
from src.core.metrics import timed
from src.processing.enhancers.base import BaseEnhancer

logger = logging.getLogger("processing.enhancer")


class CpuEnhancer(BaseEnhancer):

    @timed("cpu_enhance")
    def enhance(self, img: np.ndarray, *, resize: bool = True) -> np.ndarray:
        logger.debug(
            f"Enhancement iniciado - shape: {img.shape}, dtype: {img.dtype}, min: {img.min()}, max: {img.max()}"
        )

        if resize:
            rows, cols = img.shape
            aspect_ratio = float(rows) / float(cols)
            new_rows = 350
            new_cols = int(new_rows / aspect_ratio)
            img = cv2.resize(img, (new_cols, new_rows))
            logger.debug(f"Imagen redimensionada a: {img.shape}")

        # 1. Normalization
        normim = self._normalize(img)
        logger.debug(
            f"Normalización - mean: {np.mean(normim):.3f}, std: {np.std(normim):.3f}"
        )

        # 2. Orientation
        orientim = self._ridge_orient(normim)
        logger.debug(f"Orientación calculada - shape: {orientim.shape}")

        # 3. Frequency
        mean_freq = self._ridge_freq(normim, orientim)
        logger.debug(f"Frecuencia media: {mean_freq:.3f}")

        if mean_freq <= 0 or not np.isfinite(mean_freq):
            logger.warning(
                f"Frecuencia inválida: {mean_freq}, usando valor por defecto"
            )
            mean_freq = config.enhancer_defaults.mean_freq_default

        # 4. Gabor filtering (Optimized)
        binim = self._ridge_filter(normim, orientim, mean_freq)

        result: np.ndarray = (binim * 255).astype(np.uint8)
        white_ratio = np.sum(result > 127) / result.size
        logger.info(
            f"Enhancement completado - shape: {result.shape}, píxeles blancos: {100*white_ratio:.1f}%"
        )

        ed = config.enhancer_defaults
        if white_ratio < ed.white_ratio_min or white_ratio > ed.white_ratio_max:
            logger.warning(
                f"Imagen mejorada tiene distribución extrema de píxeles: {100*white_ratio:.1f}% blancos"
            )

        return result

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        im_mean = np.mean(img)
        im_std = np.std(img)
        if im_std == 0:
            return img
        return (img - im_mean) / im_std

    def _ridge_orient(self, normim: np.ndarray) -> np.ndarray:
        _ = normim.shape[0]
        # Gaussian gradients
        sze = int(np.trunc(6 * self.config.gradient_sigma))
        if sze % 2 == 0:
            sze += 1
        gauss = cv2.getGaussianKernel(sze, self.config.gradient_sigma)
        filter_gauss = gauss * gauss.T
        fy, fx = np.gradient(filter_gauss)

        Gx = signal.convolve2d(normim, fx, mode="same")
        Gy = signal.convolve2d(normim, fy, mode="same")

        Gxx = Gx**2
        Gyy = Gy**2
        Gxy = Gx * Gy

        # Smoothing
        sze = int(np.trunc(6 * self.config.block_sigma))
        if sze % 2 == 0:
            sze += 1
        gauss = cv2.getGaussianKernel(sze, self.config.block_sigma)
        filter_gauss = gauss * gauss.T

        Gxx = ndimage.convolve(Gxx, filter_gauss)
        Gyy = ndimage.convolve(Gyy, filter_gauss)
        Gxy = 2 * ndimage.convolve(Gxy, filter_gauss)

        denom = np.sqrt(Gxy**2 + (Gxx - Gyy) ** 2) + np.finfo(float).eps
        sin2theta = Gxy / denom
        cos2theta = (Gxx - Gyy) / denom

        if self.config.orient_smooth_sigma:
            sze = int(np.trunc(6 * self.config.orient_smooth_sigma))
            if sze % 2 == 0:
                sze += 1
            gauss = cv2.getGaussianKernel(sze, self.config.orient_smooth_sigma)
            filter_gauss = gauss * gauss.T
            sin2theta = ndimage.convolve(sin2theta, filter_gauss)
            cos2theta = ndimage.convolve(cos2theta, filter_gauss)

        return np.pi / 2 + np.arctan2(sin2theta, cos2theta) / 2

    def _ridge_freq(self, normim: np.ndarray, orientim: np.ndarray) -> float:
        """Estimates the average ridge frequency."""
        rows, cols = normim.shape
        freq = np.zeros((rows, cols))

        # Configuration parameters
        blk_size = self.config.ridge_freq_blksze
        wind_size = self.config.ridge_freq_windsze
        min_wave = self.config.min_wave_length
        max_wave = self.config.max_wave_length

        for r in range(0, rows - blk_size, blk_size):
            for c in range(0, cols - blk_size, blk_size):
                blkim = normim[r : r + blk_size, c : c + blk_size]
                blkor = orientim[r : r + blk_size, c : c + blk_size]

                freq_block = self._frequest(
                    blkim, blkor, blk_size, wind_size, min_wave, max_wave
                )
                freq[r : r + blk_size, c : c + blk_size] = freq_block

        # Filter invalid frequencies and average
        freq_1d = freq.flatten()
        ind = np.where(freq_1d > 0)[0]

        if len(ind) == 0:
            return config.enhancer_defaults.mean_freq_default  # safe fallback

        non_zero_freqs = freq_1d[ind]
        mean_freq = np.mean(non_zero_freqs)
        return float(mean_freq)

    def _frequest(self, blkim: np.ndarray, blkor: np.ndarray, blk_size: int, wind_size: int, min_wave: float, max_wave: float) -> float:
        # 1. Average block orientation
        cosorient = np.mean(np.cos(2 * blkor))
        sinorient = np.mean(np.sin(2 * blkor))
        orient = np.arctan2(sinorient, cosorient) / 2

        # 2. Rotate the block to align ridges vertically
        # Rotate -orient + 90 (or simply adjust according to coordinate system)
        # ndimage.rotate rotates counter-clockwise
        rotim = ndimage.rotate(
            blkim, orient * 180 / np.pi + 90, reshape=False, mode="nearest"
        )

        # 3. Crop to avoid rotation artifacts
        cropsze = int(np.trunc(blk_size / np.sqrt(2)))
        offset = int(np.trunc((blk_size - cropsze) / 2))
        rotim = rotim[offset : offset + cropsze, offset : offset + cropsze]

        # 4. Projection on X axis (sum columns)
        proj = np.sum(rotim, axis=0)

        # 5. Dilation to find peaks
        dilation = ndimage.grey_dilation(proj, wind_size, structure=np.ones(wind_size))
        temp = np.abs(dilation - proj)

        peak_thresh = 2
        maxpts = (temp < peak_thresh) & (proj > np.mean(proj))
        maxind = np.where(maxpts)[0]

        cols_maxind = len(maxind)
        if cols_maxind < 2:
            return 0.0

        # 6. Calculate average wavelength
        wave_length = (maxind[-1] - maxind[0]) / (cols_maxind - 1)
        if min_wave <= wave_length <= max_wave:
            return 1.0 / wave_length  # type: ignore[no-any-return]

        return 0.0

    def _ridge_filter(
        self, normim: np.ndarray, orientim: np.ndarray, freq: float
    ) -> np.ndarray:
        # Vectorized Optimization: Filter Bank
        angle_inc = self.config.angle_inc
        num_filters = int(180 / angle_inc)

        # 1. Precompute filters
        filters = []
        sigmax = 1 / freq * self.config.relative_scale_factor_x
        sigmay = 1 / freq * self.config.relative_scale_factor_y
        sze = int(np.round(3 * max(sigmax, sigmay)))
        x, y = np.meshgrid(
            np.linspace(-sze, sze, 2 * sze + 1), np.linspace(-sze, sze, 2 * sze + 1)
        )

        ref_filter = np.exp(-(x**2 / sigmax**2 + y**2 / sigmay**2)) * np.cos(
            2 * np.pi * freq * x
        )

        for i in range(num_filters):
            angle = -(i * angle_inc + 90)
            filters.append(ndimage.rotate(ref_filter, angle, reshape=False))

        # 2. Convolve image with ALL filters
        filtered_layers = np.zeros((num_filters, *normim.shape))
        for i, kern in enumerate(filters):
            filtered_layers[i] = signal.convolve2d(normim, kern, mode="same")

        # 3. Select response based on pixel-wise orientation
        orient_deg = orientim * 180 / np.pi
        orient_idx = np.round(orient_deg / angle_inc).astype(int)
        orient_idx = orient_idx % num_filters

        rows, cols = normim.shape
        r_idx, c_idx = np.indices((rows, cols))
        final_img = filtered_layers[orient_idx, r_idx, c_idx]

        return final_img < self.config.ridge_filter_thresh
