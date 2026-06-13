"""
GPU Enhancer implementation using CuPy.
"""
import numpy as np
import logging
import cv2

from src.processing.enhancers.base import BaseEnhancer, EnhancerConfig
from src.core.gpu_utils import get_array_module, to_cpu, to_gpu, GPUConfig, ensure_gpu
from src.core.metrics import timed

logger = logging.getLogger(__name__)

class GpuEnhancer(BaseEnhancer):
    
    def __init__(self, config: EnhancerConfig):
        super().__init__(config)
        self.xp = get_array_module() # cupy if enabled
        
    @timed("gpu_enhance")
    def enhance(self, img: np.ndarray, resize: bool = True) -> np.ndarray:
        # Initial transfer to GPU
        img_gpu = ensure_gpu(img)
        xp = self.xp
        
        if resize:
            # Resize on GPU may require cupy.ndimage or custom kernel.
            # For simplicity and robustness, we resize on CPU if cupy doesn't have easy resize
            # Or we assume img already comes with the correct size.
            # Fallback to CPU for resize if complex
            pass 

        # 1. Normalization (Vectorized on GPU)
        normim = self._normalize(img_gpu)
        
        # 2. Orientation
        orientim = self._ridge_orient(normim)
        
        # 3. Frequency (Real with FFT)
        mean_freq = self._ridge_freq(normim, orientim)
        
        # 4. Gabor filtering (Tensorized)
        binim = self._ridge_filter(normim, orientim, mean_freq)
        
        # Return to CPU
        return np.uint8(to_cpu(binim) * 255)

    def _normalize(self, img):
        xp = self.xp
        im_mean = xp.mean(img)
        im_std = xp.std(img)
        if im_std == 0:
            return img
        return (img - im_mean) / im_std

    def _ridge_orient(self, normim):
        xp = self.xp
        import cupyx.scipy.signal as signal
        import cupyx.scipy.ndimage as ndimage
        
        # Generate kernels on CPU and move to GPU
        sze = int(np.fix(6 * self.config.gradient_sigma))
        if sze % 2 == 0: sze += 1
        gauss = to_gpu(cv2.getGaussianKernel(sze, self.config.gradient_sigma)) # cv2 returns numpy
        filter_gauss = gauss * gauss.T
        
        # Manual gradient or using library
        # np.gradient equivalent in cupy?
        # We simply use Sobel filters or Gaussian derivative
        # Manual construction of derivative filters
        ky, kx = xp.gradient(filter_gauss)
        
        Gx = signal.convolve2d(normim, kx, mode='same')
        Gy = signal.convolve2d(normim, ky, mode='same')
        
        Gxx = Gx**2
        Gyy = Gy**2
        Gxy = Gx*Gy
        
        # Smoothing
        sze = int(np.fix(6 * self.config.block_sigma))
        if sze % 2 == 0: sze += 1
        gauss = to_gpu(cv2.getGaussianKernel(sze, self.config.block_sigma))
        filter_gauss = gauss * gauss.T
        
        Gxx = ndimage.convolve(Gxx, filter_gauss)
        Gyy = ndimage.convolve(Gyy, filter_gauss)
        Gxy = 2 * ndimage.convolve(Gxy, filter_gauss)
        
        denom = xp.sqrt(Gxy**2 + (Gxx - Gyy)**2) + 1e-6
        sin2theta = Gxy / denom
        cos2theta = (Gxx - Gyy) / denom
        
        orientim = xp.pi/2 + xp.arctan2(sin2theta, cos2theta)/2
        return orientim

    def _ridge_freq(self, normim, orientim) -> float:
        """
        Estimates the dominant frequency using FFT on GPU.
        Much faster than iterating blocks in Python.
        """
        xp = self.xp
        
        # 1. FFT 2D
        # Pad to power of 2 for maximum speed
        rows, cols = normim.shape
        padded_rows = int(2**np.ceil(np.log2(rows)))
        padded_cols = int(2**np.ceil(np.log2(cols)))
        
        fft = xp.fft.fft2(normim, s=(padded_rows, padded_cols))
        fft_shift = xp.fft.fftshift(fft)
        magnitude = xp.abs(fft_shift)
        
        # 2. Circular mask to ignore DC and very low/high frequencies
        center_y, center_x = padded_rows // 2, padded_cols // 2
        y, x = xp.ogrid[:padded_rows, :padded_cols]
        r = xp.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Frequency range of interest (config)
        # Convert wavelength to frequency radius
        # Frequency f = 1/lambda
        # In FFT, radius R corresponds to frequency f = R / N
        # R = f * N = N / lambda
        min_r = padded_rows / self.config.max_wave_length
        max_r = padded_rows / self.config.min_wave_length
        
        mask = (r >= min_r) & (r <= max_r)
        
        # 3. Find energy-weighted average radius
        masked_mag = magnitude * mask
        total_energy = xp.sum(masked_mag)
        
        if total_energy == 0:
            return 0.12 # Fallback
            
        weighted_r = xp.sum(r * masked_mag) / total_energy
        
        # Convert radius back to spatial frequency
        # freq = R / N
        mean_freq = weighted_r / padded_rows
        
        return float(mean_freq)

    def _ridge_filter(self, normim, orientim, freq):
        xp = self.xp
        import cupyx.scipy.ndimage as ndimage
        import cupyx.scipy.signal as signal
        
        angle_inc = self.config.angle_inc
        num_filters = int(180 / angle_inc)
        
        # Create base filter on CPU and move
        sigmax = 1 / freq * self.config.relative_scale_factor_x
        sigmay = 1 / freq * self.config.relative_scale_factor_y
        sze = int(np.round(3 * max(sigmax, sigmay)))
        
        # Meshgrid on CPU then move
        x, y = np.meshgrid(np.linspace(-sze, sze, 2*sze+1), np.linspace(-sze, sze, 2*sze+1))
        ref_filter = np.exp(-((x**2/sigmax**2 + y**2/sigmay**2))) * np.cos(2*np.pi*freq*x)
        ref_filter_gpu = to_gpu(ref_filter)
        
        # Generate filter bank
        filters = []
        for i in range(num_filters):
            angle = -(i * angle_inc + 90)
            # ndimage.rotate in cupy
            filters.append(ndimage.rotate(ref_filter_gpu, angle, reshape=False))
            
        # Batch convolution or GPU loop (much faster than CPU loop)
        rows, cols = normim.shape
        filtered_layers = xp.zeros((num_filters, rows, cols), dtype=xp.float32)
        
        for i, kern in enumerate(filters):
            filtered_layers[i] = signal.convolve2d(normim, kern, mode='same')
            
        # Selection
        orient_deg = orientim * 180 / xp.pi
        orient_idx = xp.round(orient_deg / angle_inc).astype(int)
        orient_idx = orient_idx % num_filters
        
        # Advanced indexing on GPU
        r_idx, c_idx = xp.indices((rows, cols))
        final_img = filtered_layers[orient_idx, r_idx, c_idx]
        
        return final_img < self.config.ridge_filter_thresh
