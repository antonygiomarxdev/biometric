"""
Implementación de Enhancer en GPU usando CuPy.
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
        self.xp = get_array_module() # cupy si está habilitado
        
    @timed("gpu_enhance")
    def enhance(self, img: np.ndarray, resize: bool = True) -> np.ndarray:
        # Transferencia inicial a GPU
        img_gpu = ensure_gpu(img)
        xp = self.xp
        
        if resize:
            # Resize en GPU puede requerir cupy.ndimage o custom kernel. 
            # Por simplicidad y robustez, hacemos resize en CPU si cupy no tiene resize fácil
            # O asumimos que img ya viene con tamaño correcto.
            # Fallback a CPU para resize si es complejo
            pass 

        # 1. Normalización (Vectorizado en GPU)
        normim = self._normalize(img_gpu)
        
        # 2. Orientación
        orientim = self._ridge_orient(normim)
        
        # 3. Frecuencia (Real con FFT)
        mean_freq = self._ridge_freq(normim, orientim)
        
        # 4. Filtrado Gabor (Tensorizado)
        binim = self._ridge_filter(normim, orientim, mean_freq)
        
        # Retorno a CPU
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
        
        # Generar kernels en CPU y mover a GPU
        sze = int(np.fix(6 * self.config.gradient_sigma))
        if sze % 2 == 0: sze += 1
        gauss = to_gpu(cv2.getGaussianKernel(sze, self.config.gradient_sigma)) # cv2 returns numpy
        filter_gauss = gauss * gauss.T
        
        # Gradient manual o usando libreria
        # np.gradient equivalent in cupy?
        # Simplemente usamos filtros Sobel o derivativa de Gaussiana
        # Construccion manual de filtros derivativos
        ky, kx = xp.gradient(filter_gauss)
        
        Gx = signal.convolve2d(normim, kx, mode='same')
        Gy = signal.convolve2d(normim, ky, mode='same')
        
        Gxx = Gx**2
        Gyy = Gy**2
        Gxy = Gx*Gy
        
        # Suavizado
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
        Estima la frecuencia dominante usando FFT en GPU.
        Mucho más rápido que iterar bloques en Python.
        """
        xp = self.xp
        
        # 1. FFT 2D
        # Padding para potencia de 2 para máxima velocidad
        rows, cols = normim.shape
        padded_rows = int(2**np.ceil(np.log2(rows)))
        padded_cols = int(2**np.ceil(np.log2(cols)))
        
        fft = xp.fft.fft2(normim, s=(padded_rows, padded_cols))
        fft_shift = xp.fft.fftshift(fft)
        magnitude = xp.abs(fft_shift)
        
        # 2. Máscara circular para ignorar DC y frecuencias muy bajas/altas
        center_y, center_x = padded_rows // 2, padded_cols // 2
        y, x = xp.ogrid[:padded_rows, :padded_cols]
        r = xp.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Rango de frecuencias de interés (config)
        # Convertir longitud de onda a radio en frecuencia
        # Frecuencia f = 1/lambda
        # En FFT, radio R corresponde a frecuencia f = R / N
        # R = f * N = N / lambda
        min_r = padded_rows / self.config.max_wave_length
        max_r = padded_rows / self.config.min_wave_length
        
        mask = (r >= min_r) & (r <= max_r)
        
        # 3. Encontrar radio promedio ponderado por energía
        masked_mag = magnitude * mask
        total_energy = xp.sum(masked_mag)
        
        if total_energy == 0:
            return 0.12 # Fallback
            
        weighted_r = xp.sum(r * masked_mag) / total_energy
        
        # Convertir radio de vuelta a frecuencia espacial
        # freq = R / N
        mean_freq = weighted_r / padded_rows
        
        return float(mean_freq)

    def _ridge_filter(self, normim, orientim, freq):
        xp = self.xp
        import cupyx.scipy.ndimage as ndimage
        import cupyx.scipy.signal as signal
        
        angle_inc = self.config.angle_inc
        num_filters = int(180 / angle_inc)
        
        # Crear filtro base en CPU y mover
        sigmax = 1 / freq * self.config.relative_scale_factor_x
        sigmay = 1 / freq * self.config.relative_scale_factor_y
        sze = int(np.round(3 * max(sigmax, sigmay)))
        
        # Meshgrid en CPU luego mover
        x, y = np.meshgrid(np.linspace(-sze, sze, 2*sze+1), np.linspace(-sze, sze, 2*sze+1))
        ref_filter = np.exp(-((x**2/sigmax**2 + y**2/sigmay**2))) * np.cos(2*np.pi*freq*x)
        ref_filter_gpu = to_gpu(ref_filter)
        
        # Generar banco de filtros
        filters = []
        for i in range(num_filters):
            angle = -(i * angle_inc + 90)
            # ndimage.rotate en cupy
            filters.append(ndimage.rotate(ref_filter_gpu, angle, reshape=False))
            
        # Convolución por lotes o bucle en GPU (mucho más rápido que CPU loop)
        rows, cols = normim.shape
        filtered_layers = xp.zeros((num_filters, rows, cols), dtype=xp.float32)
        
        for i, kern in enumerate(filters):
            filtered_layers[i] = signal.convolve2d(normim, kern, mode='same')
            
        # Selección
        orient_deg = orientim * 180 / xp.pi
        orient_idx = xp.round(orient_deg / angle_inc).astype(int)
        orient_idx = orient_idx % num_filters
        
        # Advanced indexing en GPU
        r_idx, c_idx = xp.indices((rows, cols))
        final_img = filtered_layers[orient_idx, r_idx, c_idx]
        
        return final_img < self.config.ridge_filter_thresh
