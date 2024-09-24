import math

import cv2
import numpy as np
from scipy import ndimage, signal
from skimage.morphology import skeletonize


class FingerprintImageEnhancerImpl:
    """Implementación del servicio de mejora de imagen de huellas dactilares."""

    def __init__(self):
        """Inicializa los parámetros para la mejora de imagen."""
        self.ridge_segment_blksze: int = 16
        self.ridge_segment_thresh: float = 0.1
        self.gradient_sigma: int = 1
        self.block_sigma: int = 7
        self.orient_smooth_sigma: int = 7
        self.ridge_freq_blksze: int = 38
        self.ridge_freq_windsze: int = 5
        self.min_wave_length: int = 5
        self.max_wave_length: int = 15
        self.relative_scale_factor_x: float = 0.65
        self.relative_scale_factor_y: float = 0.65
        self.angle_inc: int = 3
        self.ridge_filter_thresh: float = -3

        # Atributos para almacenar datos intermedios
        self._mask: np.ndarray = np.array([])
        self._normim: np.ndarray = np.array([])
        self._orientim: np.ndarray = np.array([])
        self._mean_freq: float = 0
        self._median_freq: float = 0
        self._freq: np.ndarray = np.array([])
        self._binim: np.ndarray = np.array([])

    def enhance(self, img: np.ndarray, resize: bool = True) -> np.ndarray:
        """Mejora la imagen de entrada aplicando diversos filtros y operaciones.

        Args:
            img (np.ndarray): Imagen de entrada.
            resize (bool, optional): Si se debe redimensionar la imagen. Defaults to True.

        Returns:
            np.ndarray: Imagen mejorada y esqueletonizada.
        """
        if resize:
            rows, cols = np.shape(img)
            aspect_ratio = np.double(rows) / np.double(cols)
            new_rows = 350  # Altura arbitraria para redimensionar
            new_cols = int(new_rows / aspect_ratio)
            img = cv2.resize(img, (new_cols, new_rows))

        self.__ridge_segment(img)  # Normalización y segmentación
        self.__ridge_orient()  # Cálculo de la orientación de las crestas
        self.__ridge_freq()  # Cálculo de la frecuencia de las crestas
        self.__ridge_filter()  # Filtro Gabor aplicado a las crestas

        # Retorna la imagen binarizada y esqueletonizada
        return np.uint8(self._binim * 255)

    def __normalise(self, img: np.ndarray) -> np.ndarray:
        """Normaliza la imagen para tener media cero y desviación estándar unitaria.

        Args:
            img (np.ndarray): Imagen de entrada.

        Returns:
            np.ndarray: Imagen normalizada.
        """
        if np.std(img) == 0:
            raise ValueError("La imagen tiene una desviación estándar de 0.")
        return (img - np.mean(img)) / np.std(img)

    def __ridge_segment(self, img: np.ndarray) -> None:
        """Segmenta la imagen de huellas dactilares y normaliza la intensidad en las regiones de crestas.

        Args:
            img (np.ndarray): Imagen de entrada.
        """
        rows, cols = img.shape
        normalized_im = self.__normalise(
            img
        )  # Normaliza para media 0 y desviación estándar unitaria
        new_rows = int(
            self.ridge_segment_blksze * np.ceil(rows / self.ridge_segment_blksze)
        )
        new_cols = int(
            self.ridge_segment_blksze * np.ceil(cols / self.ridge_segment_blksze)
        )

        padded_img = np.zeros((new_rows, new_cols))
        stddevim = np.zeros((new_rows, new_cols))
        padded_img[0:rows, 0:cols] = normalized_im

        for i in range(0, new_rows, self.ridge_segment_blksze):
            for j in range(0, new_cols, self.ridge_segment_blksze):
                block = padded_img[
                    i : i + self.ridge_segment_blksze, j : j + self.ridge_segment_blksze
                ]
                stddevim[
                    i : i + self.ridge_segment_blksze, j : j + self.ridge_segment_blksze
                ] = np.std(block)

        stddevim = stddevim[0:rows, 0:cols]
        self._mask = stddevim > self.ridge_segment_thresh
        mean_val = np.mean(normalized_im[self._mask])
        std_val = np.std(normalized_im[self._mask])
        self._normim = (normalized_im - mean_val) / std_val

    def __ridge_orient(self) -> None:
        """Estima la orientación local de las crestas en la imagen de huellas."""
        sze = int(np.fix(6 * self.gradient_sigma))
        if sze % 2 == 0:
            sze += 1

        gauss = cv2.getGaussianKernel(sze, self.gradient_sigma)
        filter_gauss = gauss * gauss.T

        filter_grad_y, filter_grad_x = np.gradient(
            filter_gauss
        )  # Calcula gradiente de la imagen
        gradient_x = signal.convolve2d(self._normim, filter_grad_x, mode="same")
        gradient_y = signal.convolve2d(self._normim, filter_grad_y, mode="same")

        grad_x2 = gradient_x**2
        grad_y2 = gradient_y**2
        grad_xy = gradient_x * gradient_y

        sze = int(np.fix(6 * self.block_sigma))
        gauss = cv2.getGaussianKernel(sze, self.block_sigma)
        filter_gauss = gauss * gauss.T

        grad_x2 = ndimage.convolve(grad_x2, filter_gauss)
        grad_y2 = ndimage.convolve(grad_y2, filter_gauss)
        grad_xy = 2 * ndimage.convolve(grad_xy, filter_gauss)

        denom = np.sqrt(grad_xy**2 + (grad_x2 - grad_y2) ** 2) + np.finfo(float).eps
        sin_2_theta = grad_xy / denom
        cos_2_theta = (grad_x2 - grad_y2) / denom

        if self.orient_smooth_sigma:
            sze = int(np.fix(6 * self.orient_smooth_sigma))
            if sze % 2 == 0:
                sze += 1
            gauss = cv2.getGaussianKernel(sze, self.orient_smooth_sigma)
            filter_gauss = gauss * gauss.T
            cos_2_theta = ndimage.convolve(cos_2_theta, filter_gauss)
            sin_2_theta = ndimage.convolve(sin_2_theta, filter_gauss)

        self._orientim = np.pi / 2 + np.arctan2(sin_2_theta, cos_2_theta) / 2

    def __ridge_freq(self) -> None:
        """Calcula la frecuencia de las crestas a lo largo de la imagen."""
        rows, cols = self._normim.shape
        freq = np.zeros((rows, cols))

        for i in range(0, rows - self.ridge_freq_blksze, self.ridge_freq_blksze):
            for j in range(0, cols - self.ridge_freq_blksze, self.ridge_freq_blksze):
                blkim = self._normim[
                    i : i + self.ridge_freq_blksze, j : j + self.ridge_freq_blksze
                ]
                blkor = self._orientim[
                    i : i + self.ridge_freq_blksze, j : j + self.ridge_freq_blksze
                ]
                freq[i : i + self.ridge_freq_blksze, j : j + self.ridge_freq_blksze] = (
                    self.__frequest(blkim, blkor)
                )

        self._freq = freq * self._mask
        freq_1d = self._freq.ravel()
        non_zero_elems_in_freq = freq_1d[freq_1d > 0]

        self._mean_freq = np.mean(non_zero_elems_in_freq)
        self._freq = self._mean_freq * self._mask

    def __frequest(self, blkim: np.ndarray, blkor: np.ndarray) -> np.ndarray:
        """Estima la frecuencia de las crestas en un bloque de la imagen."""
        rows, _ = blkim.shape
        cosorient = np.mean(np.cos(2 * blkor))
        sinorient = np.mean(np.sin(2 * blkor))
        orient = math.atan2(sinorient, cosorient) / 2

        rotim = ndimage.rotate(blkim, orient / np.pi * 180 + 90, reshape=False)
        cropsze = int(np.fix(rows / np.sqrt(2)))
        offset = int(np.fix((rows - cropsze) / 2))
        rotim = rotim[offset : offset + cropsze, offset : offset + cropsze]

        proj = np.sum(rotim, axis=0)
        dilation = ndimage.grey_dilation(
            proj, self.ridge_freq_windsze, structure=np.ones(self.ridge_freq_windsze)
        )
        temp = np.abs(dilation - proj)
        peak_thresh = 2

        maxpts = (temp < peak_thresh) & (proj > np.mean(proj))
        maxind = np.where(maxpts)

        if len(maxind[0]) < 2:
            return np.zeros(blkim.shape)

        wave_length = (maxind[0][-1] - maxind[0][0]) / (len(maxind[0]) - 1)
        if self.min_wave_length <= wave_length <= self.max_wave_length:
            return 1 / wave_length * np.ones(blkim.shape)
        return np.zeros(blkim.shape)

    def __ridge_filter(self) -> None:
        """Aplica filtros Gabor a la imagen orientada para resaltar las crestas."""
        rows, cols = self._normim.shape
        newim = np.zeros((rows, cols))

        freq_1d = self._freq.ravel()
        non_zero_elems_in_freq = freq_1d[freq_1d > 0]
        unfreq = np.unique(np.round(non_zero_elems_in_freq * 100) / 100)

        sigmax = 1 / unfreq[0] * self.relative_scale_factor_x
        sigmay = 1 / unfreq[0] * self.relative_scale_factor_y
        sze = int(np.round(3 * max(sigmax, sigmay)))

        mesh_x, mesh_y = np.meshgrid(
            np.linspace(-sze, sze, 2 * sze + 1), np.linspace(-sze, sze, 2 * sze + 1)
        )
        reffilter = np.exp(
            -((mesh_x**2 / sigmax**2) + (mesh_y**2 / sigmay**2))
        ) * np.cos(2 * np.pi * unfreq[0] * mesh_x)

        gabor_filter = [
            ndimage.rotate(reffilter, -(i * self.angle_inc + 90), reshape=False)
            for i in range(180 // self.angle_inc)
        ]

        maxsze = int(sze)
        validr, validc = np.where(self._freq > 0)
        valid_ind = (
            (validr > maxsze)
            & (validr < rows - maxsze)
            & (validc > maxsze)
            & (validc < cols - maxsze)
        )

        orientindex = np.round(self._orientim / np.pi * 180 / self.angle_inc).astype(
            int
        )
        orientindex = (orientindex + 180 // self.angle_inc) % (180 // self.angle_inc)

        for i, j in zip(validr[valid_ind], validc[valid_ind]):
            img_block = self._normim[i - sze : i + sze + 1, j - sze : j + sze + 1]
            newim[i, j] = np.sum(img_block * gabor_filter[orientindex[i, j]])

        self._binim = newim < self.ridge_filter_thresh

    def skeletonize(self, img: np.ndarray) -> np.ndarray:
        """Aplica esqueletización a la imagen mejorada.

        Args:
            img (np.ndarray): Imagen binaria mejorada.

        Returns:
            np.ndarray: Imagen esqueletizada.
        """
        img_bin = img // 255
        skeleton = skeletonize(img_bin.astype(np.uint8))
        return (skeleton * 255).astype(np.uint8)
