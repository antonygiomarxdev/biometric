import cv2
import numpy as np
from keras.src.utils.module_utils import scipy
from scipy import ndimage, signal
from skimage.morphology import skeletonize


class FingerprintImageEnhancerImpl:
    """Servicio de mejora de imagen para huellas dactilares."""

    def __init__(self):
        self.ridge_segment_blksze = 16
        self.ridge_segment_thresh = 0.1
        self.gradient_sigma = 1
        self.block_sigma = 7
        self.orient_smooth_sigma = 7
        self.ridge_freq_blksze = 38
        self.ridge_freq_windsze = 5
        self.min_wave_length = 5
        self.max_wave_length = 15
        self.relative_scale_factor_x = 0.65
        self.relative_scale_factor_y = 0.65
        self.angle_inc = 3
        self.ridge_filter_thresh = -3

        self._mask = []
        self._normim = []
        self._orientim = []
        self._freq = []
        self._binim = []

    def __normalise(self, img: np.ndarray) -> np.ndarray:
        """Normaliza la imagen."""
        if np.std(img) == 0:
            raise ValueError("La desviación estándar de la imagen es 0.")
        normed = (img - np.mean(img)) / np.std(img)
        return normed

    def __ridge_segment(self, img: np.ndarray) -> None:
        """Normaliza la imagen de la huella y segmenta la región de las crestas."""
        rows, cols = img.shape
        normalized_im = self.__normalise(img)

        # Redimensionar la imagen para ajustar a bloques
        new_rows = int(
            self.ridge_segment_blksze
            * np.ceil(float(rows) / float(self.ridge_segment_blksze))
        )
        new_cols = int(
            self.ridge_segment_blksze
            * np.ceil(float(cols) / float(self.ridge_segment_blksze))
        )

        padded_img = np.zeros((new_rows, new_cols))
        stddevim = np.zeros((new_rows, new_cols))
        padded_img[:rows, :cols] = normalized_im

        # Calcular la desviación estándar por bloque
        for i in range(0, new_rows, self.ridge_segment_blksze):
            for j in range(0, new_cols, self.ridge_segment_blksze):
                block = padded_img[
                    i : i + self.ridge_segment_blksze, j : j + self.ridge_segment_blksze
                ]
                stddevim[
                    i : i + self.ridge_segment_blksze, j : j + self.ridge_segment_blksze
                ] = np.std(block)

        stddevim = stddevim[:rows, :cols]
        self._mask = (
            stddevim > self.ridge_segment_thresh
        )  # Ajuste para una máscara menos restrictiva
        self._normim = (normalized_im - np.mean(normalized_im[self._mask])) / np.std(
            normalized_im[self._mask]
        )

    def __ridge_orient(self):
        """Mejorar el cálculo de la orientación de las crestas."""
        # Calcular los gradientes de la imagen normalizada
        sze = int(np.fix(6 * self.gradient_sigma))
        if sze % 2 == 0:
            sze += 1
        gauss = cv2.getGaussianKernel(sze, self.gradient_sigma)
        filter_gauss = gauss * gauss.T

        filter_grad_y, filter_grad_x = np.gradient(filter_gauss)
        gradient_x = signal.convolve2d(self._normim, filter_grad_x, mode="same")
        gradient_y = signal.convolve2d(self._normim, filter_grad_y, mode="same")

        # Calcular los productos de gradiente para las matrices de covarianza
        grad_x2 = gradient_x**2
        grad_y2 = gradient_y**2
        grad_xy = gradient_x * gradient_y

        # Suavizar la covarianza
        sze = int(np.fix(6 * self.block_sigma))
        gauss = cv2.getGaussianKernel(sze, self.block_sigma)
        filter_gauss = gauss * gauss.T

        grad_x2 = ndimage.convolve(grad_x2, filter_gauss)
        grad_y2 = ndimage.convolve(grad_y2, filter_gauss)
        grad_xy = 2 * ndimage.convolve(grad_xy, filter_gauss)

        # Solución analítica de la dirección principal
        denom = np.sqrt(grad_xy**2 + (grad_x2 - grad_y2) ** 2) + np.finfo(float).eps
        sin_2_theta = grad_xy / denom
        cos_2_theta = (grad_x2 - grad_y2) / denom

        # Suavizar el campo de orientación con mayor suavizado
        if self.orient_smooth_sigma:
            sze = int(np.fix(6 * self.orient_smooth_sigma))
            if sze % 2 == 0:
                sze += 1
            gauss = cv2.getGaussianKernel(sze, self.orient_smooth_sigma)
            filter_gauss = gauss * gauss.T
            cos_2_theta = ndimage.convolve(cos_2_theta, filter_gauss)
            sin_2_theta = ndimage.convolve(sin_2_theta, filter_gauss)

        # Convertir el ángulo en radianes
        self._orientim = np.pi / 2 + np.arctan2(sin_2_theta, cos_2_theta) / 2

    def __ridge_freq(self) -> None:
        """Calcula la frecuencia de las crestas en la imagen."""

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

                freq_block = self.__frequest(blkim, blkor)
                freq[i : i + self.ridge_freq_blksze, j : j + self.ridge_freq_blksze] = (
                    freq_block
                )

        self._freq = freq * self._mask
        self._mean_freq = np.mean(self._freq[self._freq > 0])
        self._freq[self._freq == 0] = (
            self._mean_freq
        )  # Rellenar con la frecuencia promedio en áreas faltantes

    def __frequest(self, blkim: np.ndarray, blkor: np.ndarray) -> np.ndarray:
        """Estimación de la frecuencia de crestas."""
        rows, _ = np.shape(blkim)
        cosorient = np.mean(np.cos(2 * blkor))
        sinorient = np.mean(np.sin(2 * blkor))
        orient = np.arctan2(sinorient, cosorient) / 2

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
        _, cols_maxind = np.shape(maxind)

        if cols_maxind < 2:
            return np.zeros(blkim.shape)
        wave_length = (maxind[0][-1] - maxind[0][0]) / (cols_maxind - 1)
        if self.min_wave_length <= wave_length <= self.max_wave_length:
            return 1 / wave_length * np.ones(blkim.shape)
        return np.zeros(blkim.shape)

    def __ridge_filter(self):
        norm_im = np.double(self._normim)
        rows, cols = norm_im.shape
        newim = np.zeros((rows, cols))

        freq_1d = np.reshape(self._freq, (1, rows * cols))
        ind = np.where(freq_1d > 0)

        ind = np.array(ind)
        ind = ind[1, :]

        non_zero_elems_in_freq = freq_1d[0][ind]
        non_zero_elems_in_freq = (
            np.double(np.round((non_zero_elems_in_freq * 100))) / 100
        )

        unfreq = np.unique(non_zero_elems_in_freq)

        sigmax = 1 / unfreq[0] * self.relative_scale_factor_x
        sigmay = 1 / unfreq[0] * self.relative_scale_factor_y

        sze = int(np.round(3 * np.max([sigmax, sigmay])))

        mesh_x, mesh_y = np.meshgrid(
            np.linspace(-sze, sze, (2 * sze + 1)), np.linspace(-sze, sze, (2 * sze + 1))
        )

        reffilter = np.exp(
            -(
                (np.power(mesh_x, 2)) / (sigmax * sigmax)
                + (np.power(mesh_y, 2)) / (sigmay * sigmay)
            )
        ) * np.cos(2 * np.pi * unfreq[0] * mesh_x)

        filt_rows, filt_cols = reffilter.shape

        angle_range = int(180 / self.angle_inc)

        gabor_filter = np.zeros((angle_range, filt_rows, filt_cols))

        for filter_idx in range(angle_range):
            rot_filt = scipy.ndimage.rotate(
                reffilter, -(filter_idx * self.angle_inc + 90), reshape=False
            )
            gabor_filter[filter_idx] = rot_filt

        maxsze = int(sze)

        temp = self._freq > 0
        valid_r, valid_c = np.where(temp)

        temp1 = valid_r > maxsze
        temp2 = valid_r < rows - maxsze
        temp3 = valid_c > maxsze
        temp4 = valid_c < cols - maxsze

        final_temp = temp1 & temp2 & temp3 & temp4

        finalind = np.where(final_temp)[
            0
        ]  # Obtenemos los índices en una única dimensión

        # Comprobar si finalind está vacío
        if len(finalind) == 0:
            return  # Si no hay índices válidos, no hacemos nada

        orientindex = np.round(self._orientim / np.pi * 180 / self.angle_inc).astype(
            int
        )
        maxorientindex = np.round(180 / self.angle_inc).astype(int)

        for i in range(rows):
            for j in range(cols):
                if orientindex[i][j] < 1:
                    orientindex[i][j] += maxorientindex
                if orientindex[i][j] > maxorientindex:
                    orientindex[i][j] -= maxorientindex

        sze = int(sze)

        for k in range(len(finalind)):  # Recorremos los índices válidos
            cur_r = valid_r[finalind[k]]
            cur_c = valid_c[finalind[k]]

            # Comprobar si los índices están dentro de los límites y corregir
            if (
                cur_r - sze >= 0
                and cur_r + sze + 1 <= rows
                and cur_c - sze >= 0
                and cur_c + sze + 1 <= cols
            ):
                img_block = norm_im[
                    cur_r - sze : cur_r + sze + 1, cur_c - sze : cur_c + sze + 1
                ]

                # Comprobar si las dimensiones de img_block son correctas
                if (
                    img_block.shape
                    == gabor_filter[int(orientindex[cur_r][cur_c]) - 1].shape
                ):
                    newim[cur_r][cur_c] = np.sum(
                        img_block * gabor_filter[int(orientindex[cur_r][cur_c]) - 1]
                    )

        self._binim = newim < self.ridge_filter_thresh

    def enhance(self, img: np.ndarray, resize: bool = True) -> np.ndarray:
        """Mejorar la imagen de entrada.

        Args:
            img (np.ndarray): Imagen de entrada.
            resize (bool, optional): Redimensionar la imagen de entrada. Defaults to True.

        Returns:
            np.ndarray: Imagen mejorada.
        """
        if resize:
            rows, cols = np.shape(img)
            aspect_ratio = np.double(rows) / np.double(cols)

            new_rows = 350  # número seleccionado aleatoriamente
            new_cols = int(new_rows / aspect_ratio)

            img = cv2.resize(img, (new_cols, new_rows))

        # Normalización
        self.__ridge_segment(img)
        cv2.imshow("Imagen Normalizada", self._normim)
        cv2.waitKey(0)

        # Orientación de crestas
        self.__ridge_orient()
        cv2.imshow("Orientación de Crestas", self._orientim)
        cv2.waitKey(0)

        # Frecuencia de crestas
        self.__ridge_freq()
        cv2.imshow("Frecuencia de Crestas", self._freq)
        cv2.waitKey(0)

        # Máscara de segmentación de crestas
        cv2.imshow("Máscara de Segmentación", self._mask.astype(np.uint8) * 255)
        cv2.waitKey(0)

        # Filtro Gabor
        self.__ridge_filter()
        cv2.imshow("Imagen Filtrada", self._binim.astype(np.uint8) * 255)
        cv2.waitKey(0)

        return np.uint8(self._binim * 255)

    def skeletonize(self, img: np.ndarray) -> np.ndarray:
        """Aplica skeletonización a la imagen mejorada."""
        img_bin = img // 255
        skeleton = skeletonize(img_bin.astype(np.uint8))
        skeleton_255 = (skeleton * 255).astype(np.uint8)
        return skeleton_255

    def save_enhanced_image(self, path: str) -> None:
        """Guarda la imagen mejorada."""
        cv2.imwrite(path, (255 * self._binim))
