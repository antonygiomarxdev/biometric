import cv2
import numpy as np
from skimage.morphology import skeletonize


class MinutiaeExtractorImpl:
    def __init__(self):
        pass

    def skeletonize(self, img: np.ndarray) -> np.ndarray:
        """Aplica el algoritmo de skeletonización a una imagen binaria.

        Args:
            img (np.ndarray): Imagen binaria en escala de grises (0 y 255).

        Returns:
            np.ndarray: Imagen skeletonizada (0 y 255).
        """
        # Asegúrate de que la imagen esté binarizada correctamente (valores 0 y 255)
        cv2.imshow("Pre-Skeletonization Binarized Image", img)
        cv2.waitKey(0)

        # Convertir la imagen a binaria (0 y 1) para skeletonize
        img_bin = img // 255  # Convertir la imagen a binaria (0 y 1)

        # Aplicar el método de skeletonization usando skimage
        skeleton = skeletonize(img_bin.astype(np.uint8))

        # Multiplicar por 255 para convertir el rango de nuevo a 0 y 255
        skeleton_255 = (skeleton * 255).astype(np.uint8)

        return skeleton_255

    def enhance(self, img: np.ndarray) -> np.ndarray:
        """Aplica el filtro de Gabor para mejorar la imagen de huella dactilar."""
        # Definir parámetros del filtro de Gabor
        kernels = []
        theta_values = np.linspace(0, np.pi, 8)  # 8 orientaciones diferentes

        for theta in theta_values:
            kernel = cv2.getGaborKernel(
                (21, 21), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F
            )
            kernels.append(kernel)

        # Aplicar el filtro a la imagen
        filtered_img = np.zeros_like(img, dtype=np.float32)
        for kernel in kernels:
            filtered = cv2.filter2D(img, cv2.CV_8UC3, kernel)
            np.maximum(filtered_img, filtered, out=filtered_img)

        # Normalizar la imagen resultante para mejorar el contraste
        enhanced_img = cv2.normalize(
            filtered_img, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        return enhanced_img

    def binarize(self, img: np.ndarray) -> np.ndarray:
        """Convierte la imagen en binaria."""
        _, binarized_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
        return binarized_img

    def erode(self, img: np.ndarray) -> np.ndarray:
        """Aplica erosión a la imagen."""
        kernel = np.ones((3, 3), np.uint8)
        eroded_img = cv2.erode(img, kernel, iterations=1)
        return eroded_img
