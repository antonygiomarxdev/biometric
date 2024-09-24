import cv2
import numpy as np

from src.fingerprint.infrastructure.opencv.fingerprint_image_enhancer_impl import (
    FingerprintImageEnhancerImpl,
)
from src.fingerprint.infrastructure.opencv.fingerprint_minutiae_extractor_impl import (
    FingerprintMinutiaeExtractorImpl,
)


def main():
    # Cargar la imagen de huella
    img = cv2.imread(
        "C:/Users/antonygiomarx/dev/biometric/data/socofing/Test/1__M_Left_index_finger.BMP",
        cv2.IMREAD_GRAYSCALE,
    )

    if img is None:
        print("Error: La imagen no se pudo cargar correctamente.")
        return

    # Mostrar la imagen original
    cv2.imshow("Imagen original", img)

    # Mejorar la imagen
    enhancer_impl = FingerprintImageEnhancerImpl()
    enhanced_img = enhancer_impl.enhance(img)

    if enhanced_img is None or enhanced_img.size == 0:
        print("Error: No se pudo mejorar la imagen.")
        return

    # Mostrar la imagen mejorada
    cv2.imshow("Imagen mejorada", enhanced_img)

    # Skeletonizar la imagen mejorada
    skeletonized_img = enhancer_impl.skeletonize(enhanced_img)

    if skeletonized_img is None or skeletonized_img.size == 0:
        print("Error: No se pudo skeletonizar la imagen.")
        return

    # Mostrar la imagen skeletonizada
    cv2.imshow("Imagen skeletonizada", skeletonized_img)

    # Extraer minucias
    minutiae_extractor = FingerprintMinutiaeExtractorImpl()
    minutiae = minutiae_extractor.extract_minutiae(skeletonized_img)

    if not minutiae:
        print("Error: No se detectaron minucias.")
        return

    # Mostrar la cantidad de minucias encontradas
    num_terminations = sum(1 for m in minutiae if m.type == "termination")
    num_bifurcations = sum(1 for m in minutiae if m.type == "bifurcation")
    print(f"Terminaciones: {num_terminations}")
    print(f"Bifurcaciones: {num_bifurcations}")

    # Visualizar las minucias en la imagen
    display_minutiae_on_image(skeletonized_img, minutiae)


def display_minutiae_on_image(img: np.ndarray, minutiae: list) -> None:
    """Dibuja las minucias encontradas sobre la imagen de huella."""
    img_copy = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Dibujar minucias en la imagen
    for minutia in minutiae:
        if minutia.type == "termination":
            cv2.circle(
                img_copy, minutia.position, 3, (0, 0, 255), -1
            )  # Rojo para terminaciones
        elif minutia.type == "bifurcation":
            cv2.circle(
                img_copy, minutia.position, 3, (255, 0, 0), -1
            )  # Azul para bifurcaciones

    # Mostrar imagen con minucias
    cv2.imshow("Minucias detectadas", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
