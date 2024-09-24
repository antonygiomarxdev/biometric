import cv2

from src.fingerprint.application.services.image_processing_service import (
    ImageProcessingService,
)
from src.fingerprint.infrastructure.opencv.fingerprint_image_enhancer_impl import (
    FingerprintImageEnhancerImpl,
)


def main():
    # Cargar la imagen
    img = cv2.imread(
        "C:/Users/antonygiomarx/dev/biometric/data/socofing/Test/1__M_Left_index_finger.BMP",
        cv2.IMREAD_GRAYSCALE,
    )

    # Validar si la imagen se ha cargado correctamente
    if img is None:
        print("Error: La imagen no se pudo cargar.")
        return

    print(f"Dimensiones de la imagen original: {img.shape}, Tipo: {img.dtype}")
    cv2.imshow("Imagen original", img)
    cv2.waitKey(0)

    # Crear las instancias del servicio de mejora y procesamiento de imagen
    enhancer_impl = FingerprintImageEnhancerImpl()
    processing_service = ImageProcessingService(enhancer_impl)

    # Realzar la imagen
    enhanced_img = enhancer_impl.enhance(img)
    print(
        f"Dimensiones de la imagen mejorada: {enhanced_img.shape}, Tipo: {enhanced_img.dtype}"
    )
    cv2.imshow("Imagen mejorada", enhanced_img)
    cv2.waitKey(0)

    # Binarizar la imagen
    binarized_img = processing_service.binarize_image(enhanced_img)
    print(
        f"Dimensiones de la imagen binarizada: {binarized_img.shape}, Tipo: {binarized_img.dtype}"
    )
    cv2.imshow("Imagen binarizada", binarized_img)
    cv2.waitKey(0)

    # Aplicar erosión
    eroded_img = processing_service.erode_image(binarized_img)
    print(
        f"Dimensiones de la imagen erosionada: {eroded_img.shape}, Tipo: {eroded_img.dtype}"
    )
    cv2.imshow("Imagen erosionada", eroded_img)
    cv2.waitKey(0)

    # Skeletonizar la imagen
    skeletonized_img = processing_service.skeletonize_image(eroded_img)
    print(
        f"Dimensiones de la imagen skeletonizada: {skeletonized_img.shape}, Tipo: {skeletonized_img.dtype}"
    )
    cv2.imshow("Imagen skeletonizada", skeletonized_img)
    cv2.waitKey(0)

    # Cerrar las ventanas después de mostrar todas las imágenes
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
