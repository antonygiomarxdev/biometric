import os
from typing import List, Tuple

from src.data_preprocessing import preprocess_image, load_image


def load_dataset(dataset_name: str, directory: str) -> Tuple[List, List[str]]:
    """Carga el dataset basado en el nombre especificado."""
    if dataset_name == "socofing":
        return load_socofing_dataset(directory)
    elif dataset_name == "another_dataset":
        return load_another_dataset(directory)
    else:
        raise ValueError(f"Dataset {dataset_name} no está soportado.")


def load_socofing_dataset(directory: str) -> Tuple[List, List[str]]:
    """Carga y preprocesa las imágenes del dataset SOCOFing."""
    image_paths: List[str] = [
        os.path.join(directory, filename) for filename in os.listdir(directory)
    ]
    images: List = [preprocess_image(load_image(path)) for path in image_paths]
    return images, image_paths


def load_another_dataset(directory: str) -> Tuple[List, List[str]]:
    """Carga y preprocesa las imágenes de otro dataset futuro."""
    # Implementa la carga específica para el otro dataset
    pass
