import tensorflow as tf

from src.dataset_loader import load_dataset


def test_load_dataset():
    """Prueba que el dataset se cargue correctamente."""
    # Simular un directorio de dataset
    dataset_name = "socofing"
    directory = "data/socofing/Real"

    images, image_paths = load_dataset(dataset_name, directory)

    assert len(images) > 0
    assert len(image_paths) > 0
    assert isinstance(images[0], tf.Tensor)
    assert isinstance(image_paths[0], str)
