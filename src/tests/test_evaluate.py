import tensorflow as tf

from src.evaluate import evaluate_model
from src.model import create_siamese_network


def test_evaluate_model(monkeypatch):
    """Prueba que el modelo se evalúe correctamente."""

    # Simular la función load_dataset para devolver datos ficticios
    def mock_load_dataset(dataset_name, directory):
        images = [tf.random.uniform(shape=[128, 128, 1]) for _ in range(10)]
        image_paths = [f"{directory}/image_{i}.bmp" for i in range(10)]
        return images, image_paths

    monkeypatch.setattr("src.dataset_loader.load_dataset", mock_load_dataset)

    # Simular el modelo cargado
    def mock_load_model(model_path):
        model = create_siamese_network((128, 128, 1))
        return model

    monkeypatch.setattr("tensorflow.keras.models", "load_model", mock_load_model)

    evaluate_model(
        "socofing", "data/socofing/Test", "../models/siamese_model_socofing.h5"
    )
    # Asegúrate de que no hay errores durante la evaluación
