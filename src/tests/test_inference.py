import tensorflow as tf

from src.inference import make_prediction
from src.model import create_siamese_network


def test_make_prediction(monkeypatch, tmpdir):
    """Prueba que la inferencia funcione correctamente."""

    # Simular la función load_image y preprocess_image
    def mock_preprocess_image(image):
        return tf.random.uniform(shape=[128, 128, 1])

    monkeypatch.setattr(
        "src.data_preprocessing.preprocess_image", mock_preprocess_image
    )

    # Simular el modelo cargado
    def mock_load_model(model_path):
        model = create_siamese_network((128, 128, 1))
        return model

    monkeypatch.setattr("tensorflow.keras.models", "load_model", mock_load_model)

    # Simular las características del dataset
    dataset_features = tf.random.uniform(shape=[10, 512])

    best_match_index, best_match_distance = make_prediction(
        "path/to/image.bmp", dataset_features, "path/to/model.h5"
    )

    assert isinstance(best_match_index, int)
    assert isinstance(best_match_distance, float)
