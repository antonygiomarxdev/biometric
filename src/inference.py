from typing import Tuple

import numpy as np
import tensorflow as tf

from src.data_preprocessing import preprocess_image, load_image
from src.model import create_siamese_network


def extract_features(model: tf.keras.Model, images: np.ndarray) -> np.ndarray:
    """Extrae características de las imágenes usando el modelo dado."""
    return model.predict(images)


def find_best_match(
    input_image: tf.Tensor, dataset_features: np.ndarray, model: tf.keras.Model
) -> Tuple[int, float]:
    """Encuentra la mejor coincidencia para la imagen de entrada en el conjunto de datos."""
    input_feature = extract_features(model, np.expand_dims(input_image, axis=0))
    distances = np.linalg.norm(dataset_features - input_feature, axis=1)
    best_match_index = np.argmin(distances)
    best_match_distance = distances[best_match_index]
    return best_match_index, best_match_distance


def make_prediction(
    image_path: str, dataset_features: np.ndarray, model_path: str
) -> Tuple[int, float]:
    """Realiza una predicción para la imagen dada contra un conjunto de datos."""
    model = tf.keras.models.load_model(model_path)
    image = preprocess_image(load_image(image_path))
    best_match_index, best_match_distance = find_best_match(
        image, dataset_features, model
    )
    return best_match_index, best_match_distance


if __name__ == "__main__":
    # Cargar el modelo y el conjunto de características del dataset
    model_path = "../models/siamese_model_socofing.h5"
    dataset_images = np.random.random(
        (100, 128, 128, 1)
    )  # Ejemplo con datos aleatorios
    model = create_siamese_network((128, 128, 1))
    model.load_weights(model_path)
    dataset_features = extract_features(model, dataset_images)

    # Predicción para una imagen nueva
    image_path = "../data/sample.jpg"  # Ruta a la imagen de prueba
    best_match_index, best_match_distance = make_prediction(
        image_path, dataset_features, model_path
    )

    print(
        f"La mejor coincidencia es la imagen en el índice {best_match_index} con una distancia de {best_match_distance}."
    )
