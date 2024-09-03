from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, Model


class EuclideanDistance(layers.Layer):
    def call(self, inputs):
        features_a, features_b = inputs
        return tf.sqrt(
            tf.reduce_sum(tf.square(features_a - features_b), axis=-1, keepdims=True)
        )


def create_siamese_network(input_shape: Tuple[int, int, int]) -> Model:
    """Crea y devuelve un modelo siamesa."""

    # Definir el modelo base que se utilizará para ambas ramas siamesas
    base_model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(256, (3, 3), activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
        ]
    )

    # Definir las dos entradas (una para cada imagen del par)
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)

    # Aplicar el modelo base a cada entrada
    features_a = base_model(input_a)
    features_b = base_model(input_b)

    # Calcular la distancia euclidiana entre las características de ambas entradas
    distance = layers.Lambda(
        lambda tensors: tf.sqrt(
            tf.reduce_sum(tf.square(tensors[0] - tensors[1]), axis=-1)
        )
    )([features_a, features_b])

    # Asegurarse de que la distancia es 2D para la capa Dense
    distance = layers.Reshape((1,))(distance)

    # Salida final de la red: valor de similitud (0: diferente, 1: mismo)
    output = layers.Dense(1, activation="sigmoid")(distance)

    # Crear el modelo completo
    siamese_network = Model(inputs=[input_a, input_b], outputs=output)

    return siamese_network
