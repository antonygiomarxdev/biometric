import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


def build_siamese_network(input_shape):
    """Construye el modelo siamesa para la comparación de imágenes biométricas."""
    # Definir la subred base
    base_model = tf.keras.Sequential(
        [
            layers.Conv2D(64, (3, 3), activation="relu", input_shape=input_shape),
            layers.MaxPooling2D(),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(256, (3, 3), activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
        ]
    )

    # Entradas de las dos imágenes
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)

    # Extraer características con la misma subred
    features_a = base_model(input_a)
    features_b = base_model(input_b)

    # Calcular la distancia euclidiana entre las características
    distance = layers.Lambda(
        lambda tensors: tf.sqrt(
            tf.reduce_sum(tf.square(tensors[0] - tensors[1]), axis=-1)
        )
    )([features_a, features_b])

    # Salida final de la red: valor de similitud (0: diferente, 1: mismo)
    output = layers.Dense(1, activation="sigmoid")(distance)

    # Definir el modelo siamesa
    siamese_network = Model(inputs=[input_a, input_b], outputs=output)

    return siamese_network


def extract_features(model, images):
    """Extrae las características de las imágenes utilizando el modelo dado."""
    return model.predict(images)


def find_best_match(input_image, dataset_features, model):
    """Encuentra la mejor coincidencia para la imagen de entrada en el conjunto de datos."""
    # Extraer las características de la imagen de entrada
    input_feature = extract_features(model, np.expand_dims(input_image, axis=0))

    # Calcular la distancia entre el vector de características de la imagen de entrada y todos los vectores del dataset
    distances = np.linalg.norm(dataset_features - input_feature, axis=1)

    # Encontrar el índice de la menor distancia
    best_match_index = np.argmin(distances)
    best_match_distance = distances[best_match_index]

    return best_match_index, best_match_distance


if __name__ == "__main__":
    # Parámetros
    input_shape = (128, 128, 1)  # Cambia según el tamaño de tus imágenes
    threshold = 0.5  # Umbral de decisión para coincidencia

    # Crear y compilar el modelo siamesa
    siamese_model = build_siamese_network(input_shape)
    siamese_model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )

    # Ejemplo de imágenes y dataset (aquí deberías cargar tus datos reales)
    # `dataset_images` debería ser un array con las imágenes del dataset
    # `input_image` es la imagen que quieres comparar
    # Por simplicidad, vamos a crear datos ficticios
    dataset_images = np.random.random((100, 128, 128, 1))  # Dataset de 100 imágenes
    input_image = np.random.random((128, 128, 1))  # Imagen de entrada

    # Extraer características de todas las imágenes en el dataset
    feature_extractor = Model(
        inputs=siamese_model.input[0], outputs=siamese_model.layers[-2].output
    )
    dataset_features = extract_features(feature_extractor, dataset_images)

    # Encontrar la mejor coincidencia en el dataset para la imagen de entrada
    best_match_index, best_match_distance = find_best_match(
        input_image, dataset_features, feature_extractor
    )

    # Evaluar si la coincidencia es adecuada según el umbral
    if best_match_distance < threshold:
        print(
            f"Se encontró una coincidencia en el índice {best_match_index} con una distancia de {best_match_distance}."
        )
    else:
        print("No se encontró una coincidencia adecuada.")
