import os
import random
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img


# Función para cargar las rutas de las imágenes y sus etiquetas
def load_image_paths_and_labels(directory: str) -> Tuple[List[str], List[str]]:
    image_paths = []
    labels = []
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
            ext = os.path.splitext(file)[-1].lower()
            if ext in [".jpg", ".png", ".bmp"]:
                image_paths.append(os.path.join(directory, file))
                labels.append(
                    file.split("__")[0]
                )  # Extraer la etiqueta del nombre del archivo
    return image_paths, labels


# Función para generar pares de imágenes (positivos y negativos) para entrenamiento
def generate_pairs(
    images: List[str], labels: List[str]
) -> Tuple[List[Tuple[str, str]], List[int]]:
    label_to_images = {}
    for img, label in zip(images, labels):
        if label not in label_to_images:
            label_to_images[label] = []
        label_to_images[label].append(img)

    pairs = []
    pair_labels = []

    for label, img_list in label_to_images.items():
        for i in range(len(img_list)):
            for j in range(i + 1, len(img_list)):
                pairs.append((img_list[i], img_list[j]))
                pair_labels.append(1)  # Etiqueta 1 para imágenes de la misma clase

        different_labels = [l for l in label_to_images if l != label]
        if different_labels:
            different_label = np.random.choice(different_labels)
            different_image = random.choice(label_to_images[different_label])
            pairs.append((img_list[0], different_image))
            pair_labels.append(0)  # Etiqueta 0 para imágenes de clases diferentes

    return pairs, pair_labels


# Capa personalizada para calcular la distancia L1
class L1DistanceLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        output_1, output_2 = inputs
        return tf.abs(output_1 - output_2)


# Función para construir el modelo siamés
def build_model(input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    base_cnn = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(64, (10, 10), activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, (7, 7), activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, (4, 4), activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(256, (4, 4), activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation="sigmoid"),
        ]
    )

    input_1 = tf.keras.layers.Input(shape=input_shape)
    input_2 = tf.keras.layers.Input(shape=input_shape)

    output_1 = base_cnn(input_1)
    output_2 = base_cnn(input_2)

    l1_distance = L1DistanceLayer()([output_1, output_2])
    output = tf.keras.layers.Dense(1, activation="sigmoid")(l1_distance)

    model = tf.keras.Model(inputs=[input_1, input_2], outputs=output)
    return model


# Función para entrenar el modelo
def train_model(dataset_name: str, directory: str, epochs: int = 10) -> None:
    images, labels = load_image_paths_and_labels(directory)

    # Generar pares de imágenes y sus etiquetas
    pairs, pair_labels = generate_pairs(images, labels)

    # Crear el modelo
    model = build_model(input_shape=(128, 128, 1))

    # Dividir las imágenes en dos entradas para el modelo siamés
    input_1 = []
    input_2 = []

    for pair in pairs:
        # Cargar las imágenes y convertirlas en arrays
        img1 = img_to_array(
            load_img(pair[0], color_mode="grayscale", target_size=(128, 128))
        )
        img2 = img_to_array(
            load_img(pair[1], color_mode="grayscale", target_size=(128, 128))
        )
        input_1.append(img1)
        input_2.append(img2)

    # Convertir las listas de imágenes a tensores de TensorFlow
    input_1 = tf.convert_to_tensor(input_1, dtype=tf.float32)
    input_2 = tf.convert_to_tensor(input_2, dtype=tf.float32)
    pair_labels = tf.convert_to_tensor(pair_labels, dtype=tf.float32)

    # Compilar y entrenar el modelo
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit([input_1, input_2], pair_labels, epochs=epochs)

    # Convertir el modelo a TFLite y guardarlo
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Guardar el modelo en formato TFLite
    with open(f"../models/{dataset_name}_model.tflite", "wb") as f:
        f.write(tflite_model)


if __name__ == "__main__":
    # Configurar el número de hilos de CPU
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(4)

    # Verificar si GPUs están disponibles y configurar
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Configurar el crecimiento de memoria de la GPU para evitar que TensorFlow consuma toda la memoria de una vez
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Entrenar el modelo con el dataset SOCOFing
    train_model("socofing", "../data/socofing/Real", epochs=10)
