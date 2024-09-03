import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img


# Función para cargar las rutas de las imágenes y sus etiquetas
def load_image_paths_and_labels(directory):
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
def generate_pairs(images, labels):
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

        different_label = random.choice([l for l in label_to_images if l != label])
        different_image = random.choice(label_to_images[different_label])
        pairs.append((img_list[0], different_image))
        pair_labels.append(0)  # Etiqueta 0 para imágenes de clases diferentes

    return pairs, pair_labels


# Función para preprocesar las imágenes
def preprocess_image(image_path):
    image = img_to_array(
        load_img(image_path, color_mode="grayscale", target_size=(128, 128))
    )
    image = np.expand_dims(image, axis=0)
    return image


# Función para construir el modelo siamés
def build_model(input_shape):
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

    l1_distance = tf.abs(output_1 - output_2)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(l1_distance)

    model = tf.keras.Model(inputs=[input_1, input_2], outputs=output)
    return model


# Función de pérdida contrastiva
def contrastive_loss(y_true, y_pred):
    margin = 1.0
    return tf.reduce_mean(
        y_true * tf.square(y_pred)
        + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0))
    )


# Función para cargar solo las rutas de las imágenes (sin etiquetas) para evaluación
def load_image_paths(directory):
    image_paths = []
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
            ext = os.path.splitext(file)[-1].lower()
            if ext in [".jpg", ".png", ".bmp"]:
                image_paths.append(os.path.join(directory, file))
    return image_paths
