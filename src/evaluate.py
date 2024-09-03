import multiprocessing as mp
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def load_image_paths(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".bmp", ".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(root, file))
    return image_paths


def preprocess_image(image_path):
    img = load_img(image_path, color_mode="grayscale", target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype("float32") / 255.0
    return img_array


def predict(interpreter, input_data_1, input_data_2):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepara las entradas
    interpreter.set_tensor(input_details[0]["index"], input_data_1)
    interpreter.set_tensor(input_details[1]["index"], input_data_2)

    # Realiza la inferencia
    interpreter.invoke()

    # Obtén la salida
    output_data = interpreter.get_tensor(output_details[0]["index"])
    return output_data[0]


def evaluate_image_pair(interpreter_path, test_image, dataset_image_path):
    interpreter = tf.lite.Interpreter(model_path=interpreter_path)
    interpreter.allocate_tensors()

    # Preprocesar la imagen del dataset
    dataset_image = preprocess_image(dataset_image_path)

    # Hacer la predicción
    prediction = predict(interpreter, test_image, dataset_image)
    return dataset_image_path, prediction


def evaluate_against_dataset_multiprocessing(
    tflite_model_path, test_image_path, dataset_dir
):
    # Procesar la imagen de prueba
    test_image = preprocess_image(test_image_path)

    # Obtener todas las rutas de las imágenes del dataset
    dataset_image_paths = load_image_paths(dataset_dir)

    # Usar Pool para manejar procesos en paralelo
    with mp.Pool(mp.cpu_count()) as pool:
        # Se ejecuta evaluate_image_pair en paralelo para cada imagen del dataset
        results = pool.starmap(
            evaluate_image_pair,
            [
                (tflite_model_path, test_image, dataset_image_path)
                for dataset_image_path in dataset_image_paths
            ],
        )

    # Selecciona la mejor coincidencia (menor distancia)
    best_match = min(results, key=lambda x: x[1])
    print(f"Best match: {best_match[0]} with score: {best_match[1]}")
    return best_match[0], best_match[1]


if __name__ == "__main__":
    evaluate_against_dataset_multiprocessing(
        "../models/socofing_model.tflite",
        "../data/socofing/Test/1__M_Left_index_finger.BMP",
        "../data/socofing/Real",
    )
