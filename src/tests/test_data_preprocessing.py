import tensorflow as tf

from src.data_preprocessing import (
    generate_pairs_and_labels,
    create_tf_dataset,
    same_person,
    preprocess_image,
    load_image,
)


def test_same_person():
    """Prueba la función same_person que verifica si dos imágenes pertenecen a la misma persona."""
    path1 = "100__M_Left_index_finger.BMP"
    path2 = "100__M_Right_thumb_finger.BMP"
    path3 = "101__F_Left_index_finger.BMP"

    assert same_person(path1, path2) == True
    assert same_person(path1, path3) == False


def test_generate_pairs_and_labels():
    """Prueba que los pares de imágenes y las etiquetas se generen correctamente."""
    image_paths = [
        "100__M_Left_index_finger.BMP",
        "100__M_Right_thumb_finger.BMP",
        "101__F_Left_index_finger.BMP",
    ]

    pairs, labels = generate_pairs_and_labels(image_paths)

    assert len(pairs) == 3  # 3 pares posibles
    assert len(labels) == 3
    assert labels[0] == 1  # Las dos primeras imágenes pertenecen a la misma persona
    assert labels[1] == 0  # Diferentes personas
    assert labels[2] == 0  # Diferentes personas


def test_create_tf_dataset():
    """Prueba que el tf.data.Dataset se cree correctamente."""
    image_paths = [
        "100__M_Left_index_finger.BMP",
        "100__M_Right_thumb_finger.BMP",
        "101__F_Left_index_finger.BMP",
    ]

    pairs, labels = generate_pairs_and_labels(image_paths)
    dataset = create_tf_dataset(pairs, labels, batch_size=2)

    for batch in dataset:
        (images_a, images_b), batch_labels = batch
        assert images_a.shape == (2, 128, 128, 1)
        assert images_b.shape == (2, 128, 128, 1)
        assert batch_labels.shape == (2,)
        break  # Solo probar el primer batch para verificar las dimensiones


def test_preprocess_image():
    """Prueba que el preprocesamiento de la imagen funcione correctamente."""
    image = tf.random.uniform(shape=[256, 256, 1], maxval=255, dtype=tf.int32)
    processed_image = preprocess_image(image)

    assert processed_image.shape == (128, 128, 1)
    assert processed_image.numpy().min() >= 0.0
    assert processed_image.numpy().max() <= 1.0


def test_load_image(tmp_path):
    """Prueba que la imagen se cargue correctamente desde el disco."""
    img_path = tmp_path / "test_image.bmp"
    image_data = tf.random.uniform(
        shape=[256, 256, 1], maxval=255, dtype=tf.uint8
    ).numpy()
    tf.io.write_file(str(img_path), tf.io.encode_bmp(image_data))

    image = load_image(str(img_path))

    assert isinstance(image, tf.Tensor)
    assert image.shape == (256, 256, 1)
