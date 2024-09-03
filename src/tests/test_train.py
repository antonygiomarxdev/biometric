import tensorflow as tf

from src.train import train_model


def test_train_model(monkeypatch, tmpdir):
    """Prueba que el modelo se entrene y se guarde correctamente."""

    # Simular la función load_dataset para devolver datos ficticios
    def mock_load_dataset(dataset_name, directory):
        images = [tf.random.uniform(shape=[128, 128, 1]) for _ in range(10)]
        image_paths = [f"{directory}/image_{i}.bmp" for i in range(10)]
        return images, image_paths

    monkeypatch.setattr("src.dataset_loader.load_dataset", mock_load_dataset)

    model_path = tmpdir / "siamese_model_socofing.h5"
    train_model("socofing", "data/socofing/Real", epochs=1)

    assert model_path.exists()
