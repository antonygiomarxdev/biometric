from src.model import create_siamese_network


def test_create_siamese_network():
    """Prueba que el modelo siamesa se cree correctamente."""
    input_shape = (128, 128, 1)
    model = create_siamese_network(input_shape)

    assert model is not None
    assert len(model.layers) > 0
    assert model.input_shape == [(None, 128, 128, 1), (None, 128, 128, 1)]
    assert model.output_shape == (None, 1)
