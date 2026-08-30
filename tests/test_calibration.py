import numpy as np

from src.calibration import TemperatureScaler, expected_calibration_error, softmax


def test_temperature_scaler_is_positive_and_preserves_argmax():
    logits = np.array(
        [
            [4.0, 1.0, 0.0],
            [0.1, 3.0, 0.2],
            [0.3, 0.4, 2.0],
            [2.0, 0.5, 0.1],
        ]
    )
    y_true = np.array([0, 1, 2, 0])

    scaler = TemperatureScaler().fit(logits, y_true)
    scaled_logits = scaler.transform_logits(logits)

    assert scaler.temperature > 0
    assert np.array_equal(logits.argmax(axis=1), scaled_logits.argmax(axis=1))


def test_softmax_and_ece_outputs_are_valid():
    probs = softmax(np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 0.0]]))
    ece = expected_calibration_error(np.array([2, 0]), probs, n_bins=5)

    assert np.allclose(probs.sum(axis=1), 1.0)
    assert 0.0 <= ece <= 1.0

