import os
import tempfile
import unittest

import numpy as np

from task_inference_utils.nn_dynamics_baseline import NNDynamicsBaseline, train_baseline_model
from task_inference_utils.sr_inference import SymbolicRegressionInference


class DummyBaseline:
    def __init__(self, delta):
        self.delta = np.asarray(delta, dtype=np.float32)

    def predict_delta(self, states, actions):
        return np.tile(self.delta, (states.shape[0], 1))


class TestNNDynamicsBaseline(unittest.TestCase):
    def test_train_save_load_roundtrip_shapes(self):
        rng = np.random.default_rng(0)
        states = rng.normal(size=(128, 4)).astype(np.float32)
        actions = rng.normal(size=(128, 2)).astype(np.float32)
        deltas = (states[:, :4] * 0.1 + np.pad(actions, ((0, 0), (0, 2)), mode="constant")).astype(np.float32)
        dataset = {"states": states, "actions": actions, "deltas": deltas}
        config = {"hidden_dims": [16, 16], "epochs": 2, "batch_size": 32, "lr": 1e-3}

        model, _ = train_baseline_model(dataset=dataset, config=config, env_id="test-env")
        pred = model.predict_delta(states[:3], actions[:3])
        self.assertEqual(pred.shape, (3, 4))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "baseline.pt")
            model.save(path)
            loaded = NNDynamicsBaseline.load(path)
            loaded_pred = loaded.predict_delta(states[:3], actions[:3])
            self.assertEqual(loaded_pred.shape, (3, 4))


class TestResidualSymbolicInference(unittest.TestCase):
    def test_residual_target_uses_nn_subtraction(self):
        baseline = DummyBaseline(delta=[0.25, -0.25])
        infer = SymbolicRegressionInference(
            context_size=8,
            residual=True,
            residual_mode="nn_delta",
            baseline_model=baseline,
        )
        trajectory = [
            (np.array([0.0, 0.0]), np.array([1.0]), 0.0),
            (np.array([1.0, 1.0]), np.array([1.0]), 0.0),
            (np.array([2.0, 2.0]), np.array([1.0]), 0.0),
            (np.array([3.0, 3.0]), np.array([1.0]), 0.0),
            (np.array([4.0, 4.0]), np.array([1.0]), 0.0),
            (np.array([5.0, 5.0]), np.array([1.0]), 0.0),
        ]

        x_train, _, y_train, _ = infer._build_training_data(trajectory)
        observed_delta = np.diff(np.array([t[0] for t in trajectory]), axis=0)
        expected = observed_delta - np.array([0.25, -0.25], dtype=np.float32)

        np.testing.assert_allclose(x_train.shape, (5, 2))
        np.testing.assert_allclose(y_train, expected, atol=1e-6)

    def test_backward_compat_without_baseline(self):
        infer = SymbolicRegressionInference(context_size=8, residual=True, residual_mode="none")
        trajectory = [
            (np.array([0.0, 1.0]), np.array([0.0]), 0.0),
            (np.array([1.0, 3.0]), np.array([0.0]), 0.0),
            (np.array([2.0, 6.0]), np.array([0.0]), 0.0),
            (np.array([3.0, 10.0]), np.array([0.0]), 0.0),
            (np.array([4.0, 15.0]), np.array([0.0]), 0.0),
            (np.array([5.0, 21.0]), np.array([0.0]), 0.0),
        ]

        _, _, y_train, _ = infer._build_training_data(trajectory)
        expected = np.diff(np.array([t[0] for t in trajectory]), axis=0)
        np.testing.assert_allclose(y_train, expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
