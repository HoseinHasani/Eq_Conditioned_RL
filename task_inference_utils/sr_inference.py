import numpy as np
import pysindy as ps
from task_inference_utils.base_inference import BaseTaskInference

class SymbolicRegressionInference(BaseTaskInference):
    def __init__(self, context_size, feature_lib=None, optimizer=None,
                 ensemble=True, residual=True, use_rewards=False,
                 baseline_model=None, residual_mode="none",
                 residual_clip=20.0):
        """
        Initialize the Symbolic Regression Inference module.
        
        Args:
            context_size (int): The size of the context vector.
            feature_lib: Pysindy feature library. Defaults to PolynomialLibrary with degree 1.
            optimizer: Pysindy optimizer. Defaults to STLSQ.
            ensemble (bool): Whether to use an ensemble model. Defaults to True.
            residual (bool): Whether to subtract the previous state from the target y.
            use_rewards (bool): Whether to include rewards in the target vector y. Defaults to False.
        """
        super().__init__(context_size)
        self.feature_lib = feature_lib or ps.PolynomialLibrary(degree=1, include_bias=True)
        self.optimizer = optimizer or ps.STLSQ(threshold=0.00001, alpha=0.15, verbose=False, max_iter=40)
        self.ensemble = ensemble
        self.residual = residual
        self.use_rewards = use_rewards
        self.baseline_model = baseline_model
        self.residual_mode = residual_mode
        self.residual_clip = residual_clip

    def _build_training_data(self, trajectory_buffer):
        states, actions, rewards = zip(*trajectory_buffer)
        states = np.array(states)
        actions = np.array(actions)

        x_train = states[:-1]
        u_train = actions[:-1]
        y_train = states[1:]

        if self.residual:
            y_train = y_train - x_train

        if self.residual_mode == "nn_delta":
            if self.baseline_model is None:
                raise ValueError("residual_mode='nn_delta' requires a baseline_model")
            if not self.residual:
                raise ValueError("residual_mode='nn_delta' currently requires residual=True")

            baseline_delta = self.baseline_model.predict_delta(x_train, u_train)
            if baseline_delta.shape != y_train.shape:
                raise ValueError(
                    f"Baseline prediction shape mismatch: got {baseline_delta.shape}, expected {y_train.shape}"
                )
            if not np.all(np.isfinite(baseline_delta)):
                baseline_delta = np.nan_to_num(baseline_delta, nan=0.0, posinf=0.0, neginf=0.0)

            y_train = y_train - baseline_delta
            y_train = np.clip(y_train, -self.residual_clip, self.residual_clip)

        return x_train, u_train, y_train, np.array(rewards)

    def infer_task(self, trajectory_buffer):
        if len(trajectory_buffer) < 5:
            return np.zeros(self.context_size, dtype=np.float32)

        x_train, u_train, y_train, rewards = self._build_training_data(trajectory_buffer)

        if self.use_rewards: 
            if self.residual:
                reward_vec = rewards[1:] - rewards[:-1]
            else:
                reward_vec = rewards[1:]
                
            y_train = np.concatenate([y_train, reward_vec.reshape(-1, 1)], axis=1)

        y_train = y_train[:, :4] # temporary
        # x_train = np.ones_like(x_train[:, :1]) # temporary

        model = ps.SINDy(discrete_time=True,
                         feature_library=self.feature_lib,
                         optimizer=self.optimizer)  

        if self.ensemble:
            model.fit(x_train, u=u_train, x_dot=y_train, ensemble=self.ensemble,
                  multiple_trajectories=False, n_models=7, quiet=True)
        else:
            model.fit(x_train, u=u_train, x_dot=y_train, ensemble=self.ensemble,
                  multiple_trajectories=False, quiet=True)

        if self.ensemble:
            coeffs = np.array(model.coef_list)
            coeffs = 2 * np.tanh(0.5 * coeffs)
            coeffs = coeffs.mean(0)
        else:
            coeffs = model.coefficients()
            coeffs = 2 * np.tanh(0.5 * coeffs)
            
        flat_coeffs = np.concatenate(coeffs, axis=None).astype(np.float32)

        # print(flat_coeffs.shape)
        # exit()    
    
        
        # flat_coeffs = np.clip(flat_coeffs, -1.1, 1.1)

        return flat_coeffs
