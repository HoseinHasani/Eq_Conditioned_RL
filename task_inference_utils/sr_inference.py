import numpy as np
import pysindy as ps
from task_inference_utils.base_inference import BaseTaskInference

class SymbolicRegressionInference(BaseTaskInference):
    def __init__(self, context_size, feature_lib=None, optimizer=None,
                 ensemble=True, residual=True, use_rewards=False):
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

    def infer_task(self, trajectory_buffer):
        if len(trajectory_buffer) < 5:
            return np.zeros(self.context_size, dtype=np.float32)

        states, actions, rewards = zip(*trajectory_buffer)
        states = np.array(states)
        actions = np.array(actions)


        x_train = states[:-1]
        u_train = actions[:-1]
        y_train = states[1:]
        
        if self.residual:
            y_train = y_train - x_train
            
        if self.use_rewards: 
            rewards = np.array(rewards)
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