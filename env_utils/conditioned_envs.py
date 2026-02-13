import gymnasium as gym
import numpy as np

class ConditionalStateWrapper(gym.ObservationWrapper):
    def __init__(self, env, task_inference, trajectory_limit=60, inference_frequency=20, is_oracle=False):
        super().__init__(env)
        self.is_oracle = is_oracle
        self.task_inference = task_inference
        self.context_size = task_inference.context_size
        self.context = np.zeros(self.context_size, dtype=np.float32)

        self.trajectory_buffer = []
        self.buffer_size = trajectory_limit
        self.inference_frequency = inference_frequency
        self.step_counter = 0  
        self.episode_id = 0  
        
        obs_space = self.env.observation_space
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([obs_space.low, -np.inf * np.ones(self.context_size)]),
            high=np.concatenate([obs_space.high, np.inf * np.ones(self.context_size)]),
            dtype=np.float32
        )

    @property
    def current_task(self):
        return self.env.current_task
    
    def observation(self, obs):
        return np.concatenate([obs, self.context])

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        self.trajectory_buffer.append((self.current_obs, action, reward))
        if len(self.trajectory_buffer) > self.buffer_size:
            self.trajectory_buffer.pop(0)

        self.step_counter += 1
        if self.step_counter % self.inference_frequency == 0:
            inference_input = self.env.current_task if self.is_oracle else self.trajectory_buffer
            self.context = self.task_inference.infer_task(inference_input)

        self.current_obs = next_obs
        info = {'env_info': info, 'context': self.context, 'episode_id': self.episode_id}
        return self.observation(next_obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.trajectory_buffer = []
        self.step_counter = 0  
        self.episode_id += 1  
        initial_obs, _ = self.env.reset(**kwargs)
        self.current_obs = initial_obs

        inference_input = self.env.current_task if self.is_oracle else self.trajectory_buffer
        self.context = self.task_inference.infer_task(inference_input)
        return self.observation(initial_obs), {}

    def _add_to_buffer(self, entry):
        self.trajectory_buffer.append(entry)
        if len(self.trajectory_buffer) > self.buffer_size:
            self.trajectory_buffer.pop(0)
