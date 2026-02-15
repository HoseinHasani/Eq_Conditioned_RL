import gym
import numpy as np
from gym.spaces import Box, Discrete
from envs.lunar_wrapper import MultiTaskLunarLander

class NonStationaryLunarLander(gym.Env):
    def __init__(self, render_mode=None, gravity=-10.0, engine_power_scale=1.0):
        super().__init__()

        self.task_winds = [-12.0, +12.0]  
        self.num_tasks = len(self.task_winds)
        self.current_task_id = 0  
        
        self.render_mode = render_mode
        self.gravity = gravity
        self.engine_power_scale = engine_power_scale

        self.env = None
        self._make_env_for_task(self.current_task_id)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def _make_env_for_task(self, task_id):
        wind = self.task_winds[task_id]
        self.env = MultiTaskLunarLander(
            render_mode=self.render_mode,
            wind_force=wind,
            engine_power_scale=self.engine_power_scale,
            gravity=self.gravity
        )
        self.current_task_id = task_id

    def reset(self, *, seed=None, options=None):
        self.current_task_id = np.random.choice([0, 1])
        self._make_env_for_task(self.current_task_id)
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    @property
    def current_task(self):
        return self.current_task_id
