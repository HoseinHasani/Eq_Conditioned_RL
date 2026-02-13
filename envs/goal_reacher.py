from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import ctypes


class GoalReacherEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, seed=0, max_pos=100, goal_rad=0.08, n_tasks=3,
                 step_length=8, max_step=500, wind_power=1.5, neverending=False,
                 disable_goal=True, apply_noise=True, noise_power=0.005):
        
        self.max_pos = max_pos
        self.step_length = step_length
        self.goal_rad = goal_rad * max_pos
        self.env_rad = max_pos
        self.wind_power = wind_power
        self.apply_noise = apply_noise
        self.noise_power = noise_power
        self.max_step = max_step
        self.neverending = neverending
        self.disable_goal = disable_goal
        
        self.time = 0
        self.cur_pos = None
        self.goal_pos = None
        self.task_vectors = self._generate_task_vectors(n_tasks) if n_tasks > 1 else [np.array([0.0, 0.0])]
        self.task_id = np.random.randint(0, n_tasks)
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        self.seed(seed)
        self.reset()

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, task_ind=None):
        if seed is not None:
            self.seed(seed)

        self.start_pos = self._random_position_within_radius(self.env_rad)
        self.cur_pos = self.start_pos if not self.neverending or self.cur_pos is None else self.cur_pos
        self.goal_pos = self._random_position_within_radius(self.env_rad)
        self.task_id = np.random.randint(0, len(self.task_vectors)) if task_ind is None else task_ind
        self.current_task_vector = self.task_vectors[self.task_id]
        self.time = 0

        initial_state = self._get_state(self.cur_pos, self.goal_pos)
        return initial_state, {}

    def step(self, action):
        
        self.current_task_vector = self.task_vectors[self.task_id]
        
        dx, dy = self.step_length * action
        base_step = np.array([dx, dy])
        noise_step = self.step_length * np.random.normal(0, self.noise_power, 2) if self.apply_noise else np.zeros(2)
        task_step = self.wind_power * self.step_length * self.current_task_vector

        new_pos = self.cur_pos + base_step + task_step + noise_step
        if np.linalg.norm(new_pos) < 10.2 * self.max_pos:
            self.cur_pos = new_pos

        state = self._get_state(self.cur_pos, self.goal_pos)
        reward, done = self._get_reward(state, action)
        self.time += 1
        terminated = done
        truncated = self.time >= self.max_step

        return state.copy(), reward, terminated, truncated, {"state": state}

    def _random_position_within_radius(self, radius):
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, radius)
        x, y = r * np.cos(angle), r * np.sin(angle)
        return np.array([x, y])

    def _generate_task_vectors(self, n_tasks):
        angle_increment = 2 * np.pi / n_tasks
        return [np.array([np.cos(i * angle_increment), np.sin(i * angle_increment)]) for i in range(n_tasks)]

    def _get_state(self, pos, goal):
        return np.concatenate([pos / self.max_pos, goal / self.max_pos])

    def _get_reward(self, state, action):
        pos, goal = state[:2] * self.max_pos, state[2:] * self.max_pos
        distance_to_goal = np.linalg.norm(pos - goal)
        reward = -distance_to_goal / self.max_pos
        done = distance_to_goal < self.goal_rad
        done = False if self.disable_goal else done
        if done:
            reward += 1.0
        return reward, done

    def close(self):
        pass