import os
from typing import Dict, List, Optional, Tuple

import gymnasium
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from envs.goal_reacher import GoalReacherEnv
from envs.lunar_wrapper import MultiTaskLunarLander


def _space_dim(space) -> int:
    if isinstance(space, gymnasium.spaces.Box):
        return int(np.prod(space.shape))
    if isinstance(space, gymnasium.spaces.Discrete):
        return int(space.n)
    raise ValueError(f"Unsupported space type for baseline model: {type(space)}")


def _flatten_obs(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def _action_to_vector(action, action_space) -> np.ndarray:
    if isinstance(action_space, gymnasium.spaces.Box):
        return np.asarray(action, dtype=np.float32).reshape(-1)
    if isinstance(action_space, gymnasium.spaces.Discrete):
        vec = np.zeros(action_space.n, dtype=np.float32)
        vec[int(action)] = 1.0
        return vec
    raise ValueError(f"Unsupported action space for baseline model: {type(action_space)}")


def make_base_env(env_name: str):
    if env_name == "GoalReacher":
        return GoalReacherEnv()
    if env_name == "LunarLander":
        return MultiTaskLunarLander(
            render_mode=False,
            wind_force=-12.0,
            engine_power_scale=1.2,
            gravity=-8,
            continuous=True,
        )
    return gymnasium.make(env_name)


class StationaryDynamicsMLP(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        input_dim = obs_dim + act_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.ReLU())
            input_dim = hidden
        layers.append(nn.Linear(input_dim, obs_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1))


class NNDynamicsBaseline:
    def __init__(
        self,
        model: StationaryDynamicsMLP,
        obs_dim: int,
        act_dim: int,
        input_mean: np.ndarray,
        input_std: np.ndarray,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        env_id: str,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.env_id = env_id
        self.device = device
        self.input_mean = input_mean.astype(np.float32)
        self.input_std = np.clip(input_std.astype(np.float32), 1e-6, None)
        self.target_mean = target_mean.astype(np.float32)
        self.target_std = np.clip(target_std.astype(np.float32), 1e-6, None)

    def predict_delta(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)

        if states.ndim == 1:
            states = states[None, :]
        if actions.ndim == 1:
            actions = actions[None, :]

        if states.shape[1] != self.obs_dim:
            raise ValueError(
                f"Baseline state dimension mismatch: got {states.shape[1]}, expected {self.obs_dim}"
            )
        if actions.shape[1] != self.act_dim:
            raise ValueError(
                f"Baseline action dimension mismatch: got {actions.shape[1]}, expected {self.act_dim}"
            )

        x = np.concatenate([states, actions], axis=-1)
        x = (x - self.input_mean) / self.input_std
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pred = self.model.net(x_t).cpu().numpy()
        return pred * self.target_std + self.target_mean

    def save(self, path: str):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "hidden_dims": _extract_hidden_dims(self.model),
                "env_id": self.env_id,
                "input_mean": self.input_mean,
                "input_std": self.input_std,
                "target_mean": self.target_mean,
                "target_std": self.target_std,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        checkpoint = torch.load(path, map_location=device)
        model = StationaryDynamicsMLP(
            obs_dim=checkpoint["obs_dim"],
            act_dim=checkpoint["act_dim"],
            hidden_dims=checkpoint["hidden_dims"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        return cls(
            model=model,
            obs_dim=checkpoint["obs_dim"],
            act_dim=checkpoint["act_dim"],
            input_mean=np.asarray(checkpoint["input_mean"], dtype=np.float32),
            input_std=np.asarray(checkpoint["input_std"], dtype=np.float32),
            target_mean=np.asarray(checkpoint["target_mean"], dtype=np.float32),
            target_std=np.asarray(checkpoint["target_std"], dtype=np.float32),
            env_id=checkpoint.get("env_id", "unknown"),
            device=device,
        )


def _extract_hidden_dims(model: StationaryDynamicsMLP) -> List[int]:
    hidden_dims = []
    for layer in model.net:
        if isinstance(layer, nn.Linear):
            hidden_dims.append(layer.out_features)
    if hidden_dims:
        hidden_dims.pop()  # final output layer
    return hidden_dims


def collect_stationary_dataset(
    env_name: str,
    steps: int,
    seed: int,
    policy: str = "random",
) -> Dict[str, np.ndarray]:
    if policy != "random":
        raise ValueError(f"Unsupported policy for stationary data collection: {policy}")

    env = make_base_env(env_name)
    obs, _ = env.reset(seed=seed)
    obs = _flatten_obs(obs)

    obs_dim = _space_dim(env.observation_space)
    act_dim = _space_dim(env.action_space)

    states = np.zeros((steps, obs_dim), dtype=np.float32)
    actions = np.zeros((steps, act_dim), dtype=np.float32)
    deltas = np.zeros((steps, obs_dim), dtype=np.float32)

    for i in range(steps):
        action = env.action_space.sample()
        action_vec = _action_to_vector(action, env.action_space)

        next_obs, _, terminated, truncated, _ = env.step(action)
        next_obs = _flatten_obs(next_obs)

        states[i] = obs
        actions[i] = action_vec
        deltas[i] = next_obs - obs

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
            obs = _flatten_obs(obs)

    env.close()
    return {
        "states": states,
        "actions": actions,
        "deltas": deltas,
        "obs_dim": np.array([obs_dim], dtype=np.int32),
        "act_dim": np.array([act_dim], dtype=np.int32),
    }


def train_baseline_model(
    dataset: Dict[str, np.ndarray],
    config: Dict,
    env_id: str,
    device: str = "cpu",
) -> Tuple[NNDynamicsBaseline, Dict[str, float]]:
    states = dataset["states"].astype(np.float32)
    actions = dataset["actions"].astype(np.float32)
    deltas = dataset["deltas"].astype(np.float32)

    x = np.concatenate([states, actions], axis=-1)
    input_mean = x.mean(axis=0)
    input_std = np.clip(x.std(axis=0), 1e-6, None)
    target_mean = deltas.mean(axis=0)
    target_std = np.clip(deltas.std(axis=0), 1e-6, None)

    x_norm = (x - input_mean) / input_std
    y_norm = (deltas - target_mean) / target_std

    hidden_dims = config.get("hidden_dims", [128, 128])
    batch_size = int(config.get("batch_size", 512))
    epochs = int(config.get("epochs", 20))
    lr = float(config.get("lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 0.0))

    obs_dim = states.shape[1]
    act_dim = actions.shape[1]
    model = StationaryDynamicsMLP(obs_dim=obs_dim, act_dim=act_dim, hidden_dims=hidden_dims).to(device)

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_norm.astype(np.float32)),
            torch.from_numpy(y_norm.astype(np.float32)),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    model.train()
    last_loss = 0.0
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model.net(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())

    baseline = NNDynamicsBaseline(
        model=model,
        obs_dim=obs_dim,
        act_dim=act_dim,
        input_mean=input_mean,
        input_std=input_std,
        target_mean=target_mean,
        target_std=target_std,
        env_id=env_id,
        device=device,
    )
    baseline.model.eval()
    return baseline, {"final_loss": last_loss}
