import os
import json
import gymnasium
import argparse
from stable_baselines3 import SAC, DDPG, DQN, PPO, TD3
from random_model import RandomAgent
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from env_utils.conditioned_envs import ConditionalStateWrapper
from env_utils.non_stationary_wrapper import NonStationaryEnv
from envs.lunar_wrapper import MultiTaskLunarLander
from envs.goal_reacher import GoalReacherEnv
from task_inference_utils.simple_inference import SimpleTaskInference
from task_inference_utils.sr_inference import SymbolicRegressionInference
from task_inference_utils.oracle_inference import OracleInference
from task_inference_utils.nn_dynamics_baseline import (
    collect_stationary_dataset,
    train_baseline_model,
    NNDynamicsBaseline,
)
from general_utils import load_monitor_data, plot_moving_average_reward, fix_seed

# Hyperparameters
learning_rate = 0.001
seed = 0
fix_seed(seed)


def str_to_bool(x):
    return str(x).lower() == "true"


def parse_hidden_dims(value):
    if value is None:
        return None
    return [int(v.strip()) for v in str(value).split(",") if v.strip()]


def make_env(env_name):
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


def resolve_baseline(args, config):
    baseline_cfg = config.get("baseline", {})
    hidden_dims = parse_hidden_dims(args.baseline_hidden_dims)
    if hidden_dims is None:
        hidden_dims = baseline_cfg.get("hidden_dims", [128, 128])

    baseline_config = {
        "hidden_dims": hidden_dims,
        "batch_size": int(baseline_cfg.get("batch_size", 512)),
        "epochs": int(baseline_cfg.get("epochs", 20)),
        "lr": float(baseline_cfg.get("lr", 1e-3)),
        "weight_decay": float(baseline_cfg.get("weight_decay", 0.0)),
    }

    default_ckpt = baseline_cfg.get(
        "save_path", os.path.join("baselines", f"{args.env}_stationary.pt")
    )
    baseline_ckpt = args.baseline_ckpt or default_ckpt
    data_steps = args.baseline_data_steps or int(baseline_cfg.get("data_steps", 50000))

    if os.path.exists(baseline_ckpt):
        print(f"[baseline] loading checkpoint: {baseline_ckpt}")
        return NNDynamicsBaseline.load(baseline_ckpt), baseline_ckpt

    if not args.baseline_train_if_missing:
        raise FileNotFoundError(
            f"Baseline checkpoint not found at '{baseline_ckpt}'. "
            "Set --baseline_train_if_missing true or provide --baseline_ckpt."
        )

    print(f"[baseline] collecting stationary dataset for {args.env} with {data_steps} steps")
    dataset = collect_stationary_dataset(env_name=args.env, steps=data_steps, seed=seed)
    baseline_model, train_info = train_baseline_model(
        dataset=dataset, config=baseline_config, env_id=args.env
    )
    baseline_model.save(baseline_ckpt)
    print(
        f"[baseline] trained and saved checkpoint: {baseline_ckpt} "
        f"(final_loss={train_info['final_loss']:.6f})"
    )
    return baseline_model, baseline_ckpt


# Argument parser
parser = argparse.ArgumentParser(description="Expression Conditioned Reinforcement Learning")
parser.add_argument('--env', type=str, default='LunarLander',
                    choices=['HalfCheetah-v4', 'LunarLander', 'Pendulum-v1', 'Swimmer-v4', 'Reacher-v4', 'CartPole-v1', 'GoalReacher'],
                    help='Environment to use for training and evaluation.')
parser.add_argument('--algo', type=str, default='SAC',
                    choices=['SAC', 'DDPG', 'DQN', 'PPO', 'TD3', 'Random'],
                    help='Reinforcement learning algorithm to use.')
parser.add_argument('--inference', type=str, default='oracle',
                    choices=['simple', 'vae', 'sr', 'sr_residual_nn', 'oracle'],
                    help='Task inference method to use.')
parser.add_argument('--nonstationary', type=str_to_bool, default=True,
                    help='Set to False to disable nonstationary environment modifications.')
parser.add_argument('--baseline_ckpt', type=str, default=None,
                    help='Path to stationary dynamics NN checkpoint for residual symbolic inference.')
parser.add_argument('--baseline_train_if_missing', type=str_to_bool, default=True,
                    help='Whether to train stationary baseline if checkpoint is missing.')
parser.add_argument('--baseline_data_steps', type=int, default=None,
                    help='Number of steps for stationary baseline dataset collection.')
parser.add_argument('--baseline_hidden_dims', type=str, default=None,
                    help='Comma-separated hidden layer sizes for baseline MLP, e.g. "128,128".')

args = parser.parse_args()

config_path = os.path.join('configs', f"{args.env}.json")
with open(config_path, 'r') as f:
    config = json.load(f)

# Extract parameters from the config
max_ep_len = config['max_ep_len']
n_tasks = config['n_tasks']
task_name = config['task_name']
total_timesteps = config['total_timesteps']
eval_freq = config['eval_freq']
context_size = config['context_size']

log_dir = f'./logs/{args.algo.lower()}_{args.env.lower()}/{args.inference.lower()}_{task_name.lower()}_{seed}/'
os.makedirs(log_dir, exist_ok=True)

title = f'{args.algo} Performance on {args.env}'

# Select task inference method
if args.inference == 'simple':
    task_inference = SimpleTaskInference(context_size)
elif args.inference == 'vae':
    raise NotImplementedError("VAE inference is not available in this repository.")
elif args.inference == 'sr':
    task_inference = SymbolicRegressionInference(context_size=context_size)
elif args.inference == 'sr_residual_nn':
    baseline_model, baseline_ckpt = resolve_baseline(args, config)
    task_inference = SymbolicRegressionInference(
        context_size=context_size,
        baseline_model=baseline_model,
        residual_mode='nn_delta',
        residual=True,
    )
    print(f"[inference] using residual symbolic regression with baseline: {baseline_ckpt}")
elif args.inference == 'oracle':
    task_inference = OracleInference(task_size=n_tasks)

# Create environment
env = make_env(args.env)

if args.nonstationary:
    env = NonStationaryEnv(env, max_ep_len, n_tasks, task_name, args.env)
    env = ConditionalStateWrapper(env, task_inference=task_inference, is_oracle=args.inference == 'oracle')

# env = Monitor(env, log_dir)

# Select RL algorithm
if args.algo == 'SAC':
    model = SAC('MlpPolicy', env, verbose=1, learning_rate=learning_rate)
elif args.algo == 'DDPG':
    model = DDPG('MlpPolicy', env, verbose=1, learning_rate=learning_rate)
elif args.algo == 'DQN':
    model = DQN('MlpPolicy', env, verbose=1, learning_rate=learning_rate)
elif args.algo == 'PPO':
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=learning_rate)
elif args.algo == 'TD3':
    model = TD3('MlpPolicy', env, verbose=1, learning_rate=learning_rate)
elif args.algo == 'Random':
    model = RandomAgent(env, total_timesteps, output_file=log_dir)

if args.algo == 'Random':
    model.run()
else:
    eval_env = make_env(args.env)

    if args.nonstationary:
        eval_env = NonStationaryEnv(eval_env, max_ep_len, n_tasks, task_name, args.env)
        eval_env = ConditionalStateWrapper(eval_env, task_inference=task_inference, is_oracle=args.inference == 'oracle')

    eval_env = Monitor(eval_env, log_dir)

    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                log_path=log_dir, eval_freq=eval_freq,
                                deterministic=True, render=False)

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

# Load and visualize results
results_df = load_monitor_data(log_dir)
plot_moving_average_reward(results_df, title=title, label=args.algo, color='blue')
