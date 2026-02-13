import os
import json
import gymnasium
from stable_baselines3 import SAC, DDPG, DQN, PPO, TD3
from random_model import RandomAgent
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from env_utils.conditioned_envs import ConditionalStateWrapper
from env_utils.non_stationary_wrapper import NonStationaryEnv
# from env_utils.non_stationary_wrapper_new import NonStationaryEnv
from envs.lunar_wrapper import MultiTaskLunarLander
from envs.goal_reacher import GoalReacherEnv
from task_inference_utils.simple_inference import SimpleTaskInference
from task_inference_utils.sr_inference import SymbolicRegressionInference
from task_inference_utils.oracle_inference import OracleInference
from task_inference_utils.vae_inference import VAEInference
from general_utils import load_monitor_data, plot_moving_average_reward, fix_seed
import argparse

# Hyperparameters

learning_rate = 0.001
seed = 0
fix_seed(seed)

# Argument parser
parser = argparse.ArgumentParser(description="Expression Conditioned Reinforcement Learning")
parser.add_argument('--env', type=str, default='LunarLander',
                    choices=['HalfCheetah-v4', 'LunarLander', 'Pendulum-v1', 'Swimmer-v4', 'Reacher-v4', 'CartPole-v1', 'GoalReacher'],
                    help='Environment to use for training and evaluation.')
parser.add_argument('--algo', type=str, default='SAC',
                    choices=['SAC', 'DDPG', 'DQN', 'PPO', 'TD3', 'Random'],
                    help='Reinforcement learning algorithm to use.')
parser.add_argument('--inference', type=str, default='oracle',
                    choices=['simple', 'vae', 'sr', 'oracle'],
                    help='Task inference method to use.')
parser.add_argument('--nonstationary', type=lambda x: (str(x).lower() == 'true'), default=True,
                    help='Set to False to disable nonstationary environment modifications.')

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
    task_inference = VAEInference()
elif args.inference == 'sr':
    task_inference = SymbolicRegressionInference(context_size=context_size)
elif args.inference == 'oracle':
    task_inference = OracleInference(task_size=n_tasks)
    
# Create environment

if args.env == 'GoalReacher':
    env = GoalReacherEnv()
elif args.env == 'LunarLander':
    env = MultiTaskLunarLander(render_mode=False,
                               wind_force=-12.0,
                               engine_power_scale=1.2,
                               gravity=-8,
                               continuous=True)
else:
    gymnasium.make(args.env)

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
    
    if args.env == 'GoalReacher':
        eval_env = GoalReacherEnv()
    elif args.env == 'LunarLander':
        eval_env = MultiTaskLunarLander(render_mode=False,
                                   wind_force=-12.0,
                                   engine_power_scale=1.2,
                                   gravity=-8,
                                   continuous=True)
    else:
        eval_env = gymnasium.make(args.env)

    if args.nonstationary:
        eval_env = NonStationaryEnv(eval_env, max_ep_len, n_tasks, task_name, args.env)
        eval_env = ConditionalStateWrapper(eval_env, task_inference=task_inference, is_oracle=args.inference == 'oracle')

    eval_env = Monitor(eval_env, log_dir)

    # Callback for evaluation
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                log_path=log_dir, eval_freq=eval_freq,
                                deterministic=True, render=False)


    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)


# Load and visualize results
results_df = load_monitor_data(log_dir)
plot_moving_average_reward(results_df, title=title, label=args.algo, color='blue')
