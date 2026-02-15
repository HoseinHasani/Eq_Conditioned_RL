# Expression Conditioned Reinforcement Learning

This project implements reinforcement learning (RL) with expression-conditioned environments, providing flexible options for environment selection, RL algorithms, and task inference methods. The framework is designed to accommodate experimentation with various RL techniques and task inference strategies.

## Features
- Support for multiple RL algorithms: SAC, DDPG, DQN, PPO1, and TD3.
- Task inference via:
  - SimpleTaskInference
  - SymbolicRegressionInference
  - VAEInference 
- Configurable environments: HalfCheetah-v4, Pendulum-v1, Swimmer-v4, Reacher-v4, CartPole-v1.
- Modular codebase for customization and extension.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/expression-conditioned-rl.git
   cd expression-conditioned-rl
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## File Structure
```
expression-conditioned-rl/
├── main.py                # Entry point for training and evaluation
├── visualization_utils.py # Utilities for loading and visualizing results
├── task_inference_utils/  # Contains task inference methods
│   ├── base_inference.py       # Base class for task inference
│   ├── simple_inference.py     # Simple task inference implementation
│   ├── sr_inference.py         # Symbolic regression-based inference (SINDy)
│   ├── vae_inference.py        # VAE-based task inference
│   ├── oracle_inference.py     # Oracle task inference
├── env_utils/            # Environment wrappers and utilities
│   ├── conditioned_envs.py
├── envs/                 # Custom environments
│   ├── goal_reacher.py   # GoalReacherEnv implementation
├── tests/                # Evaluation scripts
│   ├── gather_condition_data.py  # Collects data for task inference evaluation
│   ├── classifier_on_conditions.py  # Trains an MLP classifier to assess accuracy
├── configs/              # Configuration files for different environments
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

## Usage
The main training script supports various configurations via command-line arguments. You can specify the environment, RL algorithm, and task inference method.

### Example Commands
#### Train with default settings (SAC and SimpleTaskInference):
```bash
python main.py --env HalfCheetah-v4
```

#### Train with PPO1 and SymbolicRegressionInference:
```bash
python main.py --env Pendulum-v1 --algo PPO1 --task_inference sr
```

#### Train with TD3 and VAEInference:
```bash
python main.py --env Swimmer-v4 --algo TD3 --task_inference vae
```

### Arguments
- `--env`: Environment to train on. Options: `HalfCheetah-v4`, `Pendulum-v1`, `Swimmer-v4`, `Reacher-v4`, `CartPole-v1`.
- `--algo`: RL algorithm. Options: `SAC` (default), `DDPG`, `DQN`, `PPO1`, `TD3`.
- `--task_inference`: Task inference method. Options: `simple` (default), `sr`, `vae`, `oracle`.

### Logs and Outputs
Logs and models are saved to directories named based on the selected algorithm and environment (e.g., `logs/SAC_HalfCheetah-v4/`). Visualization of results is automatically generated as plots of moving average rewards.

## Extending the Framework
### Adding a New Task Inference Method
1. Create a new file in the `task_inference_utils/` directory (e.g., `new_inference.py`).
2. Implement a class that inherits from `BaseTaskInference` and overrides necessary methods.
3. Update `main.py` to include the new inference method in the argument parsing logic.

## Citation
If you use this project in your research, consider citing our ECRL paper.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

