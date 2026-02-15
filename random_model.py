import gym
import csv
import time
import json

class RandomAgent:
    def __init__(self, env, max_timesteps, output_file):
        """
        Initializes the agent.

        Args:
            env: An OpenAI Gym environment.
            max_timesteps: The maximum number of timesteps for which to run the agent.
            output_file: The filename for saving the results.
        """
        self.env = env
        self.max_timesteps = max_timesteps
        self.output_file = f'{output_file}/monitor.csv'

    def run(self):
        t_start = time.time()
        # Determine environment id from the environment's spec if available
        env_id = getattr(self.env, 'spec', None)
        env_id = env_id.id if env_id is not None else type(self.env).__name__

        results = []  # To store results for each episode

        total_steps = 0
        while total_steps < self.max_timesteps:
            self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            done, truncated = False, False

            while not truncated and not done:
                # Sample a random action
                action = self.env.action_space.sample()
                _, reward, done, truncated, _ = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                total_steps += 1
            

                # Stop the run if we exceed max_timesteps
                if total_steps >= self.max_timesteps:
                    break

            # Record the result with elapsed time since t_start
            elapsed_time = time.time() - t_start
            results.append((episode_reward, episode_length, elapsed_time))

        # Write results to CSV file with the specified format
        with open(self.output_file, mode='w', newline='') as csvfile:
            # Write header with JSON comment line
            header_info = {"t_start": t_start, "env_id": env_id}
            csvfile.write(f"#{json.dumps(header_info)}\n")
            writer = csv.writer(csvfile)
            writer.writerow(['r', 'l', 't'])
            for row in results:
                writer.writerow(row)

        print(f"Results saved to {self.output_file}")

    

        
        

