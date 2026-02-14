import numpy as np
import random
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def load_monitor_data(log_dir):

    monitor_path = os.path.join(log_dir, 'monitor.csv')
    df = pd.read_csv(monitor_path, skiprows=1)
    df['r'] = pd.to_numeric(df['r'], errors='coerce')
    df['l'] = pd.to_numeric(df['l'], errors='coerce')
    df['t'] = pd.to_numeric(df['t'], errors='coerce')
    df = df.dropna()
    return df

def plot_moving_average_reward(df, window=10, title='Performance', label='Agent', color='blue'):

    df['moving_avg'] = df['r'].rolling(window=window).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['moving_avg'], label=label, color=color)
    plt.title(title)
    plt.xlabel('Episode Index')
    plt.ylabel('Moving Average Reward')
    plt.legend()
    plt.grid()
    plt.show()