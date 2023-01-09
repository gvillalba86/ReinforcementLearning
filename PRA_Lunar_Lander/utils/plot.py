import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from itertools import cycle


def plot_results_hist(n_steps, total_rewards):
    '''Plot the results of playing n games in histogram'''
    fig, ax = plt.subplots(1,2,figsize=(10,4), sharey=True)
    sns.histplot(x=n_steps, ax=ax[0], bins=20, kde=True)
    ax[0].set_title(f'Steps per episode (mean={n_steps.mean():.2f})')
    ax[0].set_xlabel('Steps')
    ax[0].set_ylabel('Frequency')
    sns.histplot(x=total_rewards, ax=ax[1], bins=20, kde=True)
    ax[1].set_title(f'Total reward per episode (mean={total_rewards.mean():.2f})')
    ax[1].set_xlabel('Total reward')
    ax[1].set_ylabel('Frequency')
    fig.tight_layout()
    plt.show()
    

def plot_test_episodes(total_rewards, reward_threshold):
    '''Plot the results (rewards) of playing n games'''
    fig, ax = plt.subplots(figsize=(8,4))
    episodes = range(1,len(total_rewards)+1)
    sns.lineplot(x=episodes, y=total_rewards, label='Reward', ax=ax)
    sns.lineplot(x=episodes, y=reward_threshold, label='Reward threshold', ax=ax)
    ax.set_title(f'Reward vs. Episodes (Mean reward: {total_rewards.mean():.2f})')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Reward')
    ax.set_xticks(episodes)
    plt.show()
    

def plot_n_training(training_res_array, reward_threshold, name):
    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True, gridspec_kw={'height_ratios':[1,10]})
    n_episodes = []
    for training_res in training_res_array:
        episodes = range(1,len(training_res['mean_training_rewards'])+1)
        sns.lineplot(x=episodes, y=reward_threshold, color='grey', ax=ax[1])
        sns.lineplot(x=episodes, y=training_res['mean_training_rewards'], color='#348ABD', alpha=0.8, ax=ax[1])
        n_episodes.append(len(training_res['mean_training_rewards']))
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Reward')
    ax[1].set_ylim(-200, 220)
    sns.boxplot(x=n_episodes, ax=ax[0])
    ax[0].set_xlabel('')
    ax[0].set_title(f'{name} ({np.mean(n_episodes):.2f} +/- {np.std(n_episodes):.2f})')
    ax[0].set_ylabel('')
    plt.tight_layout()
    plt.show()
    return fig
    

def plot_training_results(training_stats, reward_threshold):
    '''Plot the training results of an agent'''
    
    rewards = training_stats.get('training_rewards', None)
    losses = training_stats.get('training_losses', None)
    epsilon = training_stats.get('training_epsilons', None)
    mean_rewards = training_stats.get('mean_training_rewards', None)
    
    n = len([met for met in [rewards, losses, epsilon] if met is not None])
    fig, ax = plt.subplots(n, 1, figsize=(10, n*3), sharex=True)
    # Plot rewards
    episodes = range(len(rewards))
    sns.lineplot(x=episodes, y=rewards, label='Reward', ax=ax[0], lw=0.5, alpha=0.5)
    sns.lineplot(x=episodes, y=mean_rewards, label='Mean reward', ax=ax[0], lw=1)
    sns.lineplot(x=episodes, y=reward_threshold, label='Reward threshold', color='grey', ax=ax[0])
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    ax[0].set_ylim(-300, reward_threshold+100)
    # Plot losses
    sns.lineplot(x=episodes, y=losses, ax=ax[1], lw=1)
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Loss')
    ax[1].set_ylim(0, 100)
    plt.tight_layout()
    # Plot epsilon
    if epsilon is not None:
        sns.lineplot(x=episodes, y=epsilon, ax=ax[2])
        ax[2].set_xlabel('Episodes')
        ax[2].set_ylabel('Epsilon value')
    plt.tight_layout()
    plt.show()
    
    
def compare_training_hp(results, threshold, param_name:str):
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios':[1,4]})
    df_len = pd.DataFrame()
    colors = cycle(plt.rcParams['axes.prop_cycle'])
    for param_value, res_list in results.items():
        n_episodes = []
        color = next(colors)['color']
        for res in res_list:
            line = ax[1].plot(res['mean_training_rewards'], color=color, alpha=0.5)
            n_episodes.append(len(res['mean_training_rewards']))
        plt.setp(line, label=f'{param_name} = {param_value}')
        df_len[param_value] = n_episodes
    df_len = df_len.melt()
    ax[1].legend(title=param_name, loc='lower right')
    ax[1].axhline(threshold, color='grey', linestyle='--', label='Threshold')
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Reward')
    ax[1].set_ylim(-200, 250)
    ax[0].set_title(f'Mean training reward changing {param_name}')
    sns.pointplot(x='value', y='variable', hue='variable', data=df_len, ax=ax[0])
    ax[0].legend().set_visible(False)
    ax[0].set_xlabel('')
    ax[0].set_ylabel(param_name)
    plt.tight_layout()
    plt.show()
    return fig