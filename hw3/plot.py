import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_csv(csv_path, x_col, y_col, title, save_name):
    if not os.path.exists(csv_path):
        print(f"Could not find {csv_path}")
        return
    df = pd.read_csv(csv_path)
    clean_df = df.dropna(subset=[y_col])
    
    plt.figure(figsize=(8, 5))
    plt.plot(clean_df[x_col], clean_df[y_col], label=y_col)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_name)
    plt.close()

def plot_multiple(csv_paths, labels, x_col, y_col, title, save_name):
    plt.figure(figsize=(8, 5))
    for path, label in zip(csv_paths, labels):
        if os.path.exists(path):
            df = pd.read_csv(path)
            clean_df = df.dropna(subset=[y_col])
            plt.plot(clean_df[x_col], clean_df[y_col], label=label)
            
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_name)
    plt.close()

# 2.4 CartPole
plot_csv('exp/CartPole-v1_dqn_sd1_20260309_031242/log.csv', 'step', 'Eval_AverageReturn', 'CartPole-v1 Eval Average Return', 'cartpole.png')

# 2.5 LunarLander
plot_csv('exp/LunarLander-v2_dqn_sd1_20260309_032530/log.csv', 'step', 'Eval_AverageReturn', 'LunarLander-v2 Eval Average Return', 'lunarlander.png')

# 2.5 MsPacman
pacman_path = 'exp/MsPacman_dqn_sd1_20260309_104152/log.csv'
if os.path.exists(pacman_path):
    df = pd.read_csv(pacman_path)
    plt.figure(figsize=(8, 5))
    
    train_df = df.dropna(subset=['Train_EpisodeReturn'])
    eval_df = df.dropna(subset=['Eval_AverageReturn'])
    
    plt.plot(train_df['step'], train_df['Train_EpisodeReturn'], label='Train Return', alpha=0.7)
    plt.plot(eval_df['step'], eval_df['Eval_AverageReturn'], label='Eval Average Return', linewidth=2)
    
    plt.xlabel('step')
    plt.ylabel('Return')
    plt.title('MsPacman-v0 Train vs Eval Average Return')
    plt.legend()
    plt.grid(True)
    plt.savefig('mspacman.png')
    plt.close()

# 2.6 Hyperparameters (LunarLander)
plot_multiple(
    [
        'exp/LunarLander-v2_dqn_sd1_20260309_032530/log.csv', 
        'exp/LunarLander-v2_dqn_small_sd1_20260310_165907/log.csv', 
        'exp/LunarLander-v2_dpn_deep_sd1_20260310_165920/log.csv', 
        'exp/LunarLander-v2_dqn_wide_sd1_20260310_170011/log.csv'
    ],
    ['Base (256x2)', 'Small (64x2)', 'Deep (256x4)', 'Wide (512x2)'],
    'step', 'Eval_AverageReturn', 'LunarLander Hyperparameters (Network Size)', 'hyperparameters.png'
)

# 3.4 SAC HalfCheetah
plot_csv('exp/HalfCheetah-v4_sac_sd1_20260309_050753/log.csv', 'step', 'Eval_AverageReturn', 'HalfCheetah-v4 SAC Eval Average Return', 'halfcheetah.png')

# 3.5 Autotune HalfCheetah
plot_multiple(
    ['exp/HalfCheetah-v4_sac_sd1_20260309_050753/log.csv', 'exp/HalfCheetah-v4_sac_autotune_sd1_20260309_050936/log.csv'],
    ['Fixed Temp', 'Auto-tuned Temp'],
    'step', 'Eval_AverageReturn', 'HalfCheetah-v4 Fixed vs Autotune', 'halfcheetah_compare.png'
)
plot_csv('exp/HalfCheetah-v4_sac_autotune_sd1_20260309_050936/log.csv', 'step', 'temperature', 'HalfCheetah-v4 Alpha (Temperature)', 'halfcheetah_alpha.png')

plot_multiple(
    ['exp/Hopper-v4_sac_singleq_sd1_20260309_124250/log.csv', 'exp/Hopper-v4_sac_clipq_sd1_20260309_130918/log.csv'],
    ['Single-Q', 'Clipped Double-Q'],
    'step', 'Eval_AverageReturn', 'Hopper-v4 Return Comparison', 'hopper_returns.png'
)
plot_multiple(
    ['exp/Hopper-v4_sac_singleq_sd1_20260309_124250/log.csv', 'exp/Hopper-v4_sac_clipq_sd1_20260309_130918/log.csv'],
    ['Single-Q', 'Clipped Double-Q'],
    'step', 'q_values', 'Hopper-v4 Q-Value Comparison', 'hopper_qvalues.png'
)