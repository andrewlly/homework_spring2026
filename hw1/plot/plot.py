import pandas as pd
import matplotlib.pyplot as plt

def plot_combined_loss(mse_log_path, flow_log_path):
    """
    Plots training loss for MSE and Flow policies on the same graph.
    """
    plt.figure(figsize=(10, 6))

    try:
        df_mse = pd.read_csv(mse_log_path)
        df_mse = df_mse.dropna(subset=['train/loss', 'step'])
        plt.plot(df_mse['step'], df_mse['train/loss'], 
                 label='MSE loss', color='tab:blue', alpha=0.7)
    except FileNotFoundError:
        print(f"File not found: {mse_log_path}")

    try:
        df_flow = pd.read_csv(flow_log_path)
        df_flow = df_flow.dropna(subset=['train/loss', 'step'])
        plt.plot(df_flow['step'], df_flow['train/loss'], 
                 label='Flow matching loss', color='tab:orange', alpha=0.7)
    except FileNotFoundError:
        print(f"File not found: {flow_log_path}")
        
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('Training loss: MSE vs Flow matching')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('combined_loss.png')
    print("Saved combined_loss.png")
    plt.close()

def plot_rewards(reward_csv_path):
    """
    Plots rewards from the WandB export CSV.
    Automatically detects MSE and Flow columns.
    """
    try:
        df = pd.read_csv(reward_csv_path)

        step_col = next((c for c in df.columns if 'step' in c.lower()), None)
        if not step_col:
            print("Could not find a 'Step' column in reward.csv")
            return

        reward_cols = [c for c in df.columns if 'eval/mean_reward' in c 
                       and 'MIN' not in c and 'MAX' not in c]

        plt.figure(figsize=(10, 6))
        
        for col in reward_cols:
            if 'mse' in col.lower():
                label = 'MSE'
                color = 'tab:blue'
            elif 'flow' in col.lower():
                label = 'Flow matching'
                color = 'tab:orange'
            else:
                label = col
                color = None 
            
            plt.plot(df[step_col], df[col], label=label, color=color, marker='o', markersize=4)

        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.title('Reward: MSE vs Flow matching')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('combined_reward.png')
        print("Saved combined_reward.png")
        plt.close()
        
    except Exception as e:
        print(f"Error plotting rewards: {e}")

if __name__ == "__main__":
    MSE_LOG_FILE = 'mse_log.csv'   
    FLOW_LOG_FILE = 'flow_log.csv'
    REWARD_FILE = 'reward.csv'

    plot_combined_loss(MSE_LOG_FILE, FLOW_LOG_FILE)
    plot_rewards(REWARD_FILE)