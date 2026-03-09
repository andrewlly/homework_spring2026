import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import glob

def get_exp_label(flags):
    # Determine label based on flags
    rtg = flags.get('use_reward_to_go', False)
    na = flags.get('normalize_advantages', False)
    
    if rtg and na:
        return "Reward-to-Go + Advantage Norm"
    elif rtg:
        return "Reward-to-Go"
    elif na:
        return "Advantage Norm"
    else:
        return "Vanilla"

def plot_results():
    # Lists to store data
    small_batch_data = [] # batch_size = 1000
    large_batch_data = [] # batch_size = 4000

    # Find all folders containing flags.json
    exp_dirs = [x[0] for x in os.walk('.')]
    
    print(f"Scanning {len(exp_dirs)} directories...")

    for exp_dir in exp_dirs:
        flags_path = os.path.join(exp_dir, 'flags.json')
        log_path = os.path.join(exp_dir, 'log.csv')

        if os.path.exists(flags_path) and os.path.exists(log_path):
            with open(flags_path, 'r') as f:
                flags = json.load(f)
            
            # Check if this is a CartPole experiment
            if flags.get('env_name') != 'CartPole-v0':
                continue

            batch_size = flags.get('batch_size')
            label = get_exp_label(flags)
            
            try:
                df = pd.read_csv(log_path)
                data_tuple = (label, df)
                
                if batch_size == 1000:
                    small_batch_data.append(data_tuple)
                elif batch_size == 4000:
                    large_batch_data.append(data_tuple)
            except Exception as e:
                print(f"Error reading log in {exp_dir}: {e}")

    # Helper function to plot a set of data
    def create_plot(data_list, title, filename):
        plt.figure(figsize=(10, 6))
        
        # Sort by label to ensure consistent coloring
        data_list.sort(key=lambda x: x[0])
        
        for label, df in data_list:
            plt.plot(df['Train_EnvstepsSoFar'], df['Train_AverageReturn'], label=label, linewidth=2)
        
        plt.title(title)
        plt.xlabel("Environment Steps")
        plt.ylabel("Average Return")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(filename)
        plt.show()
        print(f"Saved plot to {filename}")

    if small_batch_data:
        create_plot(small_batch_data, "CartPole Small Batch (b=1000)", "cartpole_sb_learning_curve.png")
    else:
        print("No small batch data found!")

    if large_batch_data:
        create_plot(large_batch_data, "CartPole Large Batch (b=4000)", "cartpole_lb_learning_curve.png")
    else:
        print("No large batch data found!")

if __name__ == "__main__":
    plot_results()