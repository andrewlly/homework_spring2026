import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import glob

def get_exp_label(flags):
    # Extract lambda value
    lam = flags.get('gae_lambda')
    if lam is None:
        return "No GAE"
    return f"GAE (λ={lam})"

def plot_lunar():
    experiments = []
    
    # Recursively find all flags.json files
    flag_files = glob.glob("**/*flags.json", recursive=True)
    
    print(f"Found {len(flag_files)} experiment directories...")

    for flag_path in flag_files:
        exp_dir = os.path.dirname(flag_path)
        log_path = os.path.join(exp_dir, 'log.csv')

        if os.path.exists(log_path):
            try:
                with open(flag_path, 'r') as f:
                    flags = json.load(f)
                
                # Filter for LunarLander only
                if flags.get('env_name') != 'LunarLander-v2':
                    continue

                label = get_exp_label(flags)
                df = pd.read_csv(log_path)
                
                experiments.append({
                    "label": label,
                    "data": df,
                    "lambda": flags.get('gae_lambda', -1) # for sorting
                })
            except Exception as e:
                print(f"Error processing {exp_dir}: {e}")

    # Sort experiments by lambda value for clean legend
    experiments.sort(key=lambda x: x["lambda"])

    # --- Plot: Eval Return ---
    plt.figure(figsize=(10, 6))
    
    for exp in experiments:
        if 'Eval_AverageReturn' in exp['data'].columns:
            plt.plot(exp['data']['Train_EnvstepsSoFar'], 
                     exp['data']['Eval_AverageReturn'], 
                     label=exp['label'], linewidth=2)
    
    # Add target threshold line
    plt.axhline(y=150, color='r', linestyle='--', alpha=0.5, label="Target (150)")

    plt.title("LunarLander-v2: GAE Lambda Comparison")
    plt.xlabel("Environment Steps")
    plt.ylabel("Average Eval Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("lunar_lander_lambda_comparison.png")
    plt.show()
    print("Plot saved to lunar_lander_lambda_comparison.png")

if __name__ == "__main__":
    plot_lunar()