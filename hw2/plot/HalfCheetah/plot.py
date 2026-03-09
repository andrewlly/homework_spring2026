import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

def get_exp_label(exp_name):
    if "cheetah_baseline_low_bgs" in exp_name:
        return "Baseline (Low BGS=2)"
    elif "cheetah_baseline" in exp_name:
        return "Baseline (Standard BGS=5)"
    elif "cheetah" in exp_name:
        return "No Baseline"
    else:
        return exp_name

def plot_cheetah():
    experiments = []
    
    # scan for directories containing 'cheetah'
    # Recursive search for log.csv files
    files = glob.glob("**/*log.csv", recursive=True)

    for log_path in files:
        if "HalfCheetah" not in log_path:
            continue
            
        folder_name = os.path.dirname(log_path)
        
        # Identify the experiment config from the folder name
        label = get_exp_label(folder_name)
        
        try:
            df = pd.read_csv(log_path)
            experiments.append({
                "label": label,
                "data": df
            })
        except Exception as e:
            print(f"Skipping {log_path}: {e}")

    # --- Plot 1: Eval Return ---
    plt.figure(figsize=(10, 6))
    for exp in experiments:
        if "Eval_AverageReturn" in exp["data"].columns:
            plt.plot(exp["data"]['Train_EnvstepsSoFar'], exp["data"]['Eval_AverageReturn'], label=exp["label"], linewidth=2)
    
    plt.title("HalfCheetah-v4: Policy Performance")
    plt.xlabel("Environment Steps")
    plt.ylabel("Average Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("cheetah_eval_return.png")
    plt.show()

    # --- Plot 2: Critic/Baseline Loss ---
    plt.figure(figsize=(10, 6))
    plotted_any = False
    for exp in experiments:
        # Check for 'Critic Loss' (Your log name) OR 'Baseline Loss' (Alternative name)
        if "Critic Loss" in exp["data"].columns:
            key = "Critic Loss"
        elif "Baseline Loss" in exp["data"].columns:
            key = "Baseline Loss"
        else:
            continue

        # Only plot if not all values are NaN
        if not exp["data"][key].isnull().all():
            plt.plot(exp["data"]['Train_EnvstepsSoFar'], exp["data"][key], label=exp["label"], linewidth=2)
            plotted_any = True

    if plotted_any:
        plt.title("HalfCheetah-v4: Baseline/Critic Loss")
        plt.xlabel("Environment Steps")
        plt.ylabel("Critic Loss (MSE)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("cheetah_baseline_loss.png")
        plt.show()
    else:
        print("No baseline/critic loss data found to plot.")

if __name__ == "__main__":
    plot_cheetah()