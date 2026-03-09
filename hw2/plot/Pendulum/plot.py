import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

def plot_pendulum():
    experiments = []
    # Find all log.csv files recursively
    files = glob.glob("**/*log.csv", recursive=True)

    for log_path in files:
        if "InvertedPendulum" not in log_path:
            continue
            
        folder_name = os.path.dirname(log_path)
        
        # Determine label based on folder name
        if "pendulum_tuned" in folder_name:
            label = "Tuned (b=1000, RTG, NA, Baseline)"
        elif "pendulum_default" in folder_name or "pendulum_sd" in folder_name:
            label = "Default (b=5000)"
        else:
            continue
        
        try:
            df = pd.read_csv(log_path)
            experiments.append({"label": label, "data": df})
        except Exception as e:
            print(f"Skipping {log_path}: {e}")

    plt.figure(figsize=(10, 6))
    
    for exp in experiments:
        # Plot Train Return vs Environment Steps
        plt.plot(exp["data"]['Train_EnvstepsSoFar'], 
                 exp["data"]['Train_AverageReturn'], 
                 label=exp["label"], linewidth=2)
    
    # Add a horizontal line for the target
    plt.axhline(y=1000, color='r', linestyle='--', label="Target Return (1000)")
    
    plt.title("InvertedPendulum-v4: Sample Efficiency Comparison")
    plt.xlabel("Environment Steps")
    plt.ylabel("Average Return")
    plt.xlim(0, 100000)  # Focus on the first 100k steps
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("pendulum_comparison.png")
    plt.show()

if __name__ == "__main__":
    plot_pendulum()