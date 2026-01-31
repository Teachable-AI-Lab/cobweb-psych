from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def visualize_fan_effect_results():
    """
    Visualizes the Fan Effect (Reder & Ross) experiment.
    Plots Log Probability (Model Strength) and Error Rate (1-Accuracy) on twin axes.
    """
    root_dir = Path(__file__).resolve().parent
    csv_path = root_dir / "exp_fan_effect_discrete.csv"
    
    if not csv_path.exists():
        print(f"Error: CSV not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Filter for Targets only for the main Fan Effect plot
    df_targets = df[df["type"] == "Target"].copy()
    
    # Calculate Error (1 - P)
    df_targets["error_mem"] = 1.0 - df_targets["prob_memory"]
    df_targets["error_cat"] = 1.0 - df_targets["prob_category"]
    
    # Aggregate
    summary = df_targets.groupby("fan_size").agg(
        ll_mem_mean=("log_like_memory", "mean"),
        ll_mem_sem=("log_like_memory", lambda x: np.std(x) / np.sqrt(len(x))),
        err_mem_mean=("error_mem", "mean"),
        err_mem_sem=("error_mem", lambda x: np.std(x) / np.sqrt(len(x))),
        
        ll_cat_mean=("log_like_category", "mean"),
        ll_cat_sem=("log_like_category", lambda x: np.std(x) / np.sqrt(len(x))),
        err_cat_mean=("error_cat", "mean"),
        err_cat_sem=("error_cat", lambda x: np.std(x) / np.sqrt(len(x)))
    ).reset_index()
    
    sns.set_theme(style="white", context="talk")
    
    # Create Figure: 2 Subplots (Memory, Category)
    fig, (ax1_mem, ax1_cat) = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(wspace=0.4)
    
    # --- Plot 1: MEMORY ---
    # Left Axis (Ax1): Log Likelihood (Strength)
    # Right Axis (Ax2): Error Rate
    
    fan_sizes = summary["fan_size"]
    
    # Memory: Log Prob
    ln1 = ax1_mem.errorbar(fan_sizes, summary["ll_mem_mean"], yerr=summary["ll_mem_sem"],
                             color='blue', marker='o', label='Log Prob (Strength)')
    ax1_mem.set_xlabel("Fan Size")
    ax1_mem.set_ylabel("Log Probability", color='blue')
    ax1_mem.tick_params(axis='y', labelcolor='blue')
    ax1_mem.set_title("Memory Task\n(Standard Fan Effect)")
    ax1_mem.set_xticks([1, 2, 3])
    ax1_mem.grid(True, linestyle= ':')
    
    # Memory: Error
    ax2_mem = ax1_mem.twinx()
    ln2 = ax2_mem.errorbar(fan_sizes, summary["err_mem_mean"], yerr=summary["err_mem_sem"],
                             color='red', marker='s', linestyle='--', label='Error Rate')
    ax2_mem.set_ylabel("Error Rate (1-P)", color='red')
    ax2_mem.tick_params(axis='y', labelcolor='red')
    ax2_mem.grid(False) # avoid clutter
    
    # Force alignment or just let them scale?
    # Usually twin axes scale independently.
    
    # --- Plot 2: CATEGORY ---
    
    # Category: Log Prob
    ax1_cat.errorbar(fan_sizes, summary["ll_cat_mean"], yerr=summary["ll_cat_sem"],
                     color='blue', marker='o', label='Log Prob (Strength)')
    ax1_cat.set_xlabel("Fan Size")
    ax1_cat.set_ylabel("Log Probability", color='blue')
    ax1_cat.tick_params(axis='y', labelcolor='blue')
    ax1_cat.set_title("Category Task\n(Reverse Fan Effect)")
    ax1_cat.set_xticks([1, 2, 3])
    ax1_cat.grid(True, linestyle= ':')
    
    # Category: Error
    ax2_cat = ax1_cat.twinx()
    ax2_cat.errorbar(fan_sizes, summary["err_cat_mean"], yerr=summary["err_cat_sem"],
                     color='red', marker='s', linestyle='--', label='Error Rate')
    ax2_cat.set_ylabel("Error Rate (1-P)", color='red')
    ax2_cat.tick_params(axis='y', labelcolor='red')
    ax2_cat.grid(False)
    
    # Global Legend for one of them?
    # Or just rely on axis colors. 
    
    plt.tight_layout()
    plt.savefig(root_dir / "fan_effect_logprob_error.png", dpi=300)
    print("Saved fan_effect_logprob_error.png")

if __name__ == "__main__":
    visualize_fan_effect_results()
