from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def visualize_fan_effect_results():
    """
    Visualizes the results of the Fan Effect (Reder & Ross) experiment.
    Generates a plot of Probability of Error vs Fan Size for both Memory and Category conditions.
    """
    # Paths
    root_dir = Path(__file__).resolve().parent
    csv_path = root_dir / "exp_fan_effect_discrete.csv"
    
    if not csv_path.exists():
        print(f"Error: Results file not found at {csv_path}")
        return

    # Load Data
    df = pd.read_csv(csv_path)
    
    # 1. Filter for Targets only
    # The standard Fan Effect is analyzing retrieval time/error for learned facts (Targets).
    # Foils are used to ensure discriminatory capability but usually not for the main interaction plot
    # unless analyzing Foil rejection times.
    df_targets = df[df["type"] == "Target"].copy()
    
    # 2. Transform Wide to Long format
    # We have 'prob_memory' and 'prob_category' in each row.
    # We want rows to be: fan_size, condition, error
    
    # Memory
    mem_df = df_targets[["fan_size", "prob_memory"]].copy()
    mem_df["condition"] = "Memory"
    # Logic: Error = 1 - Probability of Correct response.
    # The 'prob_memory' is the prob of the correct predicate. So 1-P is error.
    mem_df["error"] = 1.0 - mem_df["prob_memory"]
    
    # Category
    cat_df = df_targets[["fan_size", "prob_category"]].copy()
    cat_df["condition"] = "Category"
    cat_df["error"] = 1.0 - cat_df["prob_category"]
    
    # Combine
    plot_df = pd.concat([mem_df[["fan_size", "condition", "error"]], 
                         cat_df[["fan_size", "condition", "error"]]])
    
    # Calculate Mean Error per Fan Size per Condition
    summary = plot_df.groupby(["condition", "fan_size"], as_index=False)["error"].agg(["mean", "sem"]).reset_index()
    
    # Plotting
    sns.set_theme(style="whitegrid")
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Define style for BW plot:
    # Memory: Black line, Black filled circles
    # Category: Black line (dashed?), White filled circles
    
    # Memory Condition
    data_mem = summary[summary["condition"] == "Memory"]
    ax.errorbar(
        x=data_mem["fan_size"], y=data_mem["mean"], yerr=data_mem["sem"],
        fmt='-o', color='black', mfc='black', mec='black', markersize=10, 
        linewidth=2, capsize=5, label="Memory (Exact)"
    )
    
    # Category Condition
    data_cat = summary[summary["condition"] == "Category"]
    ax.errorbar(
        x=data_cat["fan_size"], y=data_cat["mean"], yerr=data_cat["sem"],
        fmt='--o', color='black', mfc='white', mec='black', markersize=10, 
        linewidth=2, capsize=5, label="Category (Plausibility)"
    )
    
    # Formatting
    ax.set_title("Cobweb Simulation of Fan Effect (Reder & Ross, 1983)", fontsize=16, pad=20)
    ax.set_xlabel("Fan Size (Facts associated with Concept)", fontsize=14)
    ax.set_ylabel("Probability of Error (1 - Accuracy)", fontsize=14)
    ax.set_xticks([1, 2, 3])
    # Auto Y-lim, but ensure 0 is included if possible or meaningful range
    
    ax.legend(title="Judgment Task", fontsize=12)
    
    # Add Annotations
    for i, row in summary.iterrows():
        # Offset slightly for clarity
        offset = 15 if row["condition"] == "Memory" else -20
        # Check if values are close to 0 or 1 to adjust offset direction
        if row['mean'] < 0.1: offset = 15
        
        ax.annotate(f"{row['mean']:.2f}", 
                    (row["fan_size"], row["mean"]),
                    textcoords="offset points", xytext=(0, offset), ha='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
        
    plt.tight_layout()
    
    # Save
    out_file = root_dir / "fan_effect_reder_ross_plot.png"
    plt.savefig(out_file, dpi=300)
    print(f"Plot saved to {out_file}")

if __name__ == "__main__":
    visualize_fan_effect_results()
