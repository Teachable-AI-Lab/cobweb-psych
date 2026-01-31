from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def visualize_fan_effect_results():
    """
    Visualizes the results of the Fan Effect (Reder & Ross) experiment.
    Generates a plot of Reaction Time vs Fan Size for both Memory and Category conditions.
    """
    # Paths
    root_dir = Path(__file__).resolve().parent
    csv_path = root_dir / "exp_fan_effect_discrete.csv"
    
    if not csv_path.exists():
        print(f"Error: Results file not found at {csv_path}")
        return

    # Load Data
    df = pd.read_csv(csv_path)
    
    # Convert RT back to Probability then to Error (1 - P)
    df['error'] = 1.0 - (1.0 / df['rt'])
    
    # Calculate Mean Error per Fan Size per Condition
    summary = df.groupby(["condition", "fan_size"], as_index=False)["error"].agg(["mean", "sem"]).reset_index()
    
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
    ax.legend(title="Judgment Task", fontsize=12)
    
    # Add Annotations
    for i, row in summary.iterrows():
        # Offset slightly for clarity
        offset = 12 if row["condition"] == "Memory" else -12
        ax.annotate(f"{row['mean']:.2f}", 
                    (row["fan_size"], row["mean"]),
                    textcoords="offset points", xytext=(0, offset), ha='center', fontsize=10)
        
    plt.tight_layout()
    
    # Save
    out_file = root_dir / "fan_effect_reder_ross_plot.png"
    plt.savefig(out_file, dpi=300)
    print(f"Plot saved to {out_file}")

if __name__ == "__main__":
    visualize_fan_effect_results()
