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
    
    # Calculate Mean RT per Fan Size per Condition
    # The new script outputs 'rt' instead of 'simulated_rt' and includes a 'condition' column
    summary = df.groupby(["condition", "fan_size"], as_index=False)["rt"].agg(["mean", "sem"]).reset_index()
    
    # Plotting
    sns.set_theme(style="whitegrid")
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Define colors
    palette = {"Memory": "#2c3e50", "Category": "#e74c3c"}
    
    # Plot Lines
    sns.lineplot(
        data=summary, x="fan_size", y="mean", hue="condition", 
        marker="o", markersize=12, linewidth=3, palette=palette, ax=ax
    )
    
    # Add Error Bars
    for cond in summary["condition"].unique():
        data = summary[summary["condition"] == cond]
        ax.errorbar(
            x=data["fan_size"], y=data["mean"], yerr=data["sem"], 
            fmt='none', ecolor=palette[cond], capsize=5, elinewidth=2
        )
    
    # Formatting
    ax.set_title("Cobweb Simulation of Fan Effect (Reder & Ross, 1983)", fontsize=16, pad=20)
    ax.set_xlabel("Fan Size (Facts associated with Concept)", fontsize=14)
    ax.set_ylabel("Simulated Reaction Time (1 / Probability)", fontsize=14)
    ax.set_xticks([1, 2, 3])
    ax.legend(title="Judgment Task", fontsize=12)
    
    # Add Annotations
    for i, row in summary.iterrows():
        ax.annotate(f"{row['mean']:.2f}", 
                    (row["fan_size"], row["mean"]),
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)
        
    plt.tight_layout()
    
    # Save
    out_file = root_dir / "fan_effect_reder_ross_plot.png"
    plt.savefig(out_file, dpi=300)
    print(f"Plot saved to {out_file}")

if __name__ == "__main__":
    visualize_fan_effect_results()
