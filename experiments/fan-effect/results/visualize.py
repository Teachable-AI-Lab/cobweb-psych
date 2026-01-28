from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def visualize_fan_effect_results():
    """
    Visualizes the results of the Fan Effect experiment.
    Generates a plot of Reaction Time vs Fan Size.
    """
    # Paths
    root_dir = Path(__file__).resolve().parent
    csv_path = root_dir / "exp_fan_effect_discrete.csv"
    
    if not csv_path.exists():
        print(f"Error: Results file not found at {csv_path}")
        return

    # Load Data
    df = pd.read_csv(csv_path)
    
    # Calculate Mean RT per Fan Size
    # We aggregate over seeds and predicates
    summary = df.groupby(["fan_size"], as_index=False)["simulated_rt"].mean()
    
    # Plotting
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot Data Points
    sns.lineplot(
        data=summary, x="fan_size", y="simulated_rt", 
        marker="o", markersize=10, linewidth=2.5, color="#2c3e50", ax=ax, label="Simulated Data"
    )
    
    # Add Regression Line to highlight trend
    x = summary["fan_size"]
    y = summary["simulated_rt"]
    z = np.polyfit(x, y, 1) # Linear fit
    p = np.poly1d(z)
    
    ax.plot(x, p(x), "--", color="#e74c3c", linewidth=2, label=f"Linear Trend (Slope={z[0]:.2f})")
    
    # Formatting
    ax.set_title("Fan Effect Simulation (Anderson, 1974)\nRetrieval Inteference in Cobweb", fontsize=14, pad=15)
    ax.set_xlabel("Fan Size (Facts associated with Concept)", fontsize=12)
    ax.set_ylabel("Simulated Reaction Time (1 / Probability)", fontsize=12)
    ax.set_xticks(summary["fan_size"].unique())
    ax.legend(fontsize=11)
    
    # Add Annotations
    for i, row in summary.iterrows():
        ax.annotate(f"{row['simulated_rt']:.2f}", 
                    (row["fan_size"], row["simulated_rt"]),
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)
        
    plt.tight_layout()
    
    # Save
    out_file = root_dir / "fan_effect_results_plot.png"
    plt.savefig(out_file, dpi=300)
    print(f"Plot saved to {out_file}")

if __name__ == "__main__":
    visualize_fan_effect_results()
