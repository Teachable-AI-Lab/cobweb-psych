from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def visualize_medin_results():
    """
    Visualizes Medin & Schaffer (1978) Replication.
    Plots Observed vs Predicted classification probabilities for the 16 stimuli.
    """
    
    # Locate CSV
    root_dir = Path(__file__).resolve().parent
    # Filename matches what was saved in exp_medin.py
    # We assumed 'face' or 'geometric', the script saved "exp_medin_continuous_face.csv" by default in main
    csv_path = root_dir / "exp_medin_continuous_face.csv"
    
    # Try alternate if not found (maybe user changed to geometric)
    if not csv_path.exists():
        csv_path = root_dir / "exp_medin_continuous_geometric.csv"
        
    if not csv_path.exists():
        print(f"Error: CSV not found at {root_dir}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Aggregate Predictions
    # Group by Stimulus ID (1-16)
    summary = df.groupby("stimulus", as_index=False).agg({
        "predicted_prob_correct": "mean",
        "observed_prob": "mean" # Should be constant per stimulus
    })
    
    # Calculate Mean Correlation
    corr = np.corrcoef(summary["predicted_prob_correct"], summary["observed_prob"])[0, 1]
    
    sns.set_theme(style="whitegrid")
    
    # Plot Scatter
    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=summary, x="observed_prob", y="predicted_prob_correct",
        s=100, color="royalblue", edgecolor="black", alpha=0.8
    )
    
    # Add identity line for perfect fit reference
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.5)
    
    # Add labels for points (Stimulus IDs)
    for i, row in summary.iterrows():
        plt.text(
            row["observed_prob"] + 0.01, row["predicted_prob_correct"], 
            str(int(row["stimulus"])), fontsize=9
        )
        
    plt.title(f"Medin & Schaffer (1978) Central Tendency\nModel Fit (Correlation r = {corr:.3f})", fontsize=14)
    plt.xlabel("Observed Probability (Human Data)", fontsize=12)
    plt.ylabel("Predicted Probability (Cobweb)", fontsize=12)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    out_file = root_dir / "medin_scatter_fit.png"
    plt.savefig(out_file, dpi=300)
    print(f"Saved scatter plot to {out_file}")

if __name__ == "__main__":
    visualize_medin_results()
