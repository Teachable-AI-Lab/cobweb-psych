from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def visualize_hayes_roth_results():
    """
    Visualizes the Hayes-Roth Specific Instance Effect.
    Comparison groups:
    1. A_Freq_Dist1 (Freq=10, Dist=1) -> "Frequent Exemplar"
    2. A_Proto (Freq=0, Dist=0)      -> "Prototype"
    3. A_Rare_Dist1 (Freq=1, Dist=1) -> "Rare Exemplar"
    
    Effect: 
    - Recognition: Freq > Proto (Specific Instance Effect)
    - Classification: Proto >= Freq (Prototype Enhancement in Class)
    """
    root_dir = Path(__file__).resolve().parent
    csv_path = root_dir / "exp_specific_instance_discrete.csv"
    
    if not csv_path.exists():
        print(f"Error: CSV not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Filter for the key comparison items
    # Note: Using the IDs defined in the new script
    target_ids = ["A_Freq_Dist1", "A_Proto", "A_Rare_Dist1"]
    subset = df[df["stimulus_id"].isin(target_ids)].copy()
    
    # Map IDs to Display Names
    name_map = {
        "A_Freq_Dist1": "Frequent Exemplar\n(Dist=1, Freq=10)",
        "A_Proto": "Prototype\n(Dist=0, Freq=0)",
        "A_Rare_Dist1": "Rare Exemplar\n(Dist=1, Freq=1)"
    }
    subset["Item Type"] = subset["stimulus_id"].map(name_map)
    
    # Order for plotting
    plot_order = ["Frequent Exemplar\n(Dist=1, Freq=10)", "Rare Exemplar\n(Dist=1, Freq=1)", "Prototype\n(Dist=0, Freq=0)"]
    
    # Aggregation
    summary = subset.groupby(["Item Type"], as_index=False)[["recognition_score", "classification_accuracy"]].mean()
    
    sns.set_theme(style="whitegrid")
    
    # --- Plot 1: Recognition (Log Likelihood) ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.barplot(
        data=summary, x="Item Type", y="recognition_score", order=plot_order,
        palette=["#27ae60", "#2980b9", "#c0392b"], ax=ax1
    )
    ax1.set_title("Hayes-Roth (1977): Recognition Memory Strength", fontsize=14, pad=15)
    ax1.set_ylabel("Recognition Feature Likelihood (Log Prob)", fontsize=12)
    ax1.set_xlabel("")
    
    # Add labels
    for i, label in enumerate(plot_order):
        row = summary[summary["Item Type"] == label]
        if not row.empty:
            val = row["recognition_score"].values[0]
            ax1.text(i, val, f"{val:.2f}", ha='center', va='bottom', fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(root_dir / "hayes_roth_recognition.png", dpi=300)
    print(f"Saved recognition plot to {root_dir}/hayes_roth_recognition.png")
    
    # --- Plot 2: Classification Accuracy ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.barplot(
        data=summary, x="Item Type", y="classification_accuracy", order=plot_order,
        palette="muted", ax=ax2
    )
    ax2.set_title("Hayes-Roth (1977): Classification Accuracy", fontsize=14, pad=15)
    ax2.set_ylabel("Probability of Correct Category (Club 1)", fontsize=12)
    ax2.set_xlabel("")
    ax2.set_ylim(0, 1.05)
    
    for i, label in enumerate(plot_order):
        row = summary[summary["Item Type"] == label]
        if not row.empty:
            val = row["classification_accuracy"].values[0]
            ax2.text(i, val + 0.02, f"{val:.2f}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(root_dir / "hayes_roth_classification.png", dpi=300)
    print(f"Saved classification plot to {root_dir}/hayes_roth_classification.png")

if __name__ == "__main__":
    visualize_hayes_roth_results()
