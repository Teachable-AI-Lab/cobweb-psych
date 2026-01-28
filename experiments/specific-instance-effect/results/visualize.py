from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def visualize_nosofsky_results():
    """
    Visualizes the Specific Instance Effect.
    Compares the classification probability of a High-Frequency Exemplar vs a Low-Frequency Exemplar
    that are equidistant from the category centroid.
    """
    root_dir = Path(__file__).resolve().parent
    csv_path = root_dir / "exp_specific_instance_continuous.csv"
    
    if not csv_path.exists():
        print(f"Error: CSV not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # We focus on the comparison between "A_Freq" and "A_Rare".
    subset = df[df["stimulus_id"].isin(["A_Freq", "A_Rare"])]
    
    # Calculate Mean Probability of Category A
    summary = subset.groupby(["stimulus_id", "frequency_condition"], as_index=False)["prob_class_A"].mean()
    
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Bar Chart
    sns.barplot(
        data=summary, x="frequency_condition", y="prob_class_A", 
        order=["frequent", "rare"], palette=["#e74c3c", "#3498db"], ax=ax, capsize=0.1
    )
    
    ax.set_ylim(0, 1.05)
    ax.set_title("Specific Instance Effect (Nosofsky, 1988)\nImpact of Presentation Frequency on Classification", fontsize=14, pad=15)
    ax.set_ylabel("Probability of Correct Classification (Category A)", fontsize=12)
    ax.set_xlabel("Exemplar Frequency Condition", fontsize=12)
    ax.set_xticklabels(["High Frequency\n(Dist=1.0)", "Low Frequency\n(Dist=1.0)"])
    
    # Add Text
    for i, row in summary.iterrows():
        freq_idx = 0 if row["frequency_condition"] == "frequent" else 1
        ax.text(freq_idx, row["prob_class_A"] + 0.02, f"{row['prob_class_A']:.3f}", ha='center', fontsize=12, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(root_dir / "specific_instance_effect_nosofsky.png", dpi=300)
    print(f"Saved plot to {root_dir}/specific_instance_effect_nosofsky.png")

if __name__ == "__main__":
    visualize_nosofsky_results()
