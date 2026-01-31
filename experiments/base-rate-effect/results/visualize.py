from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def visualize_homa_vosburgh():
    """
    Visualizes Homa & Vosburgh (1976) Experiment 1.
    Goal: Plot Mean % Correct on immediate test for New and Prototype stimuli
    as a function of Category Size, Distortion Level, and Training Condition.
    
    Generates two line graphs:
    1. Uniform-Low Condition: Accuracy vs Category Size (Lines: P, L, M, H)
    2. Mixed Condition: Accuracy vs Category Size (Lines: P, L, M, H)
    """
    
    root_dir = Path(__file__).resolve().parent
    csv_path = root_dir / "exp_base_rate_continuous.csv"
    
    if not csv_path.exists():
        print(f"Error: CSV not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Filter: "New and Prototype stimuli"
    # We exclude "Old" usually for this specific plot described in prompt.
    # "Mean percentage correct... for new and prototype... as function..."
    # stim_label holds "Prototype", "Low", "Medium", "High" (from New), and "Old".
    
    subset = df[df["stim_label"].isin(["Prototype", "Low", "Medium", "High"])].copy()
    
    # Aggregate Mean Accuracy
    curr = subset.groupby(["condition", "cat_size", "stim_label"], as_index=False)["correct"].mean()
    
    # Setup styles
    sns.set_theme(style="whitegrid")
    
    # Order for distortion levels
    hue_order = ["Prototype", "Low", "Medium", "High"]
    markers = {"Prototype": "o", "Low": "s", "Medium": "^", "High": "X"}
    dashes = {"Prototype": (2,2), "Low": "", "Medium": "", "High": ""}
    
    conditions = ["Uniform-Low", "Mixed"]
    
    for cond in conditions:
        plt.figure(figsize=(7, 6))
        
        data_cond = curr[curr["condition"] == cond]
        
        # Plot
        sns.lineplot(
            data=data_cond, x="cat_size", y="correct", hue="stim_label", style="stim_label",
            hue_order=hue_order, style_order=hue_order,
            markers=True, dashes=False, markersize=8, linewidth=2
        )
        
        plt.title(f"Homa & Vosburgh (1976): {cond} Condition", fontsize=14)
        plt.ylabel("Mean Percent Correct", fontsize=12)
        plt.xlabel("Category Size (Breadth)", fontsize=12)
        plt.ylim(0, 1.05)
        plt.xticks([3, 6, 9])
        
        plt.legend(title="Test Stimulus", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        safe_name = cond.lower().replace("-", "_")
        out_file = root_dir / f"homa_vosburgh_{safe_name}.png"
        plt.tight_layout()
        plt.savefig(out_file, dpi=300)
        print(f"Saved plot to {out_file}")

if __name__ == "__main__":
    visualize_homa_vosburgh()
