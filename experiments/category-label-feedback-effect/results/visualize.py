from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_learning_curves(df: pd.DataFrame, out_dir: Path):
    """
    User Request: Plot a separate graph with the X axis being the learning trial 
    and the Y axis being the accuracy for each label rate (Feedback condition), 
    with separate lines for each distortion.
    """
    # Filter only learning phase
    if "phase" in df.columns:
        df_learn = df[df["phase"] == "learning"].copy()
    else:
        # Fallback if CSV format is old
        # But we just updated the script, so assuming new format.
        # If running on old data, this will fail or return empty.
        # Let's assume new format "phase" exists.
        df_learn = df
    
    if df_learn.empty:
        return
    
    # Get unique feedback conditions
    # Column "feedback" (Feedback/No Feedback)
    if "feedback" in df_learn.columns:
        cond_col = "feedback"
    else:
        cond_col = "label_rate" # Fallback
    
    conditions = df_learn[cond_col].unique()
    
    for cond in conditions:
        # Filter data for this condition
        data = df_learn[df_learn[cond_col] == cond]
        
        # Plot with Seaborn aggregation (calculates CI/Error Bands automatically)
        dist_col = "learning_distortion" if "learning_distortion" in data.columns else "distortion"
        
        plt.figure(figsize=(8, 6))
        sns.lineplot(
            data=data, x="epoch", y="accuracy", hue=dist_col,
            style=dist_col, markers=True, dashes=False, linewidth=2, palette="viridis"
        )
        
        plt.title(f"Learning Curve: {cond}", fontsize=14)
        plt.xlabel("Learning Trial (Epoch)", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.ylim(-0.05, 1.05)
        plt.legend(title="Learning Distortion")
        
        # Ensure integer ticks for epoch if numeric
        if pd.api.types.is_numeric_dtype(data["epoch"]):
            plt.xticks(sorted(data["epoch"].unique()))
            
        plt.grid(True, linestyle="--", alpha=0.6)
        
        safe_name = str(cond).replace(" ", "_").lower().replace(".","")
        plt.tight_layout()
        plt.savefig(out_dir / f"learning_curve_{safe_name}.png", dpi=200)
        plt.close()

def plot_transfer_results(df: pd.DataFrame, out_dir: Path):
    """
    Plot transfer phase accuracy (Old, New Low/Med/High, Prototype).
    """
    if "phase" not in df.columns: return
    df_trans = df[df["phase"] == "transfer"].copy()
    if df_trans.empty: return

    # Focus on Feedback condition
    # Aggregation
    agg = df_trans.groupby(["feedback", "learning_distortion", "stim_type"], as_index=False)["accuracy"].mean()
    
    # Custom sort order for X-axis (stim_type)
    type_order = ["Prototype", "Old", "New_Low", "New_Medium", "New_High", "Unrelated"]
    present_types = [t for t in type_order if t in agg["stim_type"].unique()]
    
    # Plot for Feedback condition
    fb_data = agg[agg["feedback"] == "Feedback"]
    if not fb_data.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=fb_data, x="learning_distortion", y="accuracy", hue="stim_type",
            hue_order=present_types, palette="rocket"
        )
        plt.title("Transfer Accuracy by Learning Condition (Feedback Group)", fontsize=14)
        
        # Change X-axis label to match Homa: "Learning Condition"
        plt.xlabel("Learning Condition (Distortion Level)", fontsize=12)
        plt.ylabel("Transfer Accuracy", fontsize=12)
        plt.ylim(0, 1.05)
        plt.legend(title="Transfer Item Type", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(out_dir / "transfer_accuracy_feedback.png", dpi=200)
        plt.close()

def main():
    csv_path = Path(__file__).resolve().parent / "exp_category_label_feedback_continuous.csv"
    out_dir = Path(__file__).resolve().parent
    
    if not csv_path.exists():
        print(f"CSV not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    sns.set_theme(style="whitegrid")
    
    # Run plots
    plot_learning_curves(df, out_dir)
    plot_transfer_results(df, out_dir)
    
    print("Plots generated.")


if __name__ == "__main__":
	main()
