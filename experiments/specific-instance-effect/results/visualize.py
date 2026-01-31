from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def align_zero_axes(ax1, ax2):
    """
    Adjusts the y-limits of two axes (ax1, ax2) so that their y=0 lines align horizontally.
    Uses the 'Maximum Absolute Range' method to force 0 to be centered.
    """
    # Get current data limits
    lims1 = ax1.get_ylim()
    lims2 = ax2.get_ylim()
    
    # Identify the larger range requirement for each axis
    # Force symmetric limits centered on 0
    m1 = max(abs(lims1[0]), abs(lims1[1])) * 1.15
    ax1.set_ylim(-m1, m1)
    
    m2 = max(abs(lims2[0]), abs(lims2[1])) * 1.15
    ax2.set_ylim(-m2, m2)

def visualize_hayes_roth_results():
    """
    Visualizes the Hayes-Roth Specific Instance Effect.
    Generates side-by-side bar comparison of Model vs Human data.
    """
    root_dir = Path(__file__).resolve().parent
    csv_path = root_dir / "exp_specific_instance_discrete.csv"
    
    if not csv_path.exists():
        print(f"Error: CSV not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # --- 1. Process Model Data ---
    target_tags = ["Freq_Exemplar", "Prototype", "Rare_Exemplar"]
    subset = df[df["tag"].isin(target_tags)].copy()
    
    type_map = {
        "Freq_Exemplar": "Frequent",
        "Prototype": "Prototype",
        "Rare_Exemplar": "Rare"
    }
    subset["Type"] = subset["tag"].map(type_map)
    
    # Aggregate
    model_summary = subset.groupby(["Type"], as_index=False).agg(
        recog_mean=("recog_score", "mean"),
        recog_std=("recog_score", "std"),
        class_mean=("p_club1", "mean"),
        class_std=("p_club1", "std"),
        count=("seed", "count")
    )
    # SEM
    model_summary["recog_sem"] = model_summary["recog_std"] / np.sqrt(model_summary["count"])
    model_summary["class_sem"] = model_summary["class_std"] / np.sqrt(model_summary["count"])
    model_summary = model_summary.set_index("Type")

    # --- 2. Process Human Data (Table 2) ---
    # Manually aggregated means from previous step
    # Recog: Freq=3.40, Rare=-1.14, Proto=0.49
    # Class (Inv Z): Freq=2.45, Rare=2.31, Proto=2.82
    
    data_groups = ["Rare", "Frequent", "Prototype"]
    
    human_data = {
        "Recognition": {
            "Rare": -1.14, "Frequent": 3.40, "Prototype": 0.49
        },
        "Classification": {
            "Rare": 2.31, "Frequent": 2.45, "Prototype": 2.82
        }
    }

    sns.set_theme(style="white", context="talk")
    
    def plot_metric(metric_key, model_col, model_err_col, title, y1_lab, y2_lab, filename):
        fig, ax1 = plt.subplots(figsize=(10, 7))
        
        # Setup Data
        x = np.arange(len(data_groups))
        width = 0.35
        
        m_means = [model_summary.loc[g, model_col] for g in data_groups]
        m_errs = [model_summary.loc[g, model_err_col] for g in data_groups]
        h_vals = [human_data[metric_key][g] for g in data_groups]
        
        # Plot Model (Left) - Hatched, White filled
        rects1 = ax1.bar(x - width/2, m_means, width, yerr=m_errs,
                        label='Model', color='white', edgecolor='black', hatch='///', capsize=5)
        
        # Plot Human (Right) - Solid Gray
        ax2 = ax1.twinx()
        rects2 = ax2.bar(x + width/2, h_vals, width,
                        label='Human', color='gray', edgecolor='black', bottom=0)
        
        # Labels and Title
        ax1.set_title(title, pad=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels([g + "\nExemplar" if g != "Prototype" else g for g in data_groups])
        ax1.set_ylabel(y1_lab, color='black')
        ax2.set_ylabel(y2_lab, color='dimgray')
        
        # Align Zeroes (Center Symmetry method)
        align_zero_axes(ax1, ax2)
        
        # Add Zero Value Labels
        # Recalculate Y positions based on new limits if needed, but relative coords are trickier
        # Standard matplotlib text uses data coords, so we are fine.
        
        # Model Labels
        for r, v in zip(rects1, m_means):
            # Dynamic offset based on the axis scale
            y_range = ax1.get_ylim()[1] - ax1.get_ylim()[0]
            offset = y_range * 0.05
            y_pos = v + offset if v >= 0 else v - offset
            va = 'bottom' if v >= 0 else 'top'
            ax1.text(r.get_x() + r.get_width()/2, y_pos, f"{v:.2f}", 
                     ha='center', va=va, fontsize=10, color='black')
            
        # Human Labels
        for r, v in zip(rects2, h_vals):
            y_range = ax2.get_ylim()[1] - ax2.get_ylim()[0]
            offset = y_range * 0.05
            y_pos = v + offset if v >= 0 else v - offset
            va = 'bottom' if v >= 0 else 'top'
            ax2.text(r.get_x() + r.get_width()/2, y_pos, f"{v:.2f}",
                     ha='center', va=va, fontsize=10, color='dimgray')
        
        # Draw Horizontal Zero Line (on ax1)
        ax1.axhline(0, color='black', linewidth=1)
        
        # Legend
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='upper left', frameon=True)
        
        plt.tight_layout()
        plt.savefig(root_dir / filename, dpi=300)
        print(f"Saved {filename}")

    # 1. Classification
    plot_metric(
        "Classification", "class_mean", "class_sem",
        "Hayes-Roth (1977): Classification",
        "Model Probability (Club 1)", "Human Rating (Inv Z-Score)",
        "hayes_roth_classification.png"
    )

    # 2. Recognition
    plot_metric(
        "Recognition", "recog_mean", "recog_sem",
        "Hayes-Roth (1977): Recognition Confidence",
        "Model Log Likelihood", "Human Rating (Z-Score)",
        "hayes_roth_recognition.png"
    )

if __name__ == "__main__":
    visualize_hayes_roth_results()
