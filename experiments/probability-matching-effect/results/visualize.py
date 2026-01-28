from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main(
    csv_path=Path(__file__).resolve().parent / "exp_probability_matching_gluck.csv",
    out_dir=Path(__file__).resolve().parent,
):
    sns.set_theme(style="whitegrid")
    if not csv_path.exists():
        print(f"CSV not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)

    # DataFrame structure: seed, symptom, true_p_rare, pred_p_rare
    # We want to plot Mean Pred P(Rare) vs True P(Rare) for each Symptom.

    # Melt the dataframe to have "Type" (True vs Model)
    # df_melt columns: seed, symptom, Value, Type
    
    # First, get the true values. They are constant per symptom, but present in every row.
    # We can just take the first occurrence or average (same thing).
    
    # Actually, simpler to melt:
    df_long = pd.melt(
        df, 
        id_vars=["seed", "symptom"], 
        value_vars=["true_p_rare", "pred_p_rare"], 
        var_name="Type", 
        value_name="Probability"
    )
    
    # Rename Types for Legend
    df_long["Type"] = df_long["Type"].replace({
        "true_p_rare": "Objective Probability",
        "pred_p_rare": "Model Estimate"
    })
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Bar plot
    # x=symptom, y=Probability, hue=Type
    sns.barplot(
        data=df_long, 
        x="symptom", 
        y="Probability", 
        hue="Type", 
        palette={"Objective Probability": "black", "Model Estimate": "gray"},
        ax=ax,
        edgecolor="black",
        capsize=0.05
    )
    
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Symptom", fontsize=12)
    ax.set_ylabel("Probability of Rare Disease P(R|S)", fontsize=12)
    ax.set_title("Probability Matching: Subjective vs Objective Probabilities\n(Gluck & Bower, 1988, Exp 1)", fontsize=13)
    ax.legend(title=None)
    
    # Add textual labels for the bars? Maybe too cluttered.
    
    fig.tight_layout()
    out_file = out_dir / "probability_matching_gluck_1988.png"
    fig.savefig(out_file, dpi=300)
    print(f"Saved plot to {out_file}")


if __name__ == "__main__":
	main()
