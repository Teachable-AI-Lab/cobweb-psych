from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_experiment(csv_path, out_file, title_suffix="Exp 1"):
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(7, 6))
    
    sns.barplot(
        data=df, 
        x="condition", 
        y="prop_consistent", 
        palette="Blues_d", 
        ax=ax,
        capsize=0.1,
        errwidth=1.5
    )
    
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='gray', linestyle='--', label="Chance (No Preference)")
    
    ax.set_xlabel("Training Condition", fontsize=12)
    ax.set_ylabel("Proportion of Correlation-Consistent Choices", fontsize=12)
    ax.set_title(f"Correlated Feature Effect (Medin et al., 1982)\nPreference for Correlation-Consistent Test Items - {title_suffix}", fontsize=13)
    
    ax.legend(loc='lower right')
    
    fig.tight_layout()
    fig.savefig(out_file, dpi=300)
    print(f"Saved plot to {out_file}")

def main():
    sns.set_theme(style="whitegrid")
    root_dir = Path(__file__).resolve().parent 
    
    # Experiment 1
    plot_experiment(
        root_dir / "exp1_correlated_feature_medin.csv",
        root_dir / "exp1_correlated_feature_medin_1982.png",
        title_suffix="Exp 1 (Binary)"
    )

    # Experiment 2
    plot_experiment(
        root_dir / "exp2_correlated_feature_medin.csv",
        root_dir / "exp2_correlated_feature_medin_1982.png",
        title_suffix="Exp 2 (Dimensions)"
    )
    
    # Experiment 3
    exp3_csv = root_dir / "exp3_correlated_feature_medin.csv"
    if exp3_csv.exists():
        df3 = pd.read_csv(exp3_csv)
        
        # Melt for plotting
        df_long = pd.melt(
            df3, 
            id_vars=["seed"], 
            value_vars=["prop_high_typ_control", "prop_corr_conflict"],
            var_name="Test Condition", 
            value_name="Proportion"
        )
        
        df_long["Test Condition"] = df_long["Test Condition"].replace({
            "prop_high_typ_control": "Control Pairs\n(Preference for High Typicality)",
            "prop_corr_conflict": "Conflict Pairs\n(Preference for Correlation)"
        })
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(
            data=df_long, x="Test Condition", y="Proportion", palette="Greens_d", ax=ax, capsize=0.1
        )
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color='gray', linestyle='--', label="Chance")
        ax.set_title("Correlated Feature Effect Exp 3 (Medin et al., 1982)\nStrategy Shift: Feature Count vs Relational Coding", fontsize=12)
        ax.set_ylabel("Proportion of Choices Consistent with Strategy", fontsize=11)
        ax.set_xlabel("")
        
        fig.tight_layout()
        fig.savefig(root_dir / "exp3_correlated_feature_medin_1982.png", dpi=300)
        print(f"Saved plot to {root_dir / 'exp3_correlated_feature_medin_1982.png'}")
        
    # Experiment 4
    exp4_csv = root_dir / "exp4_correlated_feature_medin.csv"
    if exp4_csv.exists():
        df4 = pd.read_csv(exp4_csv, dtype={'pattern': str})
        
        # Categorize patterns
        # Old T: 1111, 1100, 0111, 1000
        # Old M: 0010, 0001, 1010, 0101
        # New Correlated (S3/S4 same): 0000, 0011, 0100, 1011
        # New Uncorrelated (S3/S4 diff): 1110, 1101, 0110, 1001
        
        def classify_pattern(p):
            p = str(p).zfill(4)
            
            old_t = ["1111", "1100", "0111", "1000"]
            old_m = ["0010", "0001", "1010", "0101"]
            new_corr = ["0000", "0011", "0100", "1011"]
            new_uncorr = ["1110", "1101", "0110", "1001"]
            
            if p in old_t: return "Old Terrigitis"
            if p in old_m: return "Old Midosis"
            if p in new_corr: return "New Correlated"
            if p in new_uncorr: return "New Uncorrelated"
            return "Other"

        df4["Pattern Type"] = df4["pattern"].apply(classify_pattern)
        
        # Plot
        summary = df4.groupby(["Pattern Type"], as_index=False)["p_terrigitis"].mean()
        # Order: Old T, New Correlated, New Uncorrelated, Old M
        order = ["Old Terrigitis", "New Correlated", "New Uncorrelated", "Old Midosis"]
        
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.barplot(
            data=df4, x="Pattern Type", y="p_terrigitis", order=order, palette="coolwarm", ax=ax, capsize=0.1
        )
        
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color='gray', linestyle='--')
        ax.set_ylabel("Probability of Classification as Terrigitis", fontsize=11)
        ax.set_title("Correlated Feature Effect Exp 4 (Medin et al., 1982)\nClassification of Transfers", fontsize=13)
        
        fig.tight_layout()
        fig.savefig(root_dir / "exp4_correlated_feature_medin_1982.png", dpi=300)
        print(f"Saved plot to {root_dir / 'exp4_correlated_feature_medin_1982.png'}")

if __name__ == "__main__":
    main()
