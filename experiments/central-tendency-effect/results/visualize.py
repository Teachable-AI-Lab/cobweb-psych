import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

# Stimulus Definitions
STIMULI_EXP2 = {
    "Stim4":  {"vec": [1, 1, 1, 0], "cat": "A"},
    "Stim7":  {"vec": [1, 0, 1, 0], "cat": "A"},
    "Stim15": {"vec": [1, 0, 1, 1], "cat": "A"},
    "Stim13": {"vec": [1, 1, 0, 1], "cat": "A"},
    "Stim5":  {"vec": [0, 1, 1, 1], "cat": "A"},
    "Stim12": {"vec": [1, 1, 0, 0], "cat": "B"},
    "Stim2":  {"vec": [0, 1, 1, 0], "cat": "B"}, 
    "Stim14": {"vec": [0, 0, 0, 1], "cat": "B"},
    "Stim10": {"vec": [0, 0, 0, 0], "cat": "B"},
    "Stim1":  {"vec": [1, 0, 0, 1], "cat": "A"}, 
    "Stim3":  {"vec": [1, 0, 0, 0], "cat": "B"}, 
    "Stim6":  {"vec": [1, 1, 1, 1], "cat": "A"}, 
    "Stim8":  {"vec": [0, 0, 1, 0], "cat": "B"}, 
    "Stim9":  {"vec": [0, 1, 0, 1], "cat": "A"}, 
    "Stim11": {"vec": [0, 0, 1, 1], "cat": "A"}, 
    "Stim16": {"vec": [0, 1, 0, 0], "cat": "B"},
}

STIMULI_EXP1 = {
    "Stim6": {"vec": [1, 1, 1, 1], "cat": "A"}, 
    "Stim7": {"vec": [1, 0, 1, 0], "cat": "A"}, 
    "Stim9": {"vec": [0, 1, 0, 1], "cat": "A"}, 
    "Stim10": {"vec": [0, 0, 0, 0], "cat": "B"}, 
    "Stim15": {"vec": [1, 0, 1, 1], "cat": "B"}, 
    "Stim16": {"vec": [0, 1, 0, 0], "cat": "B"}, 
    "Stim5":  {"vec": [0, 1, 1, 1], "cat": "A"}, 
    "Stim13": {"vec": [1, 1, 0, 1], "cat": "A"}, 
    "Stim4":  {"vec": [1, 1, 1, 0], "cat": "A"}, 
    "Stim3":  {"vec": [1, 0, 0, 0], "cat": "B"}, 
    "Stim8":  {"vec": [0, 0, 1, 0], "cat": "B"}, 
    "Stim14": {"vec": [0, 0, 0, 1], "cat": "B"}, 
}

PROTOTYPE_A = [1, 1, 1, 1]
PROTOTYPE_B = [0, 0, 0, 0]

HUMAN_RATINGS_EXP2 = {
    # Train A
    "Stim4":  4.8,
    "Stim7":  5.4,
    "Stim15": 5.1,
    "Stim13": 5.2,
    "Stim5":  5.2,
    # Train B
    "Stim12": 5.0,
    "Stim2":  5.1,
    "Stim14": 5.2,
    "Stim10": 5.5,
    # New Transfer
    "Stim1":  3.7, # A
    "Stim3":  4.4, # B
    "Stim6":  5.3, # A
    "Stim8":  4.1, # B
    "Stim9":  3.3, # A
    "Stim11": 4.1, # A
    "Stim16": 4.9, # B
}

HUMAN_RATINGS_EXP1 = {
    # Train A
    "Stim6": 4.8, "Stim7": 4.6, "Stim9": 4.8,
    # Train B
    "Stim10": 5.2, "Stim15": 4.5, "Stim16": 4.9,
    # Transfer A
    "Stim5": 4.3, "Stim13": 4.4, "Stim4": 3.6,
    # Transfer B
    "Stim3": 3.5, "Stim8": 4.0, "Stim14": 3.2
}

HUMAN_PROBS_EXP2 = {
    "Stim4": 0.78, "Stim7": 0.88, "Stim15": 0.81, "Stim13": 0.88, "Stim5": 0.81,
    "Stim12": 0.84, "Stim2": 0.84, "Stim14": 0.88, "Stim10": 0.97,
    "Stim1": 0.59, "Stim6": 0.94, "Stim9": 0.50, "Stim11": 0.62,
    "Stim3": 0.69, "Stim8": 0.66, "Stim16": 0.84
}

MEDIN_PREDICTIONS_EXP2 = {
    "Stim4": 0.79, "Stim7": 0.94, "Stim15": 0.97, "Stim13": 0.86, "Stim5": 0.86,
    "Stim12": 0.76, "Stim2": 0.76, "Stim14": 0.93, "Stim10": 0.97,
    "Stim1": 0.64, "Stim6": 0.93, "Stim9": 0.57, "Stim11": 0.64,
    "Stim3": 0.61, "Stim8": 0.61, "Stim16": 0.87
}

HUMAN_ERRORS = {
    # Train A
    "Stim4": 4.9, "Stim7": 3.3, "Stim15": 3.2, "Stim13": 4.8, "Stim5": 4.5,
    # Train B
    "Stim12": 5.5, "Stim2": 5.2, "Stim14": 3.9, "Stim10": 3.1
}

def get_distance(stim_id, category, stimuli_defs):
    if stim_id not in stimuli_defs:
        return 0
    vec = stimuli_defs[stim_id]["vec"]
    proto = PROTOTYPE_A if category == "A" else PROTOTYPE_B
    # Hamming distance
    return sum(v != p for v, p in zip(vec, proto))

def plot_line_graph(df, exp_name, stimuli_defs, ratings, human_probs=None, other_preds=None, other_label=None):
    # Filter
    df = df[df['experiment'] == exp_name].copy()
    if df.empty:
        print(f"No data for {exp_name}")
        return

    # Aggregate
    df['p_A'] = df.apply(lambda row: row['p_correct'] if row['correct_cat'] == 'A' else 1 - row['p_correct'], axis=1)
    agg = df.groupby(['stimulus_id', 'correct_cat']).agg({
        'p_A': 'mean'
    }).reset_index()

    # Map Ratings
    agg['rating'] = agg['stimulus_id'].map(ratings)
    
    # Calculate Aligned Rating (1-6 scale transformed to A-ness)
    # If A: Rating. If B: 7 - Rating.
    agg['aligned_rating'] = agg.apply(lambda row: row['rating'] if row['correct_cat'] == 'A' else 7.0 - row['rating'], axis=1)
    
    # Sort descending by aligned rating
    agg = agg.sort_values(by='aligned_rating', ascending=False)
    
    # Prepare Plot Data
    x_labels = agg['stimulus_id'].tolist()
    y_model = agg['p_A'].tolist()
    
    # Human Data Series
    if human_probs:
        # Use provided probabilities
        agg['raw_human'] = agg['stimulus_id'].map(human_probs)
        agg['human_val'] = agg.apply(lambda row: row['raw_human'] if row['correct_cat'] == 'A' else 1.0 - row['raw_human'], axis=1)
        y_human = agg['human_val'].tolist()
        human_label = "Human Probability"
    else:
        # Use Aligned Rating normalized to 0-1
        # (Val - 1) / 5
        y_human = [(v - 1.0)/5.0 for v in agg['aligned_rating'].tolist()]
        human_label = "Human Rating (Normalized)"
        agg['human_val'] = y_human

    # Calculate Spearman
    # Filter out NaNs
    valid = agg.dropna(subset=['p_A', 'human_val'])
    if len(valid) > 1:
        corr, pval = spearmanr(valid['p_A'], valid['human_val'])
        corr_text = f"r={corr:.2f}"
    else:
        corr_text = "N<2"

    plt.figure(figsize=(12, 6))
    sns.set_context("talk")
    sns.set_style("whitegrid")
    
    plt.plot(x_labels, y_model, marker='o', label='Model P(A)', linewidth=2)
    plt.plot(x_labels, y_human, marker='s', label=human_label, linestyle='--', linewidth=2)

    if other_preds:
        agg['raw_other'] = agg['stimulus_id'].map(other_preds)
        agg['other_val'] = agg.apply(lambda row: row['raw_other'] if row['correct_cat'] == 'A' else 1.0 - row['raw_other'], axis=1)
        y_other = agg['other_val'].tolist()
        plt.plot(x_labels, y_other, marker='^', label=other_label, linestyle=':', linewidth=2)
    
    plt.title(f"{exp_name}: Model vs Human ({corr_text})")
    plt.xlabel("Stimuli (Sorted by Human Rating A-ness)")
    plt.ylabel("Probability of Category A")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    
    out_path = Path(__file__).parent / f"{'experiment_1_line_graph' if exp_name == 'Exp1' else 'experiment_2_line_graph'}.png"
    plt.savefig(out_path)
    print(f"Saved {exp_name} line graph to {out_path}")
    plt.close()

def visualize_results():
    results_dir = Path(__file__).parent
    
    # --- Exp 1 ---
    p1 = results_dir / "exp_medin_exp1_results.csv"
    if p1.exists():
        df1 = pd.read_csv(p1)
        plot_line_graph(df1, "Exp1", STIMULI_EXP1, HUMAN_RATINGS_EXP1, human_probs=None)
    else:
        print(f"File not found: {p1}")

    # --- Exp 2 ---
    p2 = results_dir / "exp_medin_exp2_results.csv"
    if not p2.exists():
        print(f"File not found: {p2}")
        return

    df_exp2 = pd.read_csv(p2)
    plot_line_graph(df_exp2, "Exp2", STIMULI_EXP2, HUMAN_RATINGS_EXP2, human_probs=None, other_preds=MEDIN_PREDICTIONS_EXP2, other_label="Medin & Schaffer (1978) Predictions")

    # --- Restore/Maintain Original Exp 2 Plots ---
    # Calculate Probability of Category A
    df_exp2['p_A'] = df_exp2.apply(lambda row: row['p_correct'] if row['correct_cat'] == 'A' else 1 - row['p_correct'], axis=1)
    
    # Calculate Proportion of 'A' Choices
    df_exp2['choice_A'] = df_exp2['choice'].apply(lambda x: 1 if x == 'A' else 0)

    # Aggregation by Stimulus ID
    agg = df_exp2.groupby(['stimulus_id', 'type', 'correct_cat']).agg({
        'p_A': ['mean', 'std', 'sem'],
        'choice_A': ['mean', 'std', 'sem'],
        'accuracy': 'mean'
    }).reset_index()

    # Flatten columns
    agg.columns = ['_'.join(col).strip() if col[1] else col[0] for col in agg.columns.values]
    
    # Sort for plotting: Group by Type (Old/New) then Human Rating
    agg['human_rating_sort'] = agg['stimulus_id'].map(HUMAN_RATINGS_EXP2)
    agg['type_order'] = agg['type'].map({'Old': 1, 'New': 0})
    agg = agg.sort_values(by=['type_order', 'human_rating_sort'], ascending=False)
    
    # Create Label with Type and Category
    agg['label'] = agg.apply(lambda x: f"{x['stimulus_id']}\n({x['type']}, {x['correct_cat']})", axis=1)

    markers = {"Old": "o", "New": "s"}

    # --- Plot 1: Model Probability of Category A (Central Tendency) ---
    plt.figure(figsize=(20, 6))
    sns.set_context("talk")
    sns.set_style("whitegrid")
    
    # Explicitly use plot output path relative to results folder
    out_plot = results_dir / "plot_medin_probabilities.png"

    # Bar plot of p_A_mean
    # Using 'label' for X axis to show Train/Test status
    ax = sns.barplot(data=agg, x='label', y='p_A_mean', hue='correct_cat', dodge=False, palette={'A': 'skyblue', 'B': 'salmon'})
    
    # Add separator line between Old and New stimuli
    n_old = len(agg[agg['type'] == 'Old'])
    if 0 < n_old < len(agg):
        plt.axvline(x=n_old - 0.5, color='black', linestyle='-', linewidth=1.5)

    plt.axhline(0.5, ls='--', color='gray', label='Chance')
    plt.title("Exp 2: Model Probability of Category A by Stimulus")
    plt.xlabel("Stimulus ID")
    plt.ylabel("Probability of Category A")
    plt.xticks(rotation=0)
    plt.legend(title=None)
    plt.tight_layout()
    
    plt.savefig(out_plot)
    print(f"Saved probability plot to: {out_plot}")
    plt.close()

    # --- Plot 2: Classification Accuracy (Old vs New) ---
    agg_type = df_exp2.groupby(['type']).agg({'accuracy': ['mean', 'sem']}).reset_index()
    agg_type.columns = ['type', 'mean_acc', 'sem_acc']

    plt.figure(figsize=(6, 6))
    sns.barplot(data=agg_type, x='type', y='mean_acc', palette="viridis", order=['Old', 'New'])
    plt.errorbar(x=range(len(agg_type)), y=agg_type['mean_acc'], yerr=agg_type['sem_acc'], fmt='none', c='black', capsize=5)
    
    plt.ylim(0, 1)
    plt.axhline(0.5, ls='--', color='gray', label='Chance')
    plt.title("Exp 2: Classification Accuracy: Training vs Transfer")
    plt.ylabel("Accuracy")
    plt.xlabel("Item Type")
    
    out_acc = results_dir / "plot_medin_accuracy.png"
    plt.savefig(out_acc)
    print(f"Saved accuracy plot to: {out_acc}")
    plt.close()

    # --- Plot 3: Probability vs Distance from Prototype ---
    agg['distance'] = agg.apply(lambda row: get_distance(row['stimulus_id'], row['correct_cat'], STIMULI_EXP2), axis=1)
    agg['prob_correct'] = agg.apply(lambda row: row['p_A_mean'] if row['correct_cat'] == 'A' else 1.0 - row['p_A_mean'], axis=1)

    plt.figure(figsize=(10, 8))
    sns.set_context("talk")
    sns.set_style("whitegrid")
    
    subset = agg[['distance', 'prob_correct']].dropna()
    if len(subset) > 1:
        corr, pval = spearmanr(subset['distance'], subset['prob_correct'])
        corr_text = f"Spearman r = {corr:.2f}, p = {pval:.2f}"
    else:
        corr_text = "N < 2"

    p3 = sns.scatterplot(
        data=agg, 
        x='distance', 
        y='prob_correct', 
        hue='correct_cat', 
        style='type',
        markers=markers,
        s=200, 
        alpha=0.8,
        palette={'A': 'skyblue', 'B': 'salmon'}
    )
    
    for line in range(0, agg.shape[0]):
        if pd.isna(agg.distance.iloc[line]) or pd.isna(agg.prob_correct.iloc[line]):
            continue
        stim_label = agg.stimulus_id.iloc[line].replace("Stim", "")
        p3.text(
            agg.distance.iloc[line] + 0.1, 
            agg.prob_correct.iloc[line], 
            stim_label, 
            horizontalalignment='left', 
            size='medium', 
            color='black', 
            weight='semibold'
        )

    plt.title(f"Exp 2: Prob Correct vs Distance\n({corr_text})")
    plt.xlabel("Hamming Distance from Category Prototype")
    plt.ylabel("Probability of Correct Choice")
    
    h, l = plt.gca().get_legend_handles_labels()
    new_h, new_l = [], []
    for handler, label in zip(h, l):
        if label not in ['correct_cat', 'type']:
            new_h.append(handler)
            new_l.append(label)
    plt.legend(new_h, new_l, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    
    out_dist = results_dir / "plot_medin_distance.png"
    plt.savefig(out_dist)
    print(f"Saved distance plot to: {out_dist}")
    plt.close()

    # --- Plot 4: Human Ratings vs Model Probability (Exp 2) ---
    agg['human_rating'] = agg['stimulus_id'].map(HUMAN_RATINGS_EXP2)
    
    plt.figure(figsize=(10, 8))
    sns.set_context("talk")
    sns.set_style("whitegrid")
    
    subset = agg[['human_rating', 'prob_correct']].dropna()
    if len(subset) > 1:
        corr, pval = spearmanr(subset['human_rating'], subset['prob_correct'])
        corr_text = f"Spearman r = {corr:.2f}, p = {pval:.2f}"
    else:
        corr_text = "N < 2"

    p4 = sns.scatterplot(
        data=agg,
        x='human_rating',
        y='prob_correct',
        hue='correct_cat',
        style='type',
        markers=markers,
        s=200,
        alpha=0.8,
        palette={'A': 'skyblue', 'B': 'salmon'}
    )

    for line in range(0, agg.shape[0]):
        if pd.isna(agg.human_rating.iloc[line]) or pd.isna(agg.prob_correct.iloc[line]):
            continue
        stim_label = agg.stimulus_id.iloc[line].replace("Stim", "")
        p4.text(
            agg.human_rating.iloc[line] + 0.1, 
            agg.prob_correct.iloc[line], 
            stim_label, 
            horizontalalignment='left', 
            size='medium', 
            color='black', 
            weight='semibold'
        )
    
    plt.title(f"Exp 2: Model Prob vs Human Ratings\n({corr_text})")
    plt.xlabel("Human Rating")
    plt.ylabel("Model Probability of Correct Choice")
    
    h, l = plt.gca().get_legend_handles_labels()
    new_h, new_l = [], []
    for handler, label in zip(h, l):
        if label not in ['correct_cat', 'type']:
            new_h.append(handler)
            new_l.append(label)
    plt.legend(new_h, new_l, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    
    out_corr = results_dir / "plot_medin_correlation.png"
    plt.savefig(out_corr)
    print(f"Saved correlation plot to: {out_corr}")
    plt.close()

    # --- Plot 5: Model Probability vs Human Probability (Exp 2) ---
    agg['human_prob'] = agg['stimulus_id'].map(HUMAN_PROBS_EXP2)
    
    plt.figure(figsize=(10, 8))
    sns.set_context("talk")
    sns.set_style("whitegrid")
    
    subset = agg[['human_prob', 'prob_correct']].dropna()
    if len(subset) > 1:
        corr, pval = spearmanr(subset['human_prob'], subset['prob_correct'])
        corr_text = f"Spearman r = {corr:.2f}, p = {pval:.2f}"
    else:
        corr_text = "N < 2"
    
    p = sns.scatterplot(
        data=agg,
        x='human_prob',
        y='prob_correct',
        hue='correct_cat',
        style='type',
        markers=markers,
        s=200,
        alpha=0.8,
        palette={'A': 'skyblue', 'B': 'salmon'}
    )
    
    for line in range(0, agg.shape[0]):
        if pd.isna(agg.human_prob.iloc[line]) or pd.isna(agg.prob_correct.iloc[line]):
            continue
        stim_label = agg.stimulus_id.iloc[line].replace("Stim", "")
        p.text(
            agg.human_prob.iloc[line] + 0.005, 
            agg.prob_correct.iloc[line], 
            stim_label, 
            horizontalalignment='left', 
            size='medium', 
            color='black', 
            weight='semibold'
        )

    plt.title(f"Exp 2: Model Prob vs Human Prob\n({corr_text})")
    plt.xlabel("Human Probability of Correct Choice")
    plt.ylabel("Model Probability of Correct Choice")
    
    h, l = plt.gca().get_legend_handles_labels()
    new_h, new_l = [], []
    for handler, label in zip(h, l):
        if label not in ['correct_cat', 'type']:
            new_h.append(handler)
            new_l.append(label)
    plt.legend(new_h, new_l, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    
    out_prob_corr = results_dir / "plot_medin_prob_correlation.png"
    plt.savefig(out_prob_corr)
    print(f"Saved probability correlation plot to: {out_prob_corr}")
    plt.close()
    
    # --- Plot 6: Errors (Exp 2 only) ---
    err_path = results_dir / "exp_medin_exp2_errors.csv"
    if err_path.exists():
        df_err = pd.read_csv(err_path)
        # Assuming format is good, no need to filter by 'experiment' column now that files are split
        # But we check just in case or proceed directly
        
        agg_err = df_err.groupby(['stimulus_id', 'cat']).agg({'error_count': 'mean'}).reset_index()
        agg_err['human_error'] = agg_err['stimulus_id'].map(HUMAN_ERRORS)
        agg_err = agg_err.dropna(subset=['human_error'])
        
        plt.figure(figsize=(10, 8))
        sns.set_context("talk")
        sns.set_style("whitegrid")
        
        subset = agg_err[['human_error', 'error_count']].dropna()
        if len(subset) > 1:
            corr, pval = spearmanr(subset['human_error'], subset['error_count'])
            corr_text = f"Spearman r = {corr:.2f}, p = {pval:.2f}"
        else:
            corr_text = "N < 2"
        
        p6 = sns.scatterplot(
            data=agg_err, 
            x='human_error', 
            y='error_count', 
            hue='cat', 
            s=200, 
            alpha=0.8,
            palette={'A': 'skyblue', 'B': 'salmon'}
        )
        
        for line in range(0, agg_err.shape[0]):
            if pd.isna(agg_err.human_error.iloc[line]) or pd.isna(agg_err.error_count.iloc[line]):
                continue
            stim_label = agg_err.stimulus_id.iloc[line].replace("Stim", "")
            p6.text(
                agg_err.human_error.iloc[line] + 0.1, 
                agg_err.error_count.iloc[line], 
                stim_label, 
                horizontalalignment='left', 
                size='medium', 
                color='black', 
                weight='semibold'
            )
            
        plt.title(f"Exp 2: Cobweb vs Human Training Errors\n({corr_text})")
        plt.xlabel("Human Error Rating")
        plt.ylabel("Cobweb Mean Training Errors")
        
        h, l = plt.gca().get_legend_handles_labels()
        plt.legend(h, l, title=None, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        
        out_err = results_dir / "plot_medin_errors.png"
        plt.savefig(out_err)
        print(f"Saved error correlation plot to: {out_err}")
        plt.close()
    else:
        print("Exp 2 error log not found; skipping Plot 6.")

if __name__ == "__main__":
    visualize_results()
