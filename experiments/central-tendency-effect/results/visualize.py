import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Stimulus Definitions
STIMULI_DEFS = {
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

PROTOTYPE_A = [1, 1, 1, 1]
PROTOTYPE_B = [0, 0, 0, 0]

HUMAN_RATINGS = {
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

HUMAN_PROBS = {
    "Stim4": 0.78, "Stim7": 0.88, "Stim15": 0.81, "Stim13": 0.88, "Stim5": 0.81,
    "Stim12": 0.84, "Stim2": 0.84, "Stim14": 0.88, "Stim10": 0.97,
    "Stim1": 0.59, "Stim6": 0.94, "Stim9": 0.50, "Stim11": 0.62,
    "Stim3": 0.69, "Stim8": 0.66, "Stim16": 0.84
}

def get_distance(stim_id, category):
    if stim_id not in STIMULI_DEFS:
        return 0
    vec = STIMULI_DEFS[stim_id]["vec"]
    proto = PROTOTYPE_A if category == "A" else PROTOTYPE_B
    # Hamming distance
    return sum(v != p for v, p in zip(vec, proto))

def visualize_results():
    # Load data
    results_path = Path(__file__).parent / "exp_central_tendency_continuous.csv"
    if not results_path.exists():
        print(f"File not found: {results_path}")
        return

    df = pd.read_csv(results_path)

    # Calculate Probability of Category A
    # If correct_cat is A, p(A) = p_correct
    # If correct_cat is B, p(A) = 1 - p_correct
    df['p_A'] = df.apply(lambda row: row['p_correct'] if row['correct_cat'] == 'A' else 1 - row['p_correct'], axis=1)
    
    # Calculate Proportion of 'A' Choices
    df['choice_A'] = df['choice'].apply(lambda x: 1 if x == 'A' else 0)

    # Aggregation by Stimulus ID
    agg = df.groupby(['stimulus_id', 'type', 'correct_cat']).agg({
        'p_A': ['mean', 'std', 'sem'],
        'choice_A': ['mean', 'std', 'sem'],
        'accuracy': 'mean'
    }).reset_index()

    # Flatten columns
    agg.columns = ['_'.join(col).strip() if col[1] else col[0] for col in agg.columns.values]
    
    # Sort for plotting: Group by Type (Old/New) then ID
    agg['sort_order'] = agg['type'].map({'Old': 0, 'New': 1})
    agg = agg.sort_values(by=['sort_order', 'stimulus_id'])
    
    # Create Label with Type and Category
    agg['label'] = agg.apply(lambda x: f"{x['stimulus_id']}\n({x['type']}, {x['correct_cat']})", axis=1)

    # --- Plot 1: Model Probability of Category A (Central Tendency) ---
    plt.figure(figsize=(20, 6))
    sns.set_context("talk")
    sns.set_style("whitegrid")

    # Bar plot of p_A_mean
    # Using 'label' for X axis to show Train/Test status
    ax = sns.barplot(data=agg, x='label', y='p_A_mean', hue='correct_cat', dodge=False, palette={'A': 'skyblue', 'B': 'salmon'})
    
    # Error bars removed as requested
    # plt.errorbar(x=range(len(agg)), y=agg['p_A_mean'], yerr=agg['p_A_sem'], fmt='none', c='black', capsize=5)

    # Add separator line between Old and New stimuli
    n_old = len(agg[agg['type'] == 'Old'])
    if 0 < n_old < len(agg):
        plt.axvline(x=n_old - 0.5, color='black', linestyle='-', linewidth=1.5)

    plt.axhline(0.5, ls='--', color='gray', label='Chance')
    plt.title("Model Probability of Category A by Stimulus")
    plt.xlabel("Stimulus ID")
    plt.ylabel("Probability of Category A")
    plt.xticks(rotation=0)
    plt.legend(title=None)
    plt.tight_layout()
    
    out_plot = results_path.parent / "plot_medin_probabilities.png"
    plt.savefig(out_plot)
    print(f"Saved probability plot to: {out_plot}")
    plt.close()

    # --- Plot 2: Classification Accuracy (Old vs New) ---
    # Overall accuracy
    # Just need aggregation by Type
    agg_type = df.groupby(['type']).agg({'accuracy': ['mean', 'sem']}).reset_index()
    agg_type.columns = ['type', 'mean_acc', 'sem_acc']

    plt.figure(figsize=(6, 6))
    sns.barplot(data=agg_type, x='type', y='mean_acc', palette="viridis", order=['Old', 'New'])
    plt.errorbar(x=range(len(agg_type)), y=agg_type['mean_acc'], yerr=agg_type['sem_acc'], fmt='none', c='black', capsize=5)
    
    plt.ylim(0, 1)
    plt.axhline(0.5, ls='--', color='gray', label='Chance')
    plt.title("Classification Accuracy: Training vs Transfer")
    plt.ylabel("Accuracy")
    plt.xlabel("Item Type")
    
    out_acc = results_path.parent / "plot_medin_accuracy.png"
    plt.savefig(out_acc)
    print(f"Saved accuracy plot to: {out_acc}")
    plt.close()

    # --- Plot 3: Probability vs Distance from Prototype ---
    # Calculate distance and prob_correct for aggregated data
    agg['distance'] = agg.apply(lambda row: get_distance(row['stimulus_id'], row['correct_cat']), axis=1)
    
    # Probability of Correct Category
    # p_A_mean is Prob(A). 
    # If correct is A: prob = p_A_mean
    # If correct is B: prob = 1 - p_A_mean
    agg['prob_correct'] = agg.apply(lambda row: row['p_A_mean'] if row['correct_cat'] == 'A' else 1.0 - row['p_A_mean'], axis=1)

    plt.figure(figsize=(10, 8))
    sns.set_context("talk")
    sns.set_style("whitegrid")
    
    # Scatter plot
    # x: distance, y: prob_correct
    # hue: correct_cat (A/B)
    # style: type (Old/New) -> markers
    
    # We want circles for Old, squares for New
    # Note: 'Old' and 'New' are the values in column 'type'
    markers = {"Old": "o", "New": "s"}
    
    sns.scatterplot(
        data=agg, 
        x='distance', 
        y='prob_correct', 
        hue='correct_cat', 
        style='type',
        markers=markers,
        s=200, # size
        alpha=0.8,
        palette={'A': 'skyblue', 'B': 'salmon'}
    )
    
    plt.title("Probability of Correct Choice vs Distance from Prototype")
    plt.xlabel("Hamming Distance from Category Prototype")
    plt.ylabel("Probability of Correct Choice")
    plt.ylim(0, 1)
    plt.axhline(0.5, ls='--', color='gray', label='Chance')
    
    # Custom legend: Remove titles
    h, l = plt.gca().get_legend_handles_labels()
    new_h, new_l = [], []
    for handler, label in zip(h, l):
        if label not in ['correct_cat', 'type']:
            new_h.append(handler)
            new_l.append(label)
    plt.legend(new_h, new_l, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    plt.tight_layout()
    
    out_dist = results_path.parent / "plot_medin_distance.png"
    plt.savefig(out_dist)
    print(f"Saved distance plot to: {out_dist}")
    plt.close()

    # --- Plot 4: Human Ratings vs Model Probability ---
    # Map ratings
    # If a stimulus is missing from HUMAN_RATINGS, this will produce NaN and be skipped/warned
    agg['human_rating'] = agg['stimulus_id'].map(HUMAN_RATINGS)
    
    plt.figure(figsize=(10, 8))
    sns.set_context("talk")
    sns.set_style("whitegrid")

    sns.scatterplot(
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
    
    plt.title("Model Probability vs Human Ratings")
    plt.xlabel("Human Rating (Experiment / Predicted)")
    plt.ylabel("Model Probability of Correct Choice")
    plt.ylim(0, 1)
    # plt.xlim(1, 9) # Assuming 1-9 scale based on values ~3-5.5? Let auto-scale handle it.
    plt.axhline(0.5, ls='--', color='gray', label='Chance')
    
    # Custom legend: Remove titles
    h, l = plt.gca().get_legend_handles_labels()
    new_h, new_l = [], []
    for handler, label in zip(h, l):
        if label not in ['correct_cat', 'type']:
            new_h.append(handler)
            new_l.append(label)
    plt.legend(new_h, new_l, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    plt.tight_layout()
    
    out_corr = results_path.parent / "plot_medin_correlation.png"
    plt.savefig(out_corr)
    print(f"Saved correlation plot to: {out_corr}")
    plt.close()

    # --- Plot 5: Model Probability vs Human Probability ---
    agg['human_prob'] = agg['stimulus_id'].map(HUMAN_PROBS)
    
    plt.figure(figsize=(10, 8))
    sns.set_context("talk")
    sns.set_style("whitegrid")
    
    # Scatter plot
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
    
    # Label each dot with stimulus number (e.g. "Stim4" -> "4")
    # We iterate through the proper rows
    for line in range(0, agg.shape[0]):
        # Check for NaNs just in case
        if pd.isna(agg.human_prob.iloc[line]) or pd.isna(agg.prob_correct.iloc[line]):
            continue
            
        stim_label = agg.stimulus_id.iloc[line].replace("Stim", "")
        # Add text label, slightly offset
        p.text(
            agg.human_prob.iloc[line] + 0.005, 
            agg.prob_correct.iloc[line], 
            stim_label, 
            horizontalalignment='left', 
            size='medium', 
            color='black', 
            weight='semibold'
        )

    plt.title("Model Probability vs Human Probability")
    plt.xlabel("Human Probability of Correct Choice")
    plt.ylabel("Model Probability of Correct Choice")
    
    # Do NOT fix limits as requested ("plot only the bounds that the graph encapsulates")
    # plt.ylim(0, 1) 
    
    # Custom legend: Remove titles
    h, l = plt.gca().get_legend_handles_labels()
    new_h, new_l = [], []
    for handler, label in zip(h, l):
        if label not in ['correct_cat', 'type']:
            new_h.append(handler)
            new_l.append(label)
    plt.legend(new_h, new_l, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    plt.tight_layout()
    
    out_prob_corr = results_path.parent / "plot_medin_prob_correlation.png"
    plt.savefig(out_prob_corr)
    print(f"Saved probability correlation plot to: {out_prob_corr}")
    plt.close()

if __name__ == "__main__":
    visualize_results()
