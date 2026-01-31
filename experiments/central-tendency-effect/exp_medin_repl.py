
import os
import sys
import numpy as np
import pandas as pd
from random import seed, shuffle
import matplotlib.pyplot as plt
import seaborn as sns

from cobweb.cobweb_discrete import CobwebDiscreteTree

# Constants
RANDOM_SEED = 2026
DIMENSIONS = 4 
CATEGORIES = 2 # A, B
N_SUBJECTS = 32
MAX_TRAINING_RUNS = 16 # Exp 2
CRITERION_RUNS = 1     # Exp 2

# Stimulus Definitions from Medin & Schaffer (1978) - Experiment 2
STIMULI_EXP2 = {
    # Train A (5 items)
    "Stim4":  {"vec": [1, 1, 1, 0], "cat": "A", "type": "Train"},
    "Stim7":  {"vec": [1, 0, 1, 0], "cat": "A", "type": "Train"},
    "Stim15": {"vec": [1, 0, 1, 1], "cat": "A", "type": "Train"},
    "Stim13": {"vec": [1, 1, 0, 1], "cat": "A", "type": "Train"},
    "Stim5":  {"vec": [0, 1, 1, 1], "cat": "A", "type": "Train"},
    
    # Train B (4 items)
    "Stim12": {"vec": [1, 1, 0, 0], "cat": "B", "type": "Train"},
    "Stim2":  {"vec": [0, 1, 1, 0], "cat": "B", "type": "Train"}, 
    "Stim14": {"vec": [0, 0, 0, 1], "cat": "B", "type": "Train"},
    "Stim10": {"vec": [0, 0, 0, 0], "cat": "B", "type": "Train"},

    # New Transfer (7 items)
    "Stim1":  {"vec": [1, 0, 0, 1], "cat": "A", "type": "New"}, 
    "Stim3":  {"vec": [1, 0, 0, 0], "cat": "B", "type": "New"}, 
    "Stim6":  {"vec": [1, 1, 1, 1], "cat": "A", "type": "New"}, 
    "Stim8":  {"vec": [0, 0, 1, 0], "cat": "B", "type": "New"}, 
    "Stim9":  {"vec": [0, 1, 0, 1], "cat": "A", "type": "New"}, 
    "Stim11": {"vec": [0, 0, 1, 1], "cat": "A", "type": "New"}, 
    "Stim16": {"vec": [0, 1, 0, 0], "cat": "B", "type": "New"},
}

# Stimulus Definitions from Medin & Schaffer (1978) - Experiment 1
STIMULI_EXP1 = {
    # Train A
    "Stim6": {"vec": [1, 1, 1, 1], "cat": "A", "type": "Train"}, # E=1, S=1, C=1, P=1
    "Stim7": {"vec": [1, 0, 1, 0], "cat": "A", "type": "Train"}, # E=1, S=0, C=1, P=0
    "Stim9": {"vec": [0, 1, 0, 1], "cat": "A", "type": "Train"}, # E=0, S=1, C=0, P=1

    # Train B
    "Stim10": {"vec": [0, 0, 0, 0], "cat": "B", "type": "Train"}, # E=0, S=0, C=0, P=0
    "Stim15": {"vec": [1, 0, 1, 1], "cat": "B", "type": "Train"}, # E=1, S=0, C=1, P=1
    "Stim16": {"vec": [0, 1, 0, 0], "cat": "B", "type": "Train"}, # E=0, S=1, C=0, P=0

    # Test
    "Stim5":  {"vec": [0, 1, 1, 1], "cat": "A", "type": "New"}, # E=0, S=1, C=1, P=1
    "Stim13": {"vec": [1, 1, 0, 1], "cat": "A", "type": "New"}, # E=1, S=1, C=0, P=1
    "Stim4":  {"vec": [1, 1, 1, 0], "cat": "A", "type": "New"}, # E=1, S=1, C=1, P=0
    "Stim3":  {"vec": [1, 0, 0, 0], "cat": "B", "type": "New"}, # E=1, S=0, C=0, P=0
    "Stim8":  {"vec": [0, 0, 1, 0], "cat": "B", "type": "New"}, # E=0, S=0, C=1, P=0
    "Stim14": {"vec": [0, 0, 0, 1], "cat": "B", "type": "New"}, # E=0, S=0, C=0, P=1
}

def generate_stimuli_mapping(subject_seed, stimuli_defs):
    np.random.seed(subject_seed)
    flips = np.random.randint(0, 2, size=4)
    mapped_stimuli = []
    
    # We iterate over keys to keep ID
    for name, data in stimuli_defs.items():
        base_vec = data["vec"]
        new_vec = []
        for i, val in enumerate(base_vec):
            if flips[i]:
                new_vec.append(1 - val)
            else:
                new_vec.append(val)
        mapped_stimuli.append({
            "name": name,
            "vector": new_vec,
            "cat": data["cat"],
            "type": data["type"]
        })
    return mapped_stimuli

def run_single_experiment_sim(exp_name, stimuli_defs, n_runs=16, criterion=1):
    print(f"Starting {exp_name}...")
    results = []
    errors_log = []

    for s_idx in range(N_SUBJECTS):
        current_seed = RANDOM_SEED + s_idx + (1000 if exp_name == "Exp1" else 0)
        seed(current_seed)
        np.random.seed(current_seed)
        
        stimuli = generate_stimuli_mapping(current_seed, stimuli_defs)
        train_items = [s for s in stimuli if s["type"] == "Train"]
        transfer_items = list(stimuli) # All items
        
        tree = CobwebDiscreteTree(alpha=1)
        cat_to_int = {"A": 1, "B": 2}
        subj_errors = {item["name"]: 0 for item in train_items}

        # TRAINING
        consecutive_perfect = 0
        for run in range(1, n_runs + 1):
            shuffle(train_items)
            errors = 0
            for item in train_items:
                instance = {}
                for i, v in enumerate(item["vector"]):
                    instance[i] = {int(v): 1.0}
                
                probs = tree.predict(instance, 100, False)
                cat_idx = 4 
                
                if cat_idx in probs:
                    prob_A = probs[cat_idx].get(1, 0.0)
                    prob_B = probs[cat_idx].get(2, 0.0)
                else:
                    prob_A = 0.5
                    prob_B = 0.5
                
                choice = "A" if prob_A > prob_B else "B"
                if choice != item["cat"]:
                    errors += 1
                    subj_errors[item["name"]] += 1

                instance_with_cat = instance.copy()
                instance_with_cat[cat_idx] = {cat_to_int[item["cat"]]: 1.0}
                tree.ifit(instance_with_cat)
            
            if errors <= criterion:
                consecutive_perfect += 1
            else:
                consecutive_perfect = 0
                
            if consecutive_perfect >= CRITERION_RUNS: # Using global const
                break
        
        for s_name, count in subj_errors.items():
            errors_log.append({
                "experiment": exp_name,
                "subject": s_idx,
                "stimulus_id": s_name,
                "error_count": count,
                "cat": stimuli_defs[s_name]["cat"]
            })
        
        # TRANSFER
        shuffle(transfer_items)
        for item in transfer_items:
            instance = {}
            for i, v in enumerate(item["vector"]):
                instance[i] = {int(v): 1.0}
            
            probs = tree.predict(instance, 100, False)
            cat_idx = 4
            
            if cat_idx in probs:
                prob_A = probs[cat_idx].get(1, 0.0)
                prob_B = probs[cat_idx].get(2, 0.0)
            else:
                prob_A = 0.5
                prob_B = 0.5
                
            total = prob_A + prob_B
            if total > 0:
                p_A_norm = prob_A / total
            else:
                p_A_norm = 0.5
            
            choice = "A" if prob_A > prob_B else "B"
            is_correct = 1 if choice == item["cat"] else 0
            
            p_correct = p_A_norm if item["cat"] == "A" else (1.0 - p_A_norm)
            row_type = "Old" if item["type"] == "Train" else "New"
            
            results.append({
                "experiment": exp_name,
                "subject": s_idx,
                "stimulus_id": item["name"],
                "type": row_type,
                "correct_cat": item["cat"],
                "p_correct": p_correct,
                "choice": choice,
                "accuracy": is_correct,
                "p_A": p_A_norm
            })
    return results, errors_log

def run_experiment():
    print("Running Medin & Schaffer (1978) Replication...")
    
    # Exp 2
    res2, err2 = run_single_experiment_sim("Exp2", STIMULI_EXP2, MAX_TRAINING_RUNS, 1)
    
    # Exp 1 (Assuming similar protocol: learn to criterion)
    # Medin 1978 Exp 1 also used learning to criterion.
    res1, err1 = run_single_experiment_sim("Exp1", STIMULI_EXP1, MAX_TRAINING_RUNS, 1)
    
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)

    # Save Exp 1
    pd.DataFrame(res1).to_csv(os.path.join(output_dir, "exp_medin_exp1_results.csv"), index=False)
    pd.DataFrame(err1).to_csv(os.path.join(output_dir, "exp_medin_exp1_errors.csv"), index=False)
    print("Saved Exp 1 results and errors.")

    # Save Exp 2
    pd.DataFrame(res2).to_csv(os.path.join(output_dir, "exp_medin_exp2_results.csv"), index=False)
    pd.DataFrame(err2).to_csv(os.path.join(output_dir, "exp_medin_exp2_errors.csv"), index=False)
    print("Saved Exp 2 results and errors.")

if __name__ == "__main__":
    run_experiment()
