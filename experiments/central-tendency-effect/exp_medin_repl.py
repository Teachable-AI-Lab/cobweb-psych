
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
STIMULI_DEFS = {
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

def generate_stimuli_mapping(subject_seed):
    np.random.seed(subject_seed)
    flips = np.random.randint(0, 2, size=4)
    mapped_stimuli = []
    
    # We iterate over keys to keep ID
    for name, data in STIMULI_DEFS.items():
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

def run_experiment():
    print("Running Medin & Schaffer (1978) Experiment 2 Replication...")
    all_results = []
    
    for s_idx in range(N_SUBJECTS):
        current_seed = RANDOM_SEED + s_idx
        seed(current_seed)
        np.random.seed(current_seed)
        
        stimuli = generate_stimuli_mapping(current_seed)
        train_items = [s for s in stimuli if s["type"] == "Train"]
        transfer_items = list(stimuli) # All items
        
        # Init Discrete Tree
        tree = CobwebDiscreteTree(alpha=1)
        
        # Category Mapping (Must be Int to Int/Float for Discrete Tree)
        cat_to_int = {"A": 1, "B": 2}

        # TRAINING
        consecutive_perfect = 0
        for run in range(1, MAX_TRAINING_RUNS + 1):
            shuffle(train_items)
            errors = 0
            for item in train_items:
                # Discrete encoding: Dict of attributes (int -> {int -> float})
                instance = {}
                for i, v in enumerate(item["vector"]):
                    instance[i] = {int(v): 1.0}
                
                # Predict
                probs = tree.predict(instance, 100, False)
                
                # Check probs for category A/B
                cat_idx = 4 # Index 4, since dims 0,1,2,3 are used.
                
                if cat_idx in probs:
                    prob_A = probs[cat_idx].get(1, 0.0) # A is 1
                    prob_B = probs[cat_idx].get(2, 0.0) # B is 2
                else:
                    prob_A = 0.5
                    prob_B = 0.5
                
                choice = "A" if prob_A > prob_B else "B"
                if choice != item["cat"]:
                    errors += 1

                # Train
                instance_with_cat = instance.copy()
                instance_with_cat[cat_idx] = {cat_to_int[item["cat"]]: 1.0}
                tree.ifit(instance_with_cat)
            
            # Criterion: If more than 1 error, repeat. (So <= 1 error is success)
            if errors <= 1:
                consecutive_perfect += 1
            else:
                consecutive_perfect = 0
                
            if consecutive_perfect >= CRITERION_RUNS:
                break
        
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
            
            # Save for Visualize
            p_correct = p_A_norm if item["cat"] == "A" else (1.0 - p_A_norm)
            row_type = "Old" if item["type"] == "Train" else "New"
            
            all_results.append({
                "stimulus_id": item["name"],
                "type": row_type,
                "correct_cat": item["cat"],
                "p_correct": p_correct,
                "choice": choice,
                "accuracy": is_correct,
                "p_A": p_A_norm
            })

    # Save
    df = pd.DataFrame(all_results)
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "exp_central_tendency_continuous.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    run_experiment()
