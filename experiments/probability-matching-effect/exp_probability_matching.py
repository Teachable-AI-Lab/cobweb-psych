from cobweb.cobweb_discrete import CobwebDiscreteTree
from random import seed, shuffle, random
import numpy as np
import pandas as pd
from pathlib import Path

def generate_stimuli(n_patterns=250):
    """
    Generate stimuli based on Gluck & Bower (1988), Experiment 1.
    4 Cues (S1, S2, S3, S4), Outcomes A (Common) vs B (Rare).
    
    Structure from Gluck & Bower 1988 (Approximate medical diagnosis task):
    - Cues appear independently or conditionally? 
    - Text: "The four simulated patients (patterns) had the symptom patterns 1110, 1010, 0101, 0001..." 
      Actually, in Exp 1 they often use patterns where cues have specific validities.
      
    Let's implement the core logic: 
    - 4 Binary Cues. 
    - Outcome determined probabalistically.
    - Base Rate of A vs B varies (e.g. 70/30).
    - Cues have different validities.
    
    Using a standard probabilistic configuration for Gluck & Bower:
    Base Rate P(A) vs P(B).
    Features are conditionally independent given category (Naive Bayes structure).
    
    Returns a list of dicts.
    """
    # Using generic values typical for these experiments or Exp 1
    # Exp 1: Base Rate A = 0.25 (Rare) vs B = 0.75 (Common), or 50-50, etc.
    # The prompt mentions "base rates (e.g., 70% vs 30%)".
    # And "choosing categories in proportion to learned probabilities"
    
    # Let's generate a stream where we explicitly control P(A) vs P(B).
    # And conditional cue probabilities.
    pass

def generate_patient(base_rate_A, cue_probs):
    """
    base_rate_A: Prob of Category A.
    cue_probs: {cue_id: {A: p, B: p}}
    """
    # 1. Determine Category
    if random() < base_rate_A:
        cat = "A"
    else:
        cat = "B"
        
    # 2. Determine Cues
    stimulus = {}
    for i in range(1, 5):
        c_name = f"C{i}"
        p_present = cue_probs[c_name][cat]
        val = 1 if random() < p_present else 0
        stimulus[c_name] = val
        
    return stimulus, cat

def encode_item(stim_dict, label=None, mappings=None):
    """
    Encode item into nested dictionary format required by CobwebDiscreteTree.
    {attr_id: {val_id: count}}
    
    If mappings is None, returns (encoded_item, mappings)
    """
    if mappings is None:
        # Dynamic mapping creation
        attr_map = {} # name -> id
        val_map = {} # attr_name -> {val -> id}
        # pre-populate with known keys to keep order/structure consistent
        # Features 1..4
        for i in range(1, 5):
            k = f"C{i}"
            attr_map[k] = i
            val_map[k] = {0: 0, 1: 1}
        
        # Category
        attr_map["Category"] = 0
        val_map["Category"] = {"A": 0, "B": 1}
        
        mappings = (attr_map, val_map)
    
    attr_map, val_map = mappings
    encoded = {}
    
    # Encode Cues
    for k, v in stim_dict.items():
        if k in attr_map:
            a_id = attr_map[k]
            v_id = val_map[k][v]
            encoded[a_id] = {v_id: 1.0}
            
    # Encode Label if present
    if label is not None:
        a_id = attr_map["Category"]
        v_id = val_map["Category"][label]
        encoded[a_id] = {v_id: 1.0}
        
    return encoded, mappings

def run():
    random_seeds = [123, 456, 789, 101112, 131415]
    base_rates_A = [0.1, 0.3, 0.5, 0.7, 0.9]
    cue_probs = {
        "C1": {"A": 0.60, "B": 0.40},
        "C2": {"A": 0.40, "B": 0.60},
        "C3": {"A": 0.50, "B": 0.50},
        "C4": {"A": 0.50, "B": 0.50}
    }
    blocks = 10
    trials_per_block = 25
    rows = []
    
    # Initialize mappings once
    _, mappings = encode_item({})
    attr_map, val_map = mappings
    cat_attr_id = attr_map["Category"]
    cat_val_A = val_map["Category"]["A"]

    for rs in random_seeds:
        seed(rs)
        for br_A in base_rates_A:
            model = CobwebDiscreteTree(alpha=0.5)
            
            for b in range(1, blocks + 1):
                block_data = []
                for _ in range(trials_per_block):
                    stim, label = generate_patient(br_A, cue_probs)
                    block_data.append((stim, label))
                
                n_A_responses = 0
                n_A_actual = 0
                
                for stim, actual_label in block_data:
                    # Encode stimulus for prediction (no label)
                    encoded_stim, _ = encode_item(stim, label=None, mappings=mappings)
                    
                    # Predict
                    # CobwebDiscreteTree.predict(instance, max_nodes, greedy=False)
                    probs = model.predict(encoded_stim, 100, True)
                    
                    # Extract P(Category=A)
                    prob_A = 0.0
                    if cat_attr_id in probs and cat_val_A in probs[cat_attr_id]:
                        prob_A = probs[cat_attr_id][cat_val_A]
                    
                    pred_response_A = prob_A
                    
                    # Train
                    encoded_train, _ = encode_item(stim, label=actual_label, mappings=mappings)
                    model.fit([encoded_train])
                    
                    # Record
                    n_A_responses += pred_response_A
                    if actual_label == "A":
                        n_A_actual += 1
                        
                avg_response_A = n_A_responses / trials_per_block
                actual_base_rate_block = n_A_actual / trials_per_block
                
                rows.append({
                    "seed": rs,
                    "condition_base_rate": br_A,
                    "block": b,
                    "response_prob_A": avg_response_A,
                    "actual_prop_A": actual_base_rate_block
                })

    df = pd.DataFrame(rows)
    # ...existing code... (Save logic)
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(results_dir / "exp_probability_matching.csv"), index=False)

if __name__ == "__main__":
    run()
