from cobweb.cobweb_discrete import CobwebDiscreteTree
from random import seed, shuffle, random, choice
import numpy as np
import pandas as pd
from pathlib import Path
import json

# Hayes-Roth & Hayes-Roth (1977) - Specific Instance Effect
# 
# Replicates the "Concept learning and the recognition and classification of exemplars"
# study using the specific stimuli list from Table 1.
#
# Procedure:
# 1. Training: 132 trials.
#    - Exemplars: Defining features (Age, Edu, Marital) + Extraneous (Name, Hobby).
#    - Classes: Club 1, Club 2, Neither.
#    - 'Either' category items get random 50/50 feedback (Club 1 or 2).
# 
# 2. Recognition Test:
#    - Probe: "ANYONE" + 3 Defining Features. (No Name/Hobby).
#    - Task: Old/New + Confidence.
#    - Model: Log Likelihood of features.
#
# 3. Final Classification:
#    - Probe: "ANYONE" + 3 Defining Features.
#    - Task: Club 1 or Club 2.
#    - Model: Probability of Class 1 vs 2.

def generate_stimuli_table_1():
    """
    Returns the definition of stimuli from Table 1.
    Format: Code (e.g. '112'), True Class, Frequency.
    """
    dataset = [
        # Club 1 (Freq 10)
        ('112', '1', 10), ('121', '1', 10), ('211', '1', 10),
        # Club 1 (Freq 1)
        ('113', '1', 1), ('131', '1', 1), ('311', '1', 1), 
        ('133', '1', 1), ('313', '1', 1), ('331', '1', 1),
        
        # Club 2 (Freq 10)
        ('221', '2', 10), ('212', '2', 10), ('122', '2', 10),
        # Club 2 (Freq 1)
        ('223', '2', 1), ('232', '2', 1), ('322', '2', 1),
        ('233', '2', 1), ('323', '2', 1), ('332', '2', 1),
        
        # Either (Freq 10)
        ('132', 'Either', 10), ('321', 'Either', 10), ('213', 'Either', 10),
        # Either (Freq 0 - Test Only)
        ('231', 'Either', 0), ('123', 'Either', 0), ('312', 'Either', 0),
        
        # Prototype 1 (Freq 0)
        ('111', '1', 0),
        # Prototype 2 (Freq 0)
        ('222', '2', 0),
        # Prototype 3 (Freq 0)
        ('333', 'Either', 0),
        
        # Neither (Freq 0)
        ('444', 'Neither', 0),
        
        # Neither (Freq 1)
        ('411', 'Neither', 1), ('422', 'Neither', 1), ('141', 'Neither', 1),
        ('242', 'Neither', 1), ('114', 'Neither', 1), ('224', 'Neither', 1),
        ('441', 'Neither', 1), ('442', 'Neither', 1), ('144', 'Neither', 1),
        ('244', 'Neither', 1), ('414', 'Neither', 1), ('424', 'Neither', 1),
        ('134', 'Neither', 1), ('234', 'Neither', 1), ('413', 'Neither', 1),
        ('423', 'Neither', 1), ('341', 'Neither', 1), ('342', 'Neither', 1),
        ('124', 'Neither', 1), ('214', 'Neither', 1), ('412', 'Neither', 1),
        ('421', 'Neither', 1), ('241', 'Neither', 1), ('142', 'Neither', 1),
        ('143', 'Neither', 1), ('243', 'Neither', 1), ('314', 'Neither', 1),
        ('324', 'Neither', 1), ('431', 'Neither', 1), ('432', 'Neither', 1)
    ]
    return dataset

def decode_features(code):
    return [int(c) for c in code]

def get_extraneous_features(rng):
    """
    Generates random Name (1-100) and Hobby (1-100) to act as variable context.
    """
    return [rng.randint(1, 100), rng.randint(1, 100)]

def create_value_mapping():
    val_map = {}
    
    # Defining Dims 1-3
    for i in range(1, 4):
        val_map[f"dim_{i}"] = {v: v for v in range(1, 5)} # 1,2,3,4
        
    # Extraneous Dims 4-5
    val_map["name"] = {v: v for v in range(1, 101)}
    val_map["hobby"] = {v: v for v in range(1, 101)}
    
    # Class
    val_map["class"] = {"1": 1, "2": 2, "Neither": 3}
    
    return val_map

def encode_training_instance(features, extraneous, label, val_map):
    encoded = {}
    
    # Defining
    for i, val in enumerate(features):
        attr_id = i + 1
        encoded[attr_id] = {val: 1.0}
        
    # Extraneous (Attr 4, 5)
    encoded[4] = {extraneous[0]: 1.0}
    encoded[5] = {extraneous[1]: 1.0}
    
    # Class (Attr 6)
    c_id = val_map["class"][label]
    encoded[6] = {c_id: 1.0}
    
    return encoded

def encode_test_probe(features):
    """
    "ANYONE" + 3 features. No extraneous. No class.
    """
    encoded = {}
    for i, val in enumerate(features):
        attr_id = i + 1
        encoded[attr_id] = {val: 1.0}
    return encoded

def run_hayes_roth_simulation(n_subjects=20):
    dataset_def = generate_stimuli_table_1()
    val_map = create_value_mapping()
    
    results = []
    
    for s in range(n_subjects):
        seed(s)
        np.random.seed(s)
        rng = np.random.RandomState(s)
        
        # 1. Construct Training List (132 exemplars)
        training_list = []
        
        for code, club, freq in dataset_def:
            if freq == 0: continue
            
            feat = decode_features(code)
            
            for _ in range(freq):
                # Unique extraneous context per trial
                extra = get_extraneous_features(rng)
                
                # Assign Feedback Label
                # If 'Either', random feedback
                assigned_label = club
                if club == "Either":
                    assigned_label = "1" if rng.rand() < 0.5 else "2"
                
                training_list.append({
                    "features": feat,
                    "extra": extra,
                    "label": assigned_label, 
                    "true_type": club,
                    "code": code
                })
        
        # Shuffle Training Order
        shuffle(training_list)
        
        # Train Model
        tree = CobwebDiscreteTree(alpha=0.25, weight_attr=True)
        
        for trial in training_list:
            encoded = encode_training_instance(trial["features"], trial["extra"], trial["label"], val_map)
            tree.fit([encoded])
            
        # 2. Recognition Task & Final Classification Task
        # Test ALL items in Table 1 (Freq 0, 1, and 10).
        
        for code, club, freq in dataset_def:
            features = decode_features(code)
            probe = encode_test_probe(features)
            
            # --- Recognition (Old/New) ---
            # Modeled via Log Likelihood
            recog_score = tree.log_prob(probe, 10, False) # Using partial matching
            
            # --- Classification (Club 1 or 2) ---
            # We predict Class attr (Attr 6).
            preds = tree.predict(probe, 100, False)
            
            p1 = 0.0
            p2 = 0.0
            
            if 6 in preds:
                if 1 in preds[6]: p1 = preds[6][1]
                if 2 in preds[6]: p2 = preds[6][2]
                
            # Normalize to 1 vs 2 (ignore Neither mass)
            total = p1 + p2
            if total > 0:
                p1_norm = p1 / total
            else:
                p1_norm = 0.5 
                
            # Tag Item Type for Reporting
            item_tag = "Other"
            if code == '111': item_tag = "Prototype"
            elif club == '1':
                if freq == 10: item_tag = "Freq_Exemplar" 
                elif freq == 1: item_tag = "Rare_Exemplar"
                
            results.append({
                "seed": s,
                "code": code,
                "club": club,
                "freq": freq,
                "recog_score": recog_score,
                "p_club1": p1_norm,
                "tag": item_tag
            })
            
    # Save Results
    df = pd.DataFrame(results)
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "exp_specific_instance_discrete.csv", index=False)
    print("Saved specific instance results.")

if __name__ == "__main__":
    run_hayes_roth_simulation()
