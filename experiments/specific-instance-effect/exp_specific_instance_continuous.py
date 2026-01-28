from cobweb.cobweb_continuous import CobwebContinuousTree
from random import seed, shuffle
import numpy as np
import pandas as pd
from pathlib import Path
import json

# Specific-Instance / Frequency Bias Effect (Nosofsky, 1988)
#
# Reference: Nosofsky, R. M. (1988). Similarity, frequency, and category 
#            representations. Journal of Experimental Psychology: Learning, 
#            Memory, and Cognition, 14(1), 54-65.
#
# Goal: Demonstrate that increasing the presentation frequency of a specific exemplar
#       enhances classification probability for that exemplar and its local neighborhood,
#       even when controlling for distance to the category prototype.

def generate_nosofsky_1988_stimuli_structure():
    """
    Generates stimuli for a 2-Category classification task.
    Structure aims to control for 'Typicality' (distance to centroid) while varying 'Frequency'.
    
    Category A: Centroid at (2.0, 2.0).
        - A_Freq: (1.0, 2.0). Distance 1.0 from Centroid. Frequency: HIGH.
        - A_Rare: (3.0, 2.0). Distance 1.0 from Centroid. Frequency: LOW.
        - A_Context: Other points to form the cluster.
    
    Category B: Centroid at (6.0, 6.0). 
        - Distant contrast category.
    """
    
    # Category A (Cluster around 2,2)
    # A_Freq and A_Rare are equidistant from mean (2,2)
    cat_A_stimuli = [
        {"id": "A_Freq", "coords": (1.0, 2.0), "type": "frequent"},
        {"id": "A_Rare", "coords": (3.0, 2.0), "type": "rare"},
        
        # Context items to stable the concept A
        {"id": "A_Context_1", "coords": (2.0, 1.0), "type": "rare"}, 
        {"id": "A_Context_2", "coords": (2.0, 3.0), "type": "rare"},
        {"id": "A_Context_3", "coords": (2.0, 2.0), "type": "rare"}, # Prototype
    ]
    
    # Category B (Cluster around 6,6)
    cat_B_stimuli = [
        {"id": "B_1", "coords": (5.0, 6.0), "type": "rare"},
        {"id": "B_2", "coords": (6.0, 5.0), "type": "rare"},
        {"id": "B_3", "coords": (6.0, 6.0), "type": "rare"},
        {"id": "B_4", "coords": (7.0, 6.0), "type": "rare"},
        {"id": "B_5", "coords": (6.0, 7.0), "type": "rare"},
    ]
    
    all_stimuli = []
    for s in cat_A_stimuli:
        s["category"] = "A"
        all_stimuli.append(s)
    for s in cat_B_stimuli:
        s["category"] = "B"
        all_stimuli.append(s)
        
    return all_stimuli

def encode_stimulus_for_continuous_tree(stimulus):
    """
    Encodes stimulus into (Feature_Vector, Label_Vector).
    Feature Vector: [x, y]
    Label Vector: [1, 0] for A, [0, 1] for B.
    """
    x_vec = np.array([stimulus["coords"][0], stimulus["coords"][1]])
    
    label_vec = np.zeros(2)
    if stimulus["category"] == "A":
        label_vec[0] = 1.0
    else:
        label_vec[1] = 1.0
        
    return x_vec, label_vec

def execute_specific_instance_simulation_nosofsky(number_of_seeds=20):
    """
    Runs the simulation.
    Key Manipulation: 'A_Freq' is presented 5x more often than 'A_Rare'.
    Observation: Classification probability P(A) should be higher for A_Freq.
    """
    results = []
    
    freq_multiplier = 5
    
    # Dataset definition
    base_stimuli = generate_nosofsky_1988_stimuli_structure()
    
    for s in range(number_of_seeds):
        seed(s)
        np.random.seed(s)
        
        # Construct Training Set with Frequency Manipulation
        training_batch = []
        for item in base_stimuli:
            count = freq_multiplier if item["type"] == "frequent" else 1
            for _ in range(count):
                training_batch.append(item)
                
        # Initialize Continuous Tree
        # Dims: 2 (x,y), Classes: 2 (A,B)
        tree = CobwebContinuousTree(2, 2, alpha=0.45) 
        
        # Train (Multiple Blocks)
        epochs = 5
        for e in range(epochs):
            shuffle(training_batch)
            for item in training_batch:
                x, y = encode_stimulus_for_continuous_tree(item)
                tree.ifit(x, y)
                
        # Test (on distinctive exemplars)
        # We test specifically on A_Freq and A_Rare
        test_targets = [x for x in base_stimuli if x["id"] in ["A_Freq", "A_Rare"]]
        
        for t in test_targets:
            x_query, _ = encode_stimulus_for_continuous_tree(t)
            
            # Predict
            # Args: (instance, label_buffer, num_nodes_to_search, greedy)
            prediction = tree.predict(x_query, np.zeros(2), 200, False)
            
            prob_A = prediction[0]
            
            # Distance check (Sanity)
            dist_to_mean = ((t["coords"][0]-2.0)**2 + (t["coords"][1]-2.0)**2)**0.5
            
            results.append({
                "seed": s,
                "stimulus_id": t["id"],
                "frequency_condition": t["type"], # frequent vs rare
                "distance_to_prototype": dist_to_mean,
                "prob_class_A": prob_A
            })
            
    return results

def run_and_save_specific_instance_results():
    results = execute_specific_instance_simulation_nosofsky(number_of_seeds=30)
    
    df = pd.DataFrame(results)
    
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp_specific_instance_continuous.csv"
    
    df.to_csv(out_path, index=False)
    print(f"Specific Instance Effect Results saved to {out_path}")
    
    # Metadata
    metadata = {
        "experiment": "Specific Instance Effect (Nosofsky 1988)",
        "hypothesis": "High frequency exemplar (A_Freq) has higher P(A) than equidistant rare exemplar (A_Rare)."
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    run_and_save_specific_instance_results()
