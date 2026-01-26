from cobweb.cobweb_continuous import CobwebContinuousTree
from random import seed, shuffle, choices
import numpy as np
import pandas as pd
from pathlib import Path
import json

# Continuous specific-instance / frequency bias effect (Nosofsky 1988).
# Goal: show that over-presented exemplars bias classification locally in a continuous space.
#
# Stimuli: 12 color exemplars (Munsell chips) varying in saturation and brightness.
# Encoding: Continuous coordinates (dim1, dim2).
# Manipulation: One exemplar (C1) is presented with high frequency (e.g. 5x).
#
# Citation: Nosofsky, R. M. (1988). Similarity, frequency, and category 
#           representations. JEP: Learning, Memory, and Cognition, 14(1), 54-65.

RANDOM_SEED = 12345

# Define 12 stimuli in 2D space (Saturation, Brightness)
# We'll simulate them as points on a grid or circle. 
# Nosofsky 1988 Exp 1 schematic:
# Category A: Points 1-5? 
# Let's generate 2 clusters of points.
# Category A: Centered at (2, 2)
# Category B: Centered at (6, 6)

def generate_stimuli():
    # 6 exemplars for Category A
    # We'll make C1 the frequent one.
    cat_A_points = [
        (1.0, 2.0), (2.0, 1.0), (2.0, 2.0), # (2,2) is central?
        (2.0, 3.0), (3.0, 2.0), (3.0, 3.0)
    ]
    # 6 exemplars for Category B
    cat_B_points = [
        (5.0, 6.0), (6.0, 5.0), (6.0, 6.0),
        (6.0, 7.0), (7.0, 6.0), (7.0, 7.0)
    ]
    
    stimuli = []
    for i, p in enumerate(cat_A_points):
        stimuli.append({"id": f"A{i+1}", "coords": p, "category": "A"})
    for i, p in enumerate(cat_B_points):
        stimuli.append({"id": f"B{i+1}", "coords": p, "category": "B"})
        
    return stimuli

def encode_item(stimulus):
    return np.array([stimulus["coords"][0], stimulus["coords"][1]]), \
           np.array([1.0, 0.0] if stimulus["category"] == "A" else [0.0, 1.0])

def run():
    random_seeds = [RANDOM_SEED + i * 31 for i in range(5)]
    blocks = 10
    epochs = 4
    rows = []
    
    freq_multiplier = 5  # Frequent A1 is 5x more likely
    
    all_stimuli = generate_stimuli()
    # Frequent exemplar is A1 (index 0)
    frequent_id = "A1"
    
    for rs in random_seeds:
        seed(rs)
        np.random.seed(rs)
        
        # Build training set
        train_items = []
        for stim in all_stimuli:
            count = freq_multiplier if stim["id"] == frequent_id else 1
            for _ in range(count):
                train_items.append(stim)
                
        # Test items: All 12 exemplars
        test_items = all_stimuli
        
        for epoch in range(1, epochs + 1):
            # 2 dimensions (x, y), 2 labels (A, B)
            model = CobwebContinuousTree(2, 2, alpha=0.5)
            
            for block in range(1, blocks + 1):
                shuffle(train_items)
                for item in train_items:
                    x, y = encode_item(item)
                    model.ifit(x, y)
                    
                # Evaluate
                for stim in test_items:
                    x, _ = encode_item(stim)
                    # predict takes (features, label_buffer, num_nodes, greedy?)
                    prediction = model.predict(x, np.zeros(2), 100, False)
                    
                    prob_a = float(prediction[0])
                    prob_b = float(prediction[1])
                    
                    rows.append({
                        "seed": rs,
                        "epoch": epoch,
                        "block": block,
                        "stimulus": stim["id"],
                        "category": stim["category"],
                        "prob_A": prob_a,
                        "prob_B": prob_b,
                        "is_frequent": (stim["id"] == frequent_id)
                    })

    df = pd.DataFrame(rows)
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(results_dir / "exp_specific_instance_continuous.csv"), index=False)
    
    metadata = {
        "experiment": "specific_instance_continuous",
        "citation": "Nosofsky (1988)",
        "expected_effect": "Frequent exemplar (A1) should have higher classification accuracy/confidence than equidistant rare exemplars."
    }
    with open(results_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    run()
