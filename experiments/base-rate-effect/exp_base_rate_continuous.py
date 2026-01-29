from cobweb.cobweb_continuous import CobwebContinuousTree
from random import seed, shuffle, gauss, choice
import numpy as np
import pandas as pd
from pathlib import Path
import json

# Category Breadth / Prototype Abstraction (Homa & Vosburgh, 1976)
# Experiment 1
#
# Investigates how category size (breadth) and training variety (Uniform vs Mixed)
# affect the abstraction of the prototype and generalization to new distortions.
#
# Stimuli: 9-dot random patterns (18 dim).
# Conditions:
#   - Uniform-Low: Training on Low distortions only.
#   - Mixed: Training on Low, Medium, High distortions.
# Categories:
#   - Size 3, 6, 9.

RANDOM_SEED = 999
DIMENSIONS = 18

def generate_prototype(dims=DIMENSIONS):
    # 9 dots * 2 coords = 18 dims. 30x30 grid (Homa & Vosburgh 1976)
    return [np.random.uniform(0, 30) for _ in range(dims)]

def generate_distortion(prototype, level_name):
    # Homa & Vosburgh (1976) Bit levels:
    # Low: 3.5 bits
    # Medium: 5.6 bits
    # High: 7.7 bits
    #
    # We map these to sigma values for Gaussian displacement.
    # Assuming proportionality to Homa & Cultice units or deriving arbitrary scale that strictly orders them.
    # Let's use:
    # Low ~ 1.0
    # Med ~ 2.0
    # High ~ 4.0
    
    if level_name == "Prototype":
        return list(prototype)
        
    sigma = 0.0
    if level_name == "Low" or level_name == "L":
        sigma = 1.0
    elif level_name == "Medium" or level_name == "M":
        sigma = 2.0
    elif level_name == "High" or level_name == "H":
        sigma = 4.0
        
    return [p + gauss(0, sigma) for p in prototype]

def encode_item(vector, category_val=None):
    feat_arr = np.array(vector, dtype=float)
    label_arr = np.zeros(3) # A, B, C
    
    if category_val is not None and category_val != -1:
        label_arr[int(category_val)] = 1.0
        
    return feat_arr, label_arr

def run_experiment():
    # Design:
    # 2 Conditions (Uniform-Low, Mixed)
    # 3 Category Sizes (3, 6, 9)
    # We need to run multiple subjects/seeds per condition.
    
    conditions = ["Uniform-Low", "Mixed"]
    cat_sizes_base = [3, 6, 9] # We will assign these to Cat 0, 1, 2 randomly or counterbalanced
    
    # Paper: 36 subjects per condition. Latin square.
    # We will simulate N seeds per condition with randomized size assignment.
    
    n_seeds = 12 # Enough to smooth
    
    rows = []
    
    for cond in conditions:
        for s in range(n_seeds):
            current_seed = RANDOM_SEED + (hash(cond) % 1000) + s * 51
            seed(current_seed)
            np.random.seed(current_seed)
            
            # --- Setup Categories ---
            # Randomly assign sizes 3, 6, 9 to indices 0, 1, 2
            sizes = list(cat_sizes_base)
            shuffle(sizes)
            cat_props_map = {i: sizes[i] for i in range(3)} # Cat Idx -> Size
            
            # Generate Prototypes
            protos = [generate_prototype() for _ in range(3)]
            
            # --- Generate Training Stimuli ---
            training_items = [] 
            old_instances_map = {0: [], 1: [], 2: []}
            
            for c_idx in range(3):
                size = cat_props_map[c_idx]
                p = protos[c_idx]
                
                levels = []
                if cond == "Uniform-Low":
                    levels = ["Low"] * size
                else: # Mixed
                    # "Category size 3 contained 1L, 1M, 1H..."
                    # "Category size 6 contained 2L, 2M, 2H..."
                    k = size // 3
                    levels = ["Low"]*k + ["Medium"]*k + ["High"]*k
                    
                for lvl in levels:
                    vec = generate_distortion(p, lvl)
                    item = {
                        "vector": vec,
                        "cat": c_idx,
                        "level": lvl,
                        "type": "Old"
                    }
                    training_items.append(item)
                    old_instances_map[c_idx].append(item)
                    
            # --- Learning Phase ---
            # "Learning was terminated once two consecutive errorless trials had occurred."
            # We will use a max_epoch cap, but stop if criterion met.
            
            model = CobwebContinuousTree(18, 3, alpha=0.001) 
            
            max_epochs = 15
            consecutive_perfect = 0
            
            for epoch in range(1, max_epochs + 1):
                shuffle(training_items)
                errors = 0
                
                for item in training_items:
                    x, y = encode_item(item["vector"], item["cat"])
                    
                    # Predict (Check correctness before update)
                    # "Each response was followed by yes-no feedback"
                    pred = model.predict(x, np.zeros(3), 100, False)
                    pred_cat = np.argmax(pred) if np.sum(pred) > 0 else -1
                    
                    if pred_cat != item["cat"]:
                        errors += 1
                        
                    # Train
                    model.ifit(x, y)
                    
                if errors == 0:
                    consecutive_perfect += 1
                else:
                    consecutive_perfect = 0
                    
                if consecutive_perfect >= 2:
                    break # Criterion reached
            
            # --- Test Phase ---
            # "39 stimuli... 9 Old, 27 New, 3 Prototypes"
            # Old: 3 per category.
            #   Uniform-Low -> 3 Low
            #   Mixed -> 1L, 1M, 1H
            # New: 9 per category (3L, 3M, 3H)
            # Proto: 1 per category
            
            test_items = []
            
            for c_idx in range(3):
                p = protos[c_idx]
                
                # 1. Old (3)
                olds = old_instances_map[c_idx]
                selected_olds = []
                if cond == "Mixed":
                    # Grab 1 of each level
                    by_lev = {"Low":[], "Medium":[], "High":[]}
                    for o in olds: by_lev[o["level"]].append(o)
                    selected_olds = [by_lev["Low"][0], by_lev["Medium"][0], by_lev["High"][0]]
                else:
                    # Grab random 3 (all Low)
                    shuffle(olds)
                    selected_olds = olds[:3]
                
                for o in selected_olds:
                    test_items.append({
                        "vector": o["vector"], "cat": c_idx, "type": "Old", "level": o["level"]
                    })
                    
                # 2. Prototype (1)
                test_items.append({
                    "vector": p, "cat": c_idx, "type": "Prototype", "level": "Prototype"
                })
                
                # 3. New (9 -> 3L, 3M, 3H)
                for _ in range(3): 
                    test_items.append({"vector": generate_distortion(p, "Low"), "cat": c_idx, "type": "New", "level": "Low"})
                for _ in range(3): 
                    test_items.append({"vector": generate_distortion(p, "Medium"), "cat": c_idx, "type": "New", "level": "Medium"})
                for _ in range(3): 
                    test_items.append({"vector": generate_distortion(p, "High"), "cat": c_idx, "type": "New", "level": "High"})
            
            shuffle(test_items)
            
            # Run Test (No Feedback)
            for item in test_items:
                x, _ = encode_item(item["vector"], None)
                pred = model.predict(x, np.zeros(3), 10, False)
                pred_cat = np.argmax(pred)
                
                is_correct = 1 if pred_cat == item["cat"] else 0
                
                # Identify Viz Group
                # We need "P", "L", "M", "H" for New items, or "Old"
                # Text asks: "new and prototype stimuli... as function of distortion level"
                # So we focus on New-L, New-M, New-H, and Prototype.
                
                stim_label = "Other"
                if item["type"] == "Prototype":
                    stim_label = "Prototype"
                elif item["type"] == "New":
                    stim_label = item["level"] # "Low", "Medium", "High"
                elif item["type"] == "Old":
                    stim_label = "Old"
                
                rows.append({
                    "seed": current_seed,
                    "condition": cond,
                    "cat_size": cat_props_map[item["cat"]], # 3, 6, 9
                    "correct": is_correct,
                    "stim_label": stim_label,
                    "type": item["type"]
                })

    # Save
    df = pd.DataFrame(rows)
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "exp_base_rate_continuous.csv"
    df.to_csv(out_csv, index=False)
    print(f"Homa & Vosburgh (1976) results saved to {out_csv}")
    
    metadata = {
        "experiment": "Homa & Vosburgh (1976) Exp 1",
        "factors": ["Condition (Uniform-Low, Mixed)", "Category Size (3, 6, 9)", "Distortion (P, L, M, H)"],
        "goal": "Category Breadth effect on abstraction."
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    run_experiment()
