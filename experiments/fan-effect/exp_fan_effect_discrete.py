from cobweb.cobweb_discrete import CobwebDiscreteTree
from random import seed, shuffle, choice
import numpy as np
import pandas as pd
import json
from pathlib import Path

# Fan Effect (Anderson, 1974) & Reder & Ross (1983)
# Goal: Replicate the Fan Effect using Cobweb and the "Consistency" condition attenuation.
#
# Design:
# 1. Subjects (Seeds) learn facts about Characters.
# 2. Variable Fan Size (1, 3, 5).
# 3. Structure: 
#       - Each Character is assigned a "Theme" (e.g. Sports Fan, Writer).
#       - Predicates are Theme-Consistent (e.g. "plays Tennis", "watches Football").
# 4. Tasks (Conditions):
#       - Memory Condition: "Did [Person] [Action]?" (Exact Retrieval). RT ~ 1 / P(Action|Person)
#       - Category Condition: "Is [Action] likely for [Person]?" (Plausibility). RT ~ 1 / P(Theme|Person)

def generate_stimuli_reder_ross():
    """
    Generates thematic stimuli based on Anderson (1974) style Person-Location sentences
    but structured for Reder & Ross (1983) plausibility.
    
    Fan Sizes: 1, 2, 3 (Original Fan Effect).
    Stimuli: "The [Person] is in the [Location]".
    Themes: Locations are categorized (e.g. Nature vs Urban).
    """
    
    # Configuration: Fan sizes 1, 2, 3
    fan_groups = {1: 4, 2: 4, 3: 4} # 12 Persons total
    
    # Anderson (1974) style characters
    persons_pool = [
        "Hippie", "Captain", "Giant", "Earl", 
        "Lawyer", "Doctor", "Artist", "Debutante",
        "Fireman", "Waiter", "Clerk", "Teacher"
    ]
    shuffle(persons_pool)
    
    # Themes for consistency (Reder & Ross requirement)
    # To enable 'Consistency', predicates must share a Theme.
    themes = {
        "Nature": ["Park", "Forest", "Beach", "Meadow", "Lake", "Mountain", "Canyon", "River", "Garden", "Trail"],
        "Urban": ["Bank", "Office", "Store", "Station", "Church", "Library", "School", "Hotel", "Bar", "Cafe"]
    }
    
    stimuli = []
    char_idx = 0
    
    for fan, count in fan_groups.items():
        for _ in range(count):
            if char_idx >= len(persons_pool): break
            p_name = persons_pool[char_idx]
            char_idx += 1
            
            # Assign a random theme to this person
            theme_name = choice(list(themes.keys()))
            theme_locs = themes[theme_name].copy()
            shuffle(theme_locs)
            
            # Select 'fan' number of locations
            person_locs = theme_locs[:fan]
            
            for loc in person_locs:
                stimuli.append({
                    "person": p_name,
                    "predicate": f"is in the {loc}",
                    "theme": theme_name,
                    "fan_size": fan
                })
            
    return stimuli

def create_value_mapping(stimuli):
    val_map = {}
    
    all_persons = sorted(list(set(x["person"] for x in stimuli)))
    all_preds = sorted(list(set(x["predicate"] for x in stimuli)))
    all_themes = sorted(list(set(x["theme"] for x in stimuli)))
    
    val_map["persons"] = {p: i for i, p in enumerate(all_persons)}
    val_map["predicates"] = {p: i for i, p in enumerate(all_preds)}
    val_map["themes"] = {p: i for i, p in enumerate(all_themes)}
    val_map["relation"] = {"val": 0}
    
    return val_map

def encode_item(item, val_map, mask_target=None):
    """
    Encodes item.
    - mask_target = 'predicate' -> Mask Predicate (Attr 3)
    - mask_target = 'theme' -> Mask Theme (Attr 4)
    """
    encoded = {}
    
    # 1: Person
    encoded[1] = {val_map["persons"][item["person"]]: 1.0}
    
    # 2: Relation
    encoded[2] = {val_map["relation"]["val"]: 1.0}
    
    # 3: Predicate
    if mask_target != 'predicate':
        p_id = val_map["predicates"][item["predicate"]]
        encoded[3] = {p_id: 1.0}
        
    # 4: Theme
    if mask_target != 'theme':
        t_id = val_map["themes"][item["theme"]]
        encoded[4] = {t_id: 1.0}
        
    return encoded

def run_experiment_reder_ross(n_seeds=20):
    results = []
    
    for s in range(n_seeds):
        seed(s)
        np.random.seed(s)
        
        stimuli = generate_stimuli_reder_ross()
        val_map = create_value_mapping(stimuli)
        
        tree = CobwebDiscreteTree(alpha=0.25)
        
        # Train
        train_data = []
        for item in stimuli:
            train_data.append(encode_item(item, val_map)) # Fully observable
            
        epochs = 10
        for e in range(epochs):
            shuffle(train_data)
            for inst in train_data:
                tree.fit([inst])
                
        # Test
        # We test every studied fact in both conditions
        for item in stimuli:
            p_str = item["person"]
            pred_str = item["predicate"]
            theme_str = item["theme"]
            fan = item["fan_size"]
            
            # --- Memory Condition (Specific Retrieval) ---
            # Probe: Person (+ Relation). Target: Specific Predicate.
            # Query: {Person}. Predict: Predicate.
            
            q_mem = {} 
            q_mem[1] = {val_map["persons"][p_str]: 1.0}
            q_mem[2] = {val_map["relation"]["val"]: 1.0}
            
            # Note: We do NOT provide theme in query, as standard Fan Effect cue is Person only.
            
            probs_mem = tree.predict(q_mem, 100, False)
            
            # Get prob of correct predicate
            target_pred_id = val_map["predicates"][pred_str]
            p_mem = 0.0001
            if 3 in probs_mem and target_pred_id in probs_mem[3]:
                p_mem = probs_mem[3][target_pred_id]
                
            rt_mem = 1.0 / p_mem
            
            # --- Category Condition (Plausibility) ---
            # Probe: Person (+ Relation). Target: Theme.
            # Query: {Person}. Predict: Theme.
            
            q_cat = {}
            q_cat[1] = {val_map["persons"][p_str]: 1.0}
            q_cat[2] = {val_map["relation"]["val"]: 1.0}
             
            probs_cat = tree.predict(q_cat, 100, False)
            
            target_theme_id = val_map["themes"][theme_str]
            p_cat = 0.0001
            if 4 in probs_cat and target_theme_id in probs_cat[4]:
                p_cat = probs_cat[4][target_theme_id]
                
            rt_cat = 1.0 / p_cat
            
            results.append({
                "seed": s,
                "fan_size": fan,
                "condition": "Memory",
                "rt": rt_mem
            })
            results.append({
                "seed": s,
                "fan_size": fan,
                "condition": "Category",
                "rt": rt_cat
            })
            
    # Save
    df = pd.DataFrame(results)
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "exp_fan_effect_discrete.csv", index=False)
    print("Done.")

if __name__ == "__main__":
    run_experiment_reder_ross()
