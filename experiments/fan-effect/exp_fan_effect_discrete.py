from cobweb.cobweb_discrete import CobwebDiscreteTree
from random import seed, shuffle, choice
import numpy as np
import pandas as pd
from pathlib import Path

# Fan Effect (Anderson, 1974) & Reder & Ross (1983)
# Replicating using data from Table 2.

def generate_stimuli_table_2():
    """
    Returns data from Table 2: Materials Used in Simulation.
    Format: (Person, Theme, Predicate, Split)
    Features 1-3 correspond to col 1-3. Feature 4 is Split (0=Target/Train, 1=Foil/TestOnly).
    """
    raw_data = [
        # --- Targets (Split 0) ---
        # Fan 3 (Person 0)
        (0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0),
        # Fan 2 (Person 1)
        (1, 0, 3, 0), (1, 0, 4, 0),
        # Fan 1 (Person 2)
        (2, 0, 5, 0),
        
        # Fan 3 (Person 3)
        (3, 1, 6, 0), (3, 1, 7, 0), (3, 1, 8, 0),
        # Fan 2 (Person 4)
        (4, 1, 9, 0), (4, 1, 10, 0),
        # Fan 1 (Person 5)
        (5, 1, 11, 0),
        
        # --- Related Foils (Split 1) ---
        # Same Theme, Different Predicate (Mixed from other persons of same theme)
        (0, 0, 3, 1), (0, 0, 4, 1), (0, 0, 5, 1), # P0 (Fan 3)
        (1, 0, 0, 1), (1, 0, 1, 1),               # P1 (Fan 2)
        (2, 0, 2, 1),                             # P2 (Fan 1)
        
        (3, 1, 9, 1), (3, 1, 10, 1), (3, 1, 11, 1), # P3 (Fan 3)
        (4, 1, 6, 1), (4, 1, 7, 1),                 # P4 (Fan 2)
        (5, 1, 8, 1),                               # P5 (Fan 1)
        
        # --- Unrelated Foils (Split 1) ---
        # Different Theme (Predicates from opposite theme)
        (0, 1, 9, 1), (0, 1, 10, 1), (0, 1, 11, 1), # P0 (Fan 3)
        (1, 1, 6, 1), (1, 1, 7, 1),                 # P1 (Fan 2)
        (2, 1, 8, 1),                               # P2 (Fan 1)
        
        (3, 0, 3, 1), (3, 0, 4, 1), (3, 0, 5, 1),   # P3 (Fan 3)
        (4, 0, 0, 1), (4, 0, 1, 1),                 # P4 (Fan 2)
        (5, 0, 2, 1)                                # P5 (Fan 1)
    ]
    return raw_data

def get_fan_logic():
    # Helper to ID fan size by Person ID
    # 0,3 -> Fan 3
    # 1,4 -> Fan 2
    # 2,5 -> Fan 1
    return {0: 3, 3: 3, 1: 2, 4: 2, 2: 1, 5: 1}

def encode_item(item):
    """
    Item tuple: (Person, Theme, Predicate, Split)
    Encodes Features 1, 2, 3. Omit 4.
    """
    p, t, pr, _ = item
    encoded = {}
    encoded[1] = {p: 1.0}
    encoded[2] = {t: 1.0}
    encoded[3] = {pr: 1.0}
    return encoded

def run_experiment_reder_ross(n_seeds=20):
    raw_data = generate_stimuli_table_2()
    fan_map = get_fan_logic()
    results = []

    training_items = [x for x in raw_data if x[3] == 0]
    
    person_theme_map = {0:0, 1:0, 2:0, 3:1, 4:1, 5:1}

    for s in range(n_seeds):
        seed(s)
        np.random.seed(s)
        
        # Instantiate Tree
        tree = CobwebDiscreteTree(alpha=0.001)
        
        # Training
        # Shuffle presentation order
        train_shuffled = training_items.copy()
        epochs = 10 
        
        for e in range(epochs):
            shuffle(train_shuffled)
            for item in train_shuffled:
                # Omit split bit
                # Pass features 1, 2, 3
                inst = encode_item(item)
                tree.fit([inst])
                
        # Testing
        # We test on ALL items in Table 2 (Targets and Foils)
        for item in raw_data:
            p, t, pr, code = item
            fan = fan_map[p]
            
            # Determine Condition Label
            if code == 0:
                cond_label = "Target" # True Fact
            else:
                # Foil: approx check if Related or Unrelated
                expected_theme = person_theme_map[p]
                if t == expected_theme:
                    cond_label = "Related Foil"
                else:
                    cond_label = "Unrelated Foil"
            
            # --- MEMORY TASK (Recognition) ---
            # Task: "Judge whether specific sentences had been studied."
            # Foils: Related Foils (Same Theme, Different Predicate).
            # Expected Result: Standard Fan Effect (RT increases with Fan size).
            # Logic: We cue with Person & Theme to predict the specific Predicate.
            # As Fan increases, the probability mass for P(Predicate | Person, Theme) 
            # is split among more learned predicates, lowering the probability for the specific target
            # and increasing RT (1/P).
            query_mem = {1: {p: 1.0}, 2: {t: 1.0}}
            out_mem = tree.predict(query_mem, 30, False)
            
            p_mem = 0.0001
            if 3 in out_mem and pr in out_mem[3]:
                p_mem = out_mem[3][pr]
            
            rt_mem = 1.0 / p_mem if p_mem > 0 else 100.0
            
            # --- CATEGORY TASK (Plausibility) ---
            # Task: "Judge whether the sentence is like the sentences they had studied."
            # Foils: Unrelated Foils (Different Theme). Targets & Related Foils are "Plausible".
            # Expected Result: Reverse Fan Effect (RT decreases with Fan size for Targets).
            # Logic: We cue with Person to predict the Theme (checking consistency).
            # "Is this Person a [Theme] type?"
            # As Fan increases (more instances of Person X with Theme Y), the Person->Theme 
            # association is strengthened (higher counts relative to smoothing), 
            # increasing P(Theme | Person) and decreasing RT.
            query_cat = {1: {p: 1.0}}
            out_cat = tree.predict(query_cat, 30, False)
            
            p_cat = 0.0001
            if 2 in out_cat and t in out_cat[2]:
                p_cat = out_cat[2][t]
                
            rt_cat = 1.0 / p_cat if p_cat > 0 else 100.0
            
            results.append({
                "seed": s,
                "fan_size": fan,
                "type": cond_label,
                "rt_memory": rt_mem,
                "rt_category": rt_cat,
                "prob_memory": p_mem,
                "prob_category": p_cat
            })
            
    # Save
    df = pd.DataFrame(results)
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "exp_fan_effect_discrete.csv", index=False)
    print("Done generating fan effect data.")

if __name__ == "__main__":
    run_experiment_reder_ross()
