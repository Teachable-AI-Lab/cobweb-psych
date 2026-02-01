from cobweb.cobweb_discrete import CobwebDiscreteTree
from random import seed, shuffle, choice
import numpy as np
import pandas as pd
from pathlib import Path
from math import exp

# Fan Effect (Anderson, 1974) & Reder & Ross (1983)
# Replicating using data from Table 2.

target_stimuli = [
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
]

related_foil_stimuli = [
        # --- Related Foils (Split 1) ---
        # Same Theme, Different Predicate (Mixed from other persons of same theme)
        (0, 0, 3, 1), (0, 0, 4, 1), (0, 0, 5, 1), # P0 (Fan 3)
        (1, 0, 0, 1), (1, 0, 1, 1),               # P1 (Fan 2)
        (2, 0, 2, 1),                             # P2 (Fan 1)
        
        (3, 1, 9, 1), (3, 1, 10, 1), (3, 1, 11, 1), # P3 (Fan 3)
        (4, 1, 6, 1), (4, 1, 7, 1),                 # P4 (Fan 2)
        (5, 1, 8, 1)                               # P5 (Fan 1)
]

unrelated_foil_stimuli = [
        # --- Unrelated Foils (Split 1) ---
        # Different Theme (Predicates from opposite theme)
        (0, 1, 9, 1), (0, 1, 10, 1), (0, 1, 11, 1), # P0 (Fan 3)
        (1, 1, 6, 1), (1, 1, 7, 1),                 # P1 (Fan 2)
        (2, 1, 8, 1),                               # P2 (Fan 1)
        
        (3, 0, 3, 1), (3, 0, 4, 1), (3, 0, 5, 1),   # P3 (Fan 3)
        (4, 0, 0, 1), (4, 0, 1, 1),                 # P4 (Fan 2)
        (5, 0, 2, 1)                                # P5 (Fan 1)
    ]
        

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
    p, t, pr, s = item
    encoded = {}
    encoded[1] = {p: 1.0}
    encoded[2] = {t: 1.0}
    encoded[3] = {pr: 1.0}
    encoded[4] = {s: 1.0}
    return encoded

def run_experiment_reder_ross(n_seeds=100):
    fan_map = get_fan_logic()
    results = []

    training_items = [x for x in target_stimuli]
    
    person_theme_map = {0:0, 1:0, 2:0, 3:1, 4:1, 5:1}

    for s in range(n_seeds):
        seed(s)
        np.random.seed(s)
        
        # Test on foils
        for cond, test_stimuli in [('Memory', related_foil_stimuli),
                              ('Categorization', unrelated_foil_stimuli)]:

            if cond == "Memory":
                tree = CobwebDiscreteTree(alpha=0.001)
            else:
                tree = CobwebDiscreteTree(alpha=0.1)

            # # let it know that seen and unseen both exist.
            # for _ in range(1):
            #     for l in range(2):
            #         for i in range(5):
            #             tree.ifit({1: {i: 1.0}, 4:{l: 1.0}})

            #         for i in range(2):
            #             tree.ifit({2: {i: 1.0}, 4:{l: 1.0}})

            #         for i in range(11):
            #             tree.ifit({3: {i: 1.0}, 4:{l: 1.0}})

            #         for i in range(2):
            #             tree.ifit({4: {i: 1.0}, 4:{l: 1.0}})
            
            # Training
            # Shuffle presentation order
            train_shuffled = training_items.copy() + test_stimuli.copy()
            epochs = 1 
            
            for e in range(epochs):
                shuffle(train_shuffled)
                for item in train_shuffled:
                    # Pass features 1, 2, 3 AND 4.
                    inst = encode_item(item)
                    tree.ifit(inst)

            # Testing
            for item_type, test_items in [('Foil', test_stimuli), ('Target', target_stimuli)]:
                for item in test_items:
                    fan = fan_map[item[0]]
                    query = encode_item(item)

                    # remove label when testing
                    del query[4]

                    if cond == "Memory":
                        pred = exp(tree.log_prob(query, 100, False))
                        # pred = tree.predict(query, 30, False)[4][item[4]]
                    else:
                        pred = tree.predict(query, 100, False)[4][item[3]]
                        # pred = tree.predict(query, 30, False)[2][1]

                    results.append({
                        "seed": s,
                        "fan_size": fan,
                        "condition": cond,
                        "type": item_type,
                        "pred": pred,
                        # 'loglikelihood': ll
                    })
            
    # Save
    df = pd.DataFrame(results)
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "exp_fan_effect_discrete.csv", index=False)
    print("Done generating fan effect data.")

if __name__ == "__main__":
    run_experiment_reder_ross()
