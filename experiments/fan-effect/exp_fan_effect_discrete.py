from cobweb.cobweb_discrete import CobwebDiscreteTree
from random import seed, shuffle, choice
import numpy as np
import pandas as pd
import json
from pathlib import Path

# Fan Effect (Anderson, 1974; Reder & Ross, 1983)
# Goal: Mimic the experimental setup where participants learn facts about characters
#       and reaction time is measured during a verification task.
#       Key constraints:
#       1. Manipulate Fan Size (facts per character).
#       2. Each predicate is studied with exactly two characters (to control object fan).
#       3. Measure retrieval probability as a proxy for Accuracy/RT.

def generate_experiment_stimuli_anderson_reder_ross():
    """
    Generates a set of stimuli (Person-Predicate pairs) following the constraints:
    - 12 Characters divided into Fan Size groups (e.g., Fan 2, Fan 4, Fan 6).
    - Predicates are shared such that each predicate appears exactly twice in the dataset.
    
    Returns:
        learning_stimuli: List of {"person": P, "predicate": O}
        fan_mapping: Dict mapping Person -> Fan Size
    """
    # 12 Characters total
    # Group 1: Fan 1 (4 chars) -> 4 facts
    # Group 2: Fan 3 (4 chars) -> 12 facts
    # Group 3: Fan 5 (4 chars) -> 20 facts
    # Total facts = 36.
    # Total unique predicates needed = 18 (since each used twice).
    
    characters = [f"Person_{i}" for i in range(1, 13)]
    # Shuffle assignment to groups
    shuffle(characters)
    
    fan_groups = {
        1: characters[0:4],
        3: characters[4:8],
        5: characters[8:12]
    }
    
    # Create pool of predicates (locations/objects)
    # We need 18 unique predicates.
    predicates = [f"Predicate_{i}" for i in range(1, 19)]
    
    # We need to assign predicates to (Person, Slot) such that every predicate is used 2 times.
    # We have 36 "slots" to fill (sum of fans).
    # We have 18 predicates * 2 copies = 36 items.
    
    predicate_pool = predicates * 2
    shuffle(predicate_pool)
    
    # Assign
    learning_stimuli = []
    fan_mapping = {}
    
    # Helper to check if person already has this predicate (prevent duplicates)
    person_facts = {c: set() for c in characters}
    
    # Simplegreedy assignment with backtrack or just retry if fail
    # Since pool is large, random assignment usually works, but we need to ensure unique predicates per person.
    
    assignments_valid = False
    attempt = 0
    max_attempts = 1000
    
    final_stimuli = []
    
    while not assignments_valid and attempt < max_attempts:
        attempt += 1
        curr_pool = predicate_pool.copy()
        shuffle(curr_pool)
        temp_facts = {c: set() for c in characters}
        possible = True
        
        for fan_size, group_chars in fan_groups.items():
            for p in group_chars:
                fan_mapping[p] = fan_size
                for _ in range(fan_size):
                    # Try to find a predicate in pool not already assigned to p
                    found = False
                    for i, pred in enumerate(curr_pool):
                        if pred not in temp_facts[p]:
                            # Valid
                            temp_facts[p].add(pred)
                            curr_pool.pop(i)
                            found = True
                            break
                    if not found:
                        possible = False
                        break
            if not possible: break
        
        if possible:
            assignments_valid = True
            # Flatten to list
            final_stimuli = []
            for p, preds in temp_facts.items():
                for pred in preds:
                    final_stimuli.append({"person": p, "predicate": pred})
                    
    if not assignments_valid:
        raise RuntimeError("Could not satisfy stimulus constraints (unique predicates per person) after retries.")
        
    return final_stimuli, fan_mapping

def generate_foil_stimuli_for_testing(learning_stimuli, all_predicates):
    """
    Generate Foil (False) facts for testing.
    Anderson (1974): Re-pair learned people with learned predicates they didn't study.
    """
    foils = []
    # Collect all persons
    persons = list(set([x["person"] for x in learning_stimuli]))
    
    # For each person, pick a predicate they definitely didn't study
    # We generate a balanced set of foils.
    
    facts_set = set([(x["person"], x["predicate"]) for x in learning_stimuli])
    
    for p in persons:
        # Generate varied foils? Or just 1 per studied fact?
        # Let's generate a fixed number of foils per person, maybe equal to their fan size?
        # Anderson design usually balances True and False probes.
        
        person_fan = len([x for x in learning_stimuli if x["person"] == p])
        
        count = 0
        shuffled_preds = all_predicates.copy()
        shuffle(shuffled_preds)
        
        for pred in shuffled_preds:
            if (p, pred) not in facts_set:
                foils.append({"person": p, "predicate": pred, "type": "foil"})
                count += 1
                if count >= person_fan: # Match number of positive probes
                    break
                    
    return foils

def create_value_mapping(stimuli):
    """
    Creates a mapping from string values to integers for Cobweb.
    """
    all_persons = sorted(list(set([x["person"] for x in stimuli])))
    all_predicates = sorted(list(set([x["predicate"] for x in stimuli])))
    
    val_map = {}
    
    # Domain: Persons (Attr 1)
    val_map["persons"] = {p: i for i, p in enumerate(all_persons)}
    
    # Domain: Relation (Attr 2)
    val_map["relation"] = {"associated_with": 0}
    
    # Domain: Predicates (Attr 3)
    val_map["predicates"] = {p: i for i, p in enumerate(all_predicates)}
    
    return val_map

def encode_stimulus_item_for_cobweb(item, val_map, mask_predicate=False):
    """
    Encodes an item {person, predicate} into Cobweb format using integer value IDs.
    """
    encoded = {}
    
    # Person Attribute (ID 1)
    p_str = item["person"]
    if p_str in val_map["persons"]:
        encoded[1] = {val_map["persons"][p_str]: 1.0}
    
    # Relation Attribute (ID 2)
    encoded[2] = {val_map["relation"]["associated_with"]: 1.0}
    
    # Predicate Attribute (ID 3)
    if not mask_predicate:
        pred_str = str(item["predicate"])
        if pred_str in val_map["predicates"]:
            encoded[3] = {val_map["predicates"][pred_str]: 1.0}
        
    return encoded

def measure_retrieval_latency_proxy(tree, person, target_predicate, val_map):
    """
    Measures the 'latency' of retrieval via probability of target predicate.
    """
    query_stim = {
        "person": person,
        "relation": "associated_with" # implicit
    }
    
    # Encode query (masking predicate)
    encoded_query = encode_stimulus_item_for_cobweb(query_stim, val_map, mask_predicate=True)
    
    # Predict
    # We want the probability distribution for Attr ID 3 (Predicate)
    probs = tree.predict(encoded_query, 100, False)
    
    prob_val = 0.0
    
    # Get prob of the specific target_predicate
    target_pred_str = str(target_predicate)
    target_pred_id = val_map["predicates"].get(target_pred_str)
    
    if target_pred_id is not None:
        if 3 in probs and target_pred_id in probs[3]:
            prob_val = probs[3][target_pred_id]
        
    return prob_val

def execute_fan_effect_simulation_experiment_anderson(number_of_seeds=30):
    """
    Main execution loop for the Fan Effect experiment.
    """
    results = []
    
    for s in range(number_of_seeds):
        seed(s)
        np.random.seed(s)
        
        # 1. Generate Stimuli
        stimuli, fan_mapping = generate_experiment_stimuli_anderson_reder_ross()
        
        # Create Mapping for this seed's stimuli
        val_map = create_value_mapping(stimuli)
        
        # 2. Train Model (Learning Phase)
        tree = CobwebDiscreteTree(alpha=0.2) 
        
        # Training loop
        training_data = stimuli.copy()
        epochs = 15 # Robust training
        
        for e in range(epochs):
            shuffle(training_data)
            for item in training_data:
                encoded = encode_stimulus_item_for_cobweb(item, val_map)
                tree.fit([encoded])
        
        # 3. Test Phase (Verification Task)
        for item in stimuli:
            p = item["person"]
            pred = item["predicate"]
            fan = fan_mapping[p]
            
            # Measure Activation
            prob_retrieval = measure_retrieval_latency_proxy(tree, p, pred, val_map)
            
            # Avoid div-by-zero
            if prob_retrieval < 0.0001: prob_retrieval = 0.0001
            
            simulated_rt = 1.0 / prob_retrieval
            
            results.append({
                "seed": s,
                "person": p,
                "fan_size": fan,
                "predicate": pred,
                "probe_type": "Target", # Studied fact
                "probability": prob_retrieval,
                "simulated_rt": simulated_rt
            })
            
    return results

def run_and_save_fan_effect_results():
    results = execute_fan_effect_simulation_experiment_anderson(number_of_seeds=30)
    
    df = pd.DataFrame(results)
    
    # Save
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp_fan_effect_discrete.csv"
    
    df.to_csv(out_path, index=False)
    print(f"Fan Effect Experiment completed. Results saved to {out_path}")
    
    # Also save metadata
    metadata = {
        "description": "Fan Effect Simulation (Anderson 1974 / Reder & Ross 1983)",
        "prediction": "RT (1/Prob) should increase with Fan Size",
        "fan_sizes_tested": sorted(list(set(df["fan_size"])))
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    run_and_save_fan_effect_results()
