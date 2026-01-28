from cobweb.cobweb_discrete import CobwebDiscreteTree
from random import seed, shuffle, random
import numpy as np
import pandas as pd
from pathlib import Path

def generate_patient():
    """
    Generates a patient dictionary and label.
    Re-samples if patient has no symptoms (Null patient) as per Gluck & Bower (1988) / Method Description.
    """
    # Parameters from Method Description
    P_RARE = 0.25
    P_COMMON = 0.75
    
    # Symptom Probabilities P(S=1 | Disease)
    # S1, S2, S3, S4
    PROBS_RARE = [0.6, 0.4, 0.3, 0.2]
    # "Analogous but inverse" for Common
    PROBS_COMMON = [0.2, 0.3, 0.4, 0.6]
    
    SYMPTOMS = ["S1", "S2", "S3", "S4"]

    while True:
        # 1. Choose Disease
        if random() < P_RARE:
            label = "Rare"
            probs = PROBS_RARE
        else:
            label = "Common"
            probs = PROBS_COMMON
        
        # 2. Generate Symptoms
        patient = {}
        has_symptom = False
        for i, p_s in enumerate(probs):
            sym_name = SYMPTOMS[i]
            if random() < p_s:
                patient[sym_name] = 1
                has_symptom = True
            else:
                patient[sym_name] = 0
        
        # 3. Check for Null Patient
        if has_symptom:
            return patient, label

def encode_item(stim_dict, label=None, mappings=None):
    """
    Encode item into nested dictionary format required by CobwebDiscreteTree.
    {attr_id: {val_id: count}}
    """
    if mappings is None:
        attr_map = {}
        val_map = {}
        SYMPTOMS = ["S1", "S2", "S3", "S4"]
        
        # S1..S4
        for s in SYMPTOMS:
            attr_map[s] = len(attr_map) + 1
            val_map[s] = {0: 0, 1: 1} # Values 0 and 1
        
        attr_map["Disease"] = 0
        val_map["Disease"] = {"Rare": 0, "Common": 1}
        mappings = (attr_map, val_map)
    
    attr_map, val_map = mappings
    encoded = {}
    
    for k, v in stim_dict.items():
        if k in attr_map:
            a_id = attr_map[k]
            # Ensure v is in val_map (it should be 0 or 1)
            if v in val_map[k]:
                v_id = val_map[k][v]
                encoded[a_id] = {v_id: 1.0}
            
    if label is not None:
        a_id = attr_map["Disease"]
        v_id = val_map["Disease"][label]
        encoded[a_id] = {v_id: 1.0}
        
    return encoded, mappings

def get_true_posteriors():
    # Calculate P(Rare | Si) for each symptom analytically
    # Based on the generative model:
    # P(R) = 0.25, P(C) = 0.75
    # P(Si|R) = [0.6, 0.4, 0.3, 0.2]
    # P(Si|C) = [0.2, 0.3, 0.4, 0.6]
    
    P_RARE = 0.25
    P_COMMON = 0.75
    PROBS_RARE = [0.6, 0.4, 0.3, 0.2]
    PROBS_COMMON = [0.2, 0.3, 0.4, 0.6]
    SYMPTOMS = ["S1", "S2", "S3", "S4"]
    
    posteriors = {}
    for i, sym in enumerate(SYMPTOMS):
        p_s_r = PROBS_RARE[i]
        p_s_c = PROBS_COMMON[i]
        
        p_s = (p_s_r * P_RARE) + (p_s_c * P_COMMON)
        p_r_s = (p_s_r * P_RARE) / p_s
        posteriors[sym] = p_r_s
    return posteriors

def run():
    # Configuration
    n_seeds = 20
    n_trials = 250
    results = []
    
    # Initialize mappings
    _, mappings = encode_item({})
    attr_map, val_map = mappings
    disease_attr_id = attr_map["Disease"]
    rare_val_id = val_map["Disease"]["Rare"]
    
    true_posteriors = get_true_posteriors()
    SYMPTOMS = ["S1", "S2", "S3", "S4"]

    for s in range(n_seeds):
        seed(s)
        tree = CobwebDiscreteTree(alpha=0.5) 
        
        # Training
        for _ in range(n_trials):
            patient, label = generate_patient()
            encoded, _ = encode_item(patient, label, mappings)
            tree.fit([encoded])
            
        # Testing
        # "Of all the patients... exhibiting [symptom]..."
        # Querying with ONE symptom present. Others are unknown (missing).
        # We pass only {Si: 1} to predict.
        
        for sym in SYMPTOMS:
            # Create test instance with only that symptom
            test_patient = {sym: 1}
            encoded_test, _ = encode_item(test_patient, mappings=mappings)
            
            # Predict
            # Returns {attr_id: {val_id: prob}}
            # We want prob of Disease=Rare
            probs = tree.predict(encoded_test, 100, True)
            
            pred_p_rare = 0.0
            if disease_attr_id in probs and rare_val_id in probs[disease_attr_id]:
                pred_p_rare = probs[disease_attr_id][rare_val_id]
            
            # Record result
            results.append({
                "seed": s,
                "symptom": sym,
                "true_p_rare": true_posteriors[sym],
                "pred_p_rare": pred_p_rare
            })
            
    # Save
    df = pd.DataFrame(results)
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True, parents=True)
    df.to_csv(out_dir / "exp_probability_matching_gluck.csv", index=False)
    print(f"Done. Saved to {out_dir / 'exp_probability_matching_gluck.csv'}")

if __name__ == "__main__":
    run()