from pathlib import Path
from random import seed, shuffle, choice
import pandas as pd
import numpy as np
from cobweb.cobweb_discrete import CobwebDiscreteTree

# Medin et al. (1982) Exp 1 Parameters
# 9 Learning Cases ("Burlosis")
# 4 Symptoms (binary)
# Base rate of each symptom = 6/9 (~0.67)

def generate_learning_cases_for_correlated_feature_experiment_1(condition):
    """
    Generate 9 learning cases based on condition ('correlated' or 'uncorrelated') 
    for Medin et al. (1982) Experiment 1.
    """
    if condition == 'correlated':
        # Perfectly correlated. 6 cases (1,1), 3 cases (0,0).
        f3 = [1]*6 + [0]*3
        f4 = [1]*6 + [0]*3
    else:
        # Uncorrelated but matching base rates.
        pairs = [(1,1)]*4 + [(1,0)]*2 + [(0,1)]*2 + [(0,0)]*1
        shuffle(pairs)
        f3 = [p[0] for p in pairs]
        f4 = [p[1] for p in pairs]
        
    base_col = [1]*6 + [0]*3
    
    data = []
    f1 = base_col.copy()
    f2 = base_col.copy()
    shuffle(f1)
    shuffle(f2)
    
    for i in range(9):
        item = {
            "S1": f1[i],
            "S2": f2[i],
            "S3": f3[i],
            "S4": f4[i]
        }
        data.append(item)
        
    return data

def generate_testing_pairs_for_correlated_feature_experiment_1():
    """
    Generate test pairs to distinguish between correlated (relational) vs independent processing.
    Each pair contains a Consistent item (follows correlation rule) and Inconsistent item.
    """
    pairs = []
    
    # Pair 1: 3 Symptoms Present
    p1 = (
        {"S3": 1, "S4": 1, "S1": 1, "S2": 0}, # Consistent (1,1)
        {"S3": 1, "S4": 0, "S1": 1, "S2": 1}  # Inconsistent (1,0)
    )
    pairs.append(p1)
    
    # Pair 2: 3 Symptoms Present (Alternative config)
    p2 = (
        {"S3": 1, "S4": 1, "S1": 0, "S2": 1}, # Consistent
        {"S3": 0, "S4": 1, "S1": 1, "S2": 1}  # Inconsistent
    )
    pairs.append(p2)
    
    # Pair 3: 2 Symptoms Present
    p3 = (
        {"S3": 0, "S4": 0, "S1": 1, "S2": 1}, # Consistent (0,0)
        {"S3": 1, "S4": 0, "S1": 0, "S2": 1}  # Inconsistent
    )
    pairs.append(p3)
    
    # Pair 4: 1 Symptom Present
    p4 = (
        {"S3": 0, "S4": 0, "S1": 1, "S2": 0}, # Consistent
        {"S3": 0, "S4": 1, "S1": 0, "S2": 0}  # Inconsistent
    )
    pairs.append(p4)

    return pairs

def encode_stimulus_for_cobweb_tree(stimulus):
    """
    Encode a dictionary of features into Cobweb's AV (Attribute-Value) format.
    """
    encoded = {}
    attr_map = {"S1": 1, "S2": 2, "S3": 3, "S4": 4}
    for k, v in stimulus.items():
        encoded[attr_map[k]] = {v: 1.0}
    return encoded

def calculate_instance_likelihood_score(tree, instance):
    """
    Calculate a score representing how well an instance fits the learned concept.
    We approximate this by the probability of the critical features (S3, S4) 
    given the context of the other features (S1, S2).
    """
    s1 = instance["S1"]
    s2 = instance["S2"]
    
    input_stim = {
        1: {s1: 1.0},
        2: {s2: 1.0}
    }
    
    # Predict probabilities of missing S3/S4
    probs = tree.predict(input_stim, 1000, False)
    
    s3_val = instance["S3"]
    s4_val = instance["S4"]
    
    p_s3 = 0.0
    if 3 in probs and s3_val in probs[3]:
        p_s3 = probs[3][s3_val]
        
    p_s4 = 0.0
    if 4 in probs and s4_val in probs[4]:
        p_s4 = probs[4][s4_val]
        
    return p_s3 * p_s4

def execute_correlated_feature_simulation_experiment_1_and_2(number_of_seeds=50):
    """
    Run simulation for Experiment 1 (and 2 since methodology matches).
    Iterates through 'correlated' and 'uncorrelated' training conditions.
    """
    results = []
    
    for cond in ["correlated", "uncorrelated"]:
        for s in range(number_of_seeds):
            seed(s) 
            np.random.seed(s)
            
            # 1. Train
            tree = CobwebDiscreteTree(alpha=0.5) 
            training_data = generate_learning_cases_for_correlated_feature_experiment_1(cond)
            
            params = training_data.copy()
            for _ in range(5):
                shuffle(params)
                for item in params:
                    encoded = encode_stimulus_for_cobweb_tree(item)
                    tree.fit([encoded])
            
            # 2. Test
            test_pairs = generate_testing_pairs_for_correlated_feature_experiment_1()
            n_consistent_chosen = 0
            
            for cons, incons in test_pairs:
                score_c = calculate_instance_likelihood_score(tree, cons)
                score_i = calculate_instance_likelihood_score(tree, incons)
                
                if score_c > score_i:
                    n_consistent_chosen += 1
                elif score_c == score_i:
                    if choice([0, 1]) == 0: n_consistent_chosen += 1
            
            prop = n_consistent_chosen / len(test_pairs)
            
            results.append({
                "condition": cond,
                "seed": s,
                "prop_consistent": prop
            })
    return results

def generate_learning_cases_for_correlated_feature_experiment_3():
    """
    Generate 9 learning cases for Experiment 3.
    5 Dimensions (S1-S5). S4 and S5 are perfectly correlated.
    """
    # S4 and S5: 6 cases (1,1), 3 cases (0,0)
    s4 = [1]*6 + [0]*3
    s5 = [1]*6 + [0]*3
    
    base = [1]*6 + [0]*3
    s1 = base.copy()
    s2 = base.copy()
    s3 = base.copy()
    
    shuffle(s1)
    shuffle(s2)
    shuffle(s3)
    
    data = []
    for i in range(9):
        item = {
            "S1": s1[i],
            "S2": s2[i],
            "S3": s3[i],
            "S4": s4[i],
            "S5": s5[i]
        }
        data.append(item)
    return data

def generate_testing_pairs_for_correlated_feature_experiment_3_strategy_shift():
    """
    Generate test pairs for Exp 3.
    Returns Control pairs (Typically vs Atypical) and Conflict pairs (Correlation vs Typicality).
    """
    tests = []
    
    # --- Control Pairs ---
    tests.append({
        "type": "control",
        "A": {"S1":1, "S2":1, "S3":1, "S4":1, "S5":0}, 
        "B": {"S1":0, "S2":0, "S3":0, "S4":1, "S5":0}  
    })
    tests.append({
        "type": "control",
        "A": {"S1":1, "S2":1, "S3":1, "S4":0, "S5":1}, 
        "B": {"S1":1, "S2":0, "S3":0, "S4":0, "S5":1}  
    })
    tests.append({
        "type": "control",
        "A": {"S1":1, "S2":1, "S3":0, "S4":1, "S5":0}, 
        "B": {"S1":0, "S2":0, "S3":1, "S4":1, "S5":0}
    })
    tests.append({
        "type": "control",
        "A": {"S1":1, "S2":0, "S3":1, "S4":0, "S5":1}, 
        "B": {"S1":0, "S2":1, "S3":0, "S4":0, "S5":1}
    })

    # --- Conflict Pairs ---
    tests.append({
        "type": "conflict",
        "A": {"S1":0, "S2":0, "S3":0, "S4":1, "S5":1}, 
        "B": {"S1":1, "S2":1, "S3":1, "S4":1, "S5":0}  
    })
    tests.append({
        "type": "conflict",
        "A": {"S1":0, "S2":0, "S3":1, "S4":0, "S5":0}, 
        "B": {"S1":1, "S2":1, "S3":0, "S4":0, "S5":1}  
    })
    tests.append({
        "type": "conflict",
        "A": {"S1":0, "S2":0, "S3":0, "S4":1, "S5":1}, 
        "B": {"S1":1, "S2":1, "S3":0, "S4":1, "S5":0}  
    })
    tests.append({
        "type": "conflict",
        "A": {"S1":0, "S2":1, "S3":0, "S4":0, "S5":0}, 
        "B": {"S1":1, "S2":0, "S3":1, "S4":0, "S5":1}  
    })
    tests.append({
        "type": "conflict",
        "A": {"S1":0, "S2":0, "S3":0, "S4":0, "S5":0}, 
        "B": {"S1":1, "S2":0, "S3":0, "S4":1, "S5":0}  
    })

    return tests

def encode_stimulus_for_experiment_3(stimulus):
    encoded = {}
    attr_map = {"S1": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5}
    for k, v in stimulus.items():
        encoded[attr_map[k]] = {v: 1.0}
    return encoded

def calculate_instance_typicality_score_experiment_3(tree, instance):
    """
    Score instance typicality for Exp 3 by calculating the joint probability
    of features given the model's prediction.
    """
    encoded = encode_stimulus_for_experiment_3(instance)
    probs = tree.predict(encoded, 10, True)
    
    score = 1.0
    attr_map = {"S1": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5}
    
    for k, v in instance.items():
        aid = attr_map[k]
        if aid in probs and v in probs[aid]:
            score *= probs[aid][v]
        else:
            score *= 0.001 
            
    return score

def execute_correlated_feature_simulation_experiment_3(number_of_seeds=30):
    results = []
    
    for s in range(number_of_seeds):
        seed(s) 
        np.random.seed(s)
        
        # 1. Train
        tree = CobwebDiscreteTree(alpha=0.5) 
        training_data = generate_learning_cases_for_correlated_feature_experiment_3()
        
        params = training_data.copy()
        for _ in range(5):
            shuffle(params)
            for item in params:
                encoded = encode_stimulus_for_experiment_3(item)
                tree.fit([encoded])
        
        # 2. Test
        tests = generate_testing_pairs_for_correlated_feature_experiment_3_strategy_shift()
        
        n_high_typ_chosen_control = 0
        n_control_trials = 0
        
        n_corr_chosen_conflict = 0
        n_conflict_trials = 0
        
        for t in tests:
            score_a = calculate_instance_typicality_score_experiment_3(tree, t["A"])
            score_b = calculate_instance_typicality_score_experiment_3(tree, t["B"])
            
            # Choice Logic
            choice_is_a = False
            if score_a > score_b:
                choice_is_a = True
            elif score_a == score_b:
                if choice([0, 1]) == 0: choice_is_a = True
            
            if t["type"] == "control":
                n_control_trials += 1
                if choice_is_a:
                    n_high_typ_chosen_control += 1
                    
            elif t["type"] == "conflict":
                n_conflict_trials += 1
                if choice_is_a:
                    n_corr_chosen_conflict += 1

        results.append({
            "seed": s,
            "prop_high_typ_control": n_high_typ_chosen_control / n_control_trials if n_control_trials > 0 else 0,
            "prop_corr_conflict": n_corr_chosen_conflict / n_conflict_trials if n_conflict_trials > 0 else 0
        })
        
    return results

def get_abstract_stimulus_structure_for_experiment_4_classification():
    """
    Exp 4: Classification (Terrigitis vs Midosis).
    Returns abstract patterns (tuples) and their correct category labels.
    """
    # Terrigitis patterns (Target)
    t_patterns_abstract = [
        (1,1,1,1),
        (1,1,0,0),
        (0,1,1,1),
        (1,0,0,0)
    ]
    
    # Midosis patterns (Reference)
    m_patterns_abstract = [
        (0,0,1,0),
        (0,0,0,1),
        (1,0,1,0),
        (0,1,0,1)
    ]
    
    train = []
    for p in t_patterns_abstract:
        train.append((p, "Terrigitis"))
    for p in m_patterns_abstract:
        train.append((p, "Midosis"))
        
    import itertools
    test = list(itertools.product([0, 1], repeat=4))
    
    return train, test

def encode_stimulus_with_optional_category_label_experiment_4(stimulus, label=None):
    encoded = {}
    attr_map = {"S1": 1, "S2": 2, "S3": 3, "S4": 4}
    
    for k, v in stimulus.items():
        encoded[attr_map[k]] = {v: 1.0}
    
    if label:
        val = 0 if label == "Terrigitis" else 1
        encoded[0] = {val: 1.0}
        
    return encoded

def execute_correlated_feature_simulation_experiment_4_classification(number_of_seeds=30):
    results = []
    
    train_abstract, test_abstract = get_abstract_stimulus_structure_for_experiment_4_classification()
    
    for s in range(number_of_seeds):
        seed(s)
        np.random.seed(s)
        
        # Randomize Feature Mapping
        phys_feats = ["S1", "S2", "S3", "S4"]
        shuffle(phys_feats)
        
        def map_to_item(p_tuple):
            return {phys_feats[i]: p_tuple[i] for i in range(4)}

        tree = CobwebDiscreteTree(alpha=0.2)
        
        training_queue = []
        for p, label in train_abstract:
            training_queue.append((map_to_item(p), label))
            
        for _ in range(10):
            shuffle(training_queue)
            for item, label in training_queue:
                encoded = encode_stimulus_with_optional_category_label_experiment_4(item, label)
                tree.fit([encoded])
                
        for p in test_abstract:
            item = map_to_item(p)
            encoded_query = encode_stimulus_with_optional_category_label_experiment_4(item)
            probs = tree.predict(encoded_query, 1000, False)
            
            p_terrigitis = 0.0
            if 0 in probs and 0 in probs[0]:
                p_terrigitis = probs[0][0]
                
            p_str = "".join(map(str, p))
            results.append({
                "seed": s,
                "pattern": p_str,
                "p_terrigitis": p_terrigitis
            })
            
    return results

def execute_all_correlated_feature_experiments():
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True, parents=True)

    # --- Experiment 1 ---
    print("Running Experiment 1 (Replication)...")
    results_exp1 = execute_correlated_feature_simulation_experiment_1_and_2(number_of_seeds=30)
    df1 = pd.DataFrame(results_exp1)
    df1.to_csv(out_dir / "exp1_correlated_feature_medin.csv", index=False)
    print(f"Exp 1 saved to {out_dir / 'exp1_correlated_feature_medin.csv'}")

    # --- Experiment 2 ---
    print("Running Experiment 2 (Dimensions)...")
    results_exp2 = execute_correlated_feature_simulation_experiment_1_and_2(number_of_seeds=30) 
    df2 = pd.DataFrame(results_exp2)
    df2.to_csv(out_dir / "exp2_correlated_feature_medin.csv", index=False)
    print(f"Exp 2 saved to {out_dir / 'exp2_correlated_feature_medin.csv'}")
    
    # --- Experiment 3 ---
    print("Running Experiment 3 (Conflict)...")
    results_exp3 = execute_correlated_feature_simulation_experiment_3(number_of_seeds=30)
    df3 = pd.DataFrame(results_exp3)
    df3.to_csv(out_dir / "exp3_correlated_feature_medin.csv", index=False)
    print(f"Exp 3 saved to {out_dir / 'exp3_correlated_feature_medin.csv'}")

    # --- Experiment 4 ---
    print("Running Experiment 4 (Classification)...")
    results_exp4 = execute_correlated_feature_simulation_experiment_4_classification(number_of_seeds=30)
    df4 = pd.DataFrame(results_exp4)
    df4.to_csv(out_dir / "exp4_correlated_feature_medin.csv", index=False)
    print(f"Exp 4 saved to {out_dir / 'exp4_correlated_feature_medin.csv'}")

if __name__ == "__main__":
    execute_all_correlated_feature_experiments()
