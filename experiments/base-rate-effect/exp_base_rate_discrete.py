from random import seed, shuffle
from pathlib import Path
import numpy as np
import pandas as pd
import json
from cobweb.cobweb_discrete import CobwebDiscreteTree

# Base-rate & Inverse Base-Rate Effect (Medin & Edelson, 1988)
# Goal: Reproduce inverse base-rate effect where on ambiguous BC trials, 
# participants often choose the rare category despite training base rates.
#
# Training: AB → Common (40 trials), AC → Rare (10 trials)
# Test: BC (ambiguous) - should bias toward Rare despite lower base rate
#
# Citation: Medin, D. L., & Edelson, S. M. (1988). Problem structure and the use of 
#           base-rate information from experience. JEP: General, 117(1), 68-85.

# Random seed for reproducibility
RANDOM_SEED = 12345

attribute_values = {
    "symptom_1": ["absent", "present"],
    "symptom_2": ["absent", "present"],
    "symptom_3": ["absent", "present"],
    "diagnosis": ["Common_Disease", "Rare_Disease"],
}


def make_mappings():
    """Create attribute and value ID mappings for Cobweb discrete encoding."""
    attr_ids = {name: idx for idx, name in enumerate(attribute_values.keys())}
    value_ids = {}
    for attr, vals in attribute_values.items():
        value_ids[attr] = {v: i for i, v in enumerate(vals)}
    return attr_ids, value_ids


def build_train_medin(ratio_common_rare=4.0):
    """
    Build training set following Medin & Edelson (1988) design.
    Simulated medical diagnosis task.
    
    Training patterns:
    - I + PC -> Common Disease (Frequent)
      Symptom 1 (I) Present
      Symptom 2 (PC) Present
      Symptom 3 (PR) Absent
    - I + PR -> Rare Disease (Infrequent)
      Symptom 1 (I) Present
      Symptom 2 (PC) Absent
      Symptom 3 (PR) Present
      
    (Using standard A,B,C mapping: A=I, B=PC, C=PR)
    
    Args:
        ratio_common_rare: Ratio of Common to Rare training trials (default 4:1)
    """
    n_rare = 10
    n_common = int(n_rare * ratio_common_rare)  # 40 trials
    
    items = []
    
    # Common Disease (A=I present, B=PC present, C=PR absent)
    for _ in range(n_common):
        items.append({
            "symptom_1": "present",
            "symptom_2": "present",
            "symptom_3": "absent",
            "diagnosis": "Common_Disease"
        })
    
    # Rare Disease (A=I present, B=PC absent, C=PR present)
    for _ in range(n_rare):
        items.append({
            "symptom_1": "present",
            "symptom_2": "absent",
            "symptom_3": "present",
            "diagnosis": "Rare_Disease"
        })
    
    return items


def build_test_medin():
    """
    Build test items for inverse base-rate effect.
    
    Critical test: BC (Symptom 2 + Symptom 3) - ambiguous
    """
    tests = [
        # Training items
        ({"symptom_1": "present", "symptom_2": "present", "symptom_3": "absent"}, "Common_Disease", "Train_Common"),
        ({"symptom_1": "present", "symptom_2": "absent", "symptom_3": "present"}, "Rare_Disease", "Train_Rare"),
        
        # Critical ambiguous test: BC (Symptom 2 + Symptom 3 present, Symptom 1 absent)
        # B is perfect predictor of Common (PC)
        # C is perfect predictor of Rare (PR)
        # But C (Rare) is stronger due to inverse base-rate effect
        ({"symptom_1": "absent", "symptom_2": "present", "symptom_3": "present"}, "ambiguous", "BC_critical"),
        
        # Imperfect predictors
        ({"symptom_1": "present", "symptom_2": "absent", "symptom_3": "absent"}, "Common_Disease", "I_Only"),
    ]
    return tests


def encode_items(raw_items, attr_ids, value_ids, include_category: bool = True):
    """Encode items for Cobweb discrete tree."""
    encoded = []
    for item in raw_items:
        entry = {}
        for attr in ["symptom_1", "symptom_2", "symptom_3"]:
            entry[attr_ids[attr]] = {value_ids[attr][item[attr]]: 1.0}
        if include_category and "diagnosis" in item:
            entry[attr_ids["diagnosis"]] = {value_ids["diagnosis"][item["diagnosis"]]: 1.0}
        encoded.append(entry)
    return encoded


def run():
    """
    Run the inverse base-rate effect experiment (Medin & Edelson, 1988).
    Medical diagnosis cover story (Symptom vectors).
    """
    random_seeds = [RANDOM_SEED + i * 31 for i in range(5)]
    ratio_conditions = [4.0] # Simplify to main effect
    blocks = 15
    epochs = 4
    rows = []

    attr_ids, value_ids = make_mappings()
    category_attr = attr_ids["diagnosis"]
    cat_id_common = value_ids["diagnosis"]["Common_Disease"]
    cat_id_rare = value_ids["diagnosis"]["Rare_Disease"]

    for rs in random_seeds:
        seed(rs)
        np.random.seed(rs)
        
        # Build test set (same for all conditions)
        test_patterns = build_test_medin()

        for ratio in ratio_conditions:
            # Build training set for this ratio
            train_raw = build_train_medin(ratio_common_rare=ratio)
            
            for epoch in range(1, epochs + 1):
                model = CobwebDiscreteTree(alpha=0.5)
                for block in range(1, blocks + 1):
                    shuffle(train_raw)
                    train_items = encode_items(train_raw, attr_ids, value_ids, include_category=True)
                    model.fit(train_items, 1, True)

                    # Evaluate on each test pattern
                    for test_item, true_label, item_type in test_patterns:
                        test_encoded = encode_items([test_item], attr_ids, value_ids, include_category=False)[0]
                        # Corrected predict call: dictionary access, not tuple index
                        pred_dict = model.predict(test_encoded, 100, True)
                        pred = pred_dict.get(category_attr, {})

                        prob_common = pred.get(cat_id_common, 0.0)
                        prob_rare = pred.get(cat_id_rare, 0.0)
                        
                        # Predicted category
                        pred_label = "Common" if prob_common >= prob_rare else "Rare"
                        
                        # For BC critical test, check if we get inverse base-rate effect
                        shows_ibre = (item_type == "BC_critical" and pred_label == "Rare")

                        rows.append({
                            "seed": rs,
                            "epoch": epoch,
                            "block": block,
                            "ratio": ratio,
                            "test_type": item_type,
                            "true_label": true_label,
                            "pred_label": pred_label,
                            "prob_common": prob_common,
                            "prob_rare": prob_rare,
                            "shows_ibre": shows_ibre,
                        })

    df = pd.DataFrame(rows)
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV results
    df.to_csv(str(results_dir / "exp_base_rate_discrete.csv"), index=False)
    
    # Calculate and save IBRE summary statistics
    bc_trials = df[df["test_type"] == "BC_critical"]
    ibre_rate = bc_trials["shows_ibre"].mean()
    
    # Save experiment metadata
    metadata = {
        "experiment": "inverse_base_rate_effect",
        "citation": "Medin, D. L., & Edelson, S. M. (1988). Problem structure and base-rate information. JEP: General, 117(1), 68-85.",
        "random_seed": RANDOM_SEED,
        "seeds_used": random_seeds,
        "ratio_conditions": ratio_conditions,
        "num_blocks": blocks,
        "num_epochs": epochs,
        "training_design": "AB->Common (frequent), AC->Rare (infrequent)",
        "critical_test": "BC (ambiguous)",
        "ibre_rate_observed": float(ibre_rate),
        "expected_effect": "On BC trials, bias toward Rare despite lower base rate (inverse base-rate effect)"
    }
    
    with open(results_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    run()
