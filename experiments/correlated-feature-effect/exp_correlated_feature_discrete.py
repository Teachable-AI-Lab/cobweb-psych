from cobweb.cobweb_discrete import CobwebDiscreteTree
from random import seed, shuffle
import numpy as np
import pandas as pd
import json
from pathlib import Path

# Correlated-feature effect / XOR difficulty (Medin et al., 1982)
# Goal: Show that XOR/configural category structures are harder to learn than 
# linearly separable structures, despite matched marginal frequencies.
#
# Citation: Medin, D. L., Altom, M. W., Edelson, S. M., & Freko, D. (1982). 
#           Correlated symptoms and simulated medical classification. 
#           Journal of Experimental Psychology: Learning, Memory, and Cognition, 8(1), 37-50.

# Random seed for reproducibility
RANDOM_SEED = 12345

attribute_values = {
    "symptom_1": ["absent", "present"],
    "symptom_2": ["absent", "present"],
    "symptom_3": ["absent", "present"],
    "symptom_4": ["absent", "present"],
    "diagnosis": ["Disease_A", "Disease_B"],
}


def make_mappings():
    attr_ids = {name: idx for idx, name in enumerate(attribute_values.keys())}
    value_ids = {}
    for attr, vals in attribute_values.items():
        value_ids[attr] = {v: i for i, v in enumerate(vals)}
    return attr_ids, value_ids


def dataset(structure, n_rep=8):
    """
    Generate training and test datasets for XOR vs separable category structures.
    Medical diagnosis cover story (Case studies).
    
    Structure based on Medin et al. (1982) Exp 2.
    4 symptoms (S1, S2, S3, S4).
    
    Correlated (XOR) Structure:
    S1 & S2 form XOR pattern. S3 & S4 are random/irrelevant or match pattern.
    Here we focus on the core effect: S1/S2 correlation.
    
    Pattern 1: 0 0 -> A
    Pattern 2: 0 1 -> B
    Pattern 3: 1 0 -> B
    Pattern 4: 1 1 -> A
    (0=absent, 1=present)
    
    Args:
        structure: 'correlated' (XOR) or 'uncorrelated' (Separable/Linear)
    """
    # Define 4 patterns
    if structure == "correlated":
        # XOR on S1, S2
        patterns = [
            ([0, 0, 0, 0], "Disease_A"), # 00
            ([0, 1, 0, 0], "Disease_B"), # 01
            ([1, 0, 0, 0], "Disease_B"), # 10
            ([1, 1, 0, 0], "Disease_A"), # 11
        ]
    else:  # uncorrelated / separable
        # S1 predicts A (0->A, 1->B) independent of S2
        patterns = [
            ([0, 0, 0, 0], "Disease_A"), # S1=0 -> A
            ([0, 1, 0, 0], "Disease_A"), # S1=0 -> A
            ([1, 0, 0, 0], "Disease_B"), # S1=1 -> B
            ([1, 1, 0, 0], "Disease_B"), # S1=1 -> B
        ]

    items = []
    # Replicate patterns
    for p, label in patterns:
        feats = {
            "symptom_1": "present" if p[0] else "absent",
            "symptom_2": "present" if p[1] else "absent",
            "symptom_3": "present" if p[2] else "absent",
            "symptom_4": "present" if p[3] else "absent",
        }
        for _ in range(n_rep):
            items.append((feats, label))
            
    # Test on unique patterns
    test_items = []
    test_targets = []
    for p, label in patterns:
        feats = {
            "symptom_1": "present" if p[0] else "absent",
            "symptom_2": "present" if p[1] else "absent",
            "symptom_3": "present" if p[2] else "absent",
            "symptom_4": "present" if p[3] else "absent",
        }
        test_items.append(feats)
        test_targets.append(1 if label == "Disease_B" else 0)
        
    return items, test_items, test_targets


def encode_item_discrete(item, attr_ids, value_ids):
    """Encode item for Cobweb discrete tree."""
    entry = {}
    for attr, val in item.items():
        entry[attr_ids[attr]] = {value_ids[attr][val]: 1.0}
    return entry


def evaluate(model, test_items_encoded, test_targets, category_attr, cat_id_A, cat_id_B):
    """
    Evaluate model on test set.
    """
    correct = 0
    share_b = []
    for feat_enc, target in zip(test_items_encoded, test_targets):
        pred = model.predict_probs(feat_enc, 100, True, False)[1]
        cat_probs = pred.get(category_attr, {})
        prob_a = cat_probs.get(cat_id_A, 0.0)
        prob_b = cat_probs.get(cat_id_B, 0.0)
        pred_label = 1 if prob_b > prob_a else 0
        if pred_label == target:
            correct += 1
        share_b.append(prob_b)
    return correct / len(test_items_encoded), float(np.mean(share_b))


def run():
    """
    Run the correlated-feature / XOR difficulty experiment.
    """
    random_seeds = [RANDOM_SEED + i * 31 for i in range(5)]
    blocks = 15
    epochs = 4
    rows = []

    attr_ids, value_ids = make_mappings()
    category_attr = attr_ids["diagnosis"]
    cat_id_A = value_ids["diagnosis"]["Disease_A"]
    cat_id_B = value_ids["diagnosis"]["Disease_B"]

    for structure in ["uncorrelated", "correlated"]:
        for rs in random_seeds:
            seed(rs)
            np.random.seed(rs)
            train_items, tests, targets = dataset(structure, n_rep=8)

            test_items_encoded = [encode_item_discrete(t, attr_ids, value_ids) for t in tests]

            for epoch in range(1, epochs + 1):
                model = CobwebDiscreteTree(alpha=0.5)
                for block in range(1, blocks + 1):
                    shuffle(train_items)
                    train_encoded = [
                        {**encode_item_discrete(feat, attr_ids, value_ids), 
                         category_attr: {cat_id_A: 1.0} if label == "Disease_A" else {cat_id_B: 1.0}}
                        for feat, label in train_items
                    ]
                    # Batch fit for speed if supported, else loop
                    model.fit(train_encoded, 1, True)

                    acc, mean_prob_b = evaluate(model, test_items_encoded, targets, category_attr, cat_id_A, cat_id_B)
                    rows.append({
                        "structure": structure,
                        "seed": rs,
                        "epoch": epoch,
                        "block": block,
                        "accuracy": acc,
                        "pred_share_B": mean_prob_b,
                    })

    df = pd.DataFrame(rows)
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV results
    df.to_csv(str(results_dir / "exp_correlated_feature_discrete.csv"), index=False)
    
    # Calculate final accuracies for effect summary
    final_block = df[df["block"] == blocks]
    sep_acc = final_block[final_block["structure"] == "uncorrelated"]["accuracy"].mean()
    xor_acc = final_block[final_block["structure"] == "correlated"]["accuracy"].mean()
    
    # Save experiment metadata
    metadata = {
        "experiment": "correlated_feature_xor_difficulty",
        "citation": "Medin, D. L., et al. (1982). Correlated symptoms and simulated medical classification. JEP: LMC, 8(1), 37-50.",
        "random_seed": RANDOM_SEED,
        "seeds_used": random_seeds,
        "num_blocks": blocks,
        "num_epochs": epochs,
        "structures": ["uncorrelated", "correlated"],
        "final_accuracy_separable": float(sep_acc),
        "final_accuracy_xor": float(xor_acc),
        "expected_effect": "Correlated (XOR) structure should show slower learning and lower accuracy than uncorrelated structure"
    }
    
    with open(results_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    run()
