from cobweb.cobweb_discrete import CobwebDiscreteTree
from random import seed, shuffle
import numpy as np
import pandas as pd
from pathlib import Path

# Discrete version of the correlated-feature effect experiment.
# Two binary attributes, structures: separable vs xor.

attribute_values = {
    "feat1": ["low", "high"],
    "feat2": ["low", "high"],
    "category": ["A", "B"],
}


def make_mappings():
    attr_ids = {name: idx for idx, name in enumerate(attribute_values.keys())}
    value_ids = {}
    for attr, vals in attribute_values.items():
        value_ids[attr] = {v: i for i, v in enumerate(vals)}
    return attr_ids, value_ids


def dataset(structure, n_rep=6):
    # base patterns (no jitter for discrete case); replicate each pattern n_rep times
    if structure == "xor":
        samples = [
            ({"feat1": "low", "feat2": "low"}, "A"),
            ({"feat1": "low", "feat2": "high"}, "B"),
            ({"feat1": "high", "feat2": "low"}, "B"),
            ({"feat1": "high", "feat2": "high"}, "A"),
        ]
    else:  # separable
        samples = [
            ({"feat1": "low", "feat2": "low"}, "A"),
            ({"feat1": "low", "feat2": "high"}, "A"),
            ({"feat1": "high", "feat2": "low"}, "B"),
            ({"feat1": "high", "feat2": "high"}, "B"),
        ]

    items = []
    for feat_vals, label in samples:
        for _ in range(n_rep):
            items.append((feat_vals, label))

    # test items are the canonical four patterns
    test = [s[0] for s in samples]
    targets = [1 if s[1] == "B" else 0 for s in samples]
    return items, test, targets


def encode_item_discrete(item, attr_ids, value_ids):
    entry = {}
    for attr, val in item.items():
        entry[attr_ids[attr]] = {value_ids[attr][val]: 1.0}
    return entry


def evaluate(model, test_items_encoded, test_targets, category_attr, cat_id_A, cat_id_B):
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
    random_seeds = [1, 32, 64, 128, 356]
    blocks = 15
    epochs = 4
    rows = []

    attr_ids, value_ids = make_mappings()
    category_attr = attr_ids["category"]
    cat_id_A = value_ids["category"]["A"]
    cat_id_B = value_ids["category"]["B"]

    for structure in ["separable", "xor"]:
        for rs in random_seeds:
            seed(rs)
            np.random.seed(rs)
            train_items, tests, targets = dataset(structure, n_rep=6)

            test_items_encoded = [encode_item_discrete(t, attr_ids, value_ids) for t in tests]

            for epoch in range(1, epochs + 1):
                model = CobwebDiscreteTree(alpha=0.5)
                for block in range(1, blocks + 1):
                    shuffle(train_items)
                    train_encoded = [
                        {**encode_item_discrete(feat, attr_ids, value_ids), category_attr: {cat_id_A: 1.0} if label == "A" else {cat_id_B: 1.0}}
                        for feat, label in train_items
                    ]
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
    df.to_csv(str(results_dir / "exp_correlated_feature_discrete.csv"), index=False)


if __name__ == "__main__":
    run()
