from random import seed, shuffle
from pathlib import Path
import numpy as np
import pandas as pd
from cobweb.cobweb_discrete import CobwebDiscreteTree

# Discrete analogue of the category label feedback effect (Posner & Keele, 1968; Homa & Cultice, 1984).
# We vary how often category labels are supplied during training and measure classification accuracy.

attribute_values = {
    "shape": ["circle", "triangle"],
    "color": ["red", "blue"],
    "fill": ["solid", "striped"],
    "category": ["A", "B"],
}


def make_mappings():
    attr_ids = {name: idx for idx, name in enumerate(attribute_values.keys())}
    value_ids = {}
    for attr, vals in attribute_values.items():
        value_ids[attr] = {v: i for i, v in enumerate(vals)}
    return attr_ids, value_ids


def sample_features(cat: str, rng: np.random.Generator, noise: float = 0.1):
    # Simple prototypes; noise flips attributes to create variation around each category center.
    prototype = {
        "A": {"shape": "circle", "color": "red", "fill": "solid"},
        "B": {"shape": "triangle", "color": "blue", "fill": "striped"},
    }[cat]
    features = {}
    for attr in ["shape", "color", "fill"]:
        if rng.random() < noise:
            options = attribute_values[attr]
            features[attr] = options[1] if prototype[attr] == options[0] else options[0]
        else:
            features[attr] = prototype[attr]
    return features


def build_dataset(rs: int, n_per_class: int = 400, noise: float = 0.1):
    rng = np.random.default_rng(rs)
    items = []
    for cat in ["A", "B"]:
        for _ in range(n_per_class):
            items.append({**sample_features(cat, rng, noise), "category": cat})
    return items


def encode_item(raw_item, attr_ids, value_ids, include_label: bool):
    entry = {}
    for attr in ["shape", "color", "fill"]:
        entry[attr_ids[attr]] = {value_ids[attr][raw_item[attr]]: 1.0}
    if include_label:
        entry[attr_ids["category"]] = {value_ids["category"][raw_item["category"]]: 1.0}
    return entry


def evaluate(model, test_items, test_targets, category_attr, cat_id_A, cat_id_B):
    correct = 0
    share_b = []
    for feat, target in zip(test_items, test_targets):
        pred = model.predict_probs(feat, 100, True, False)[1]
        cat_probs = pred.get(category_attr, {})
        prob_a = cat_probs.get(cat_id_A, 0.0)
        prob_b = cat_probs.get(cat_id_B, 0.0)
        pred_label = "A" if prob_a >= prob_b else "B"
        if pred_label == target:
            correct += 1
        share_b.append(prob_b)
    return correct / len(test_items), float(np.mean(share_b))


def run():
    random_seeds = [1, 32, 64, 128, 256]
    label_rates = [1.0, 0.5, 0.25, 0.1, 0.05]
    blocks = 12
    epochs = 3
    rows = []

    attr_ids, value_ids = make_mappings()
    category_attr = attr_ids["category"]
    cat_id_A = value_ids["category"]["A"]
    cat_id_B = value_ids["category"]["B"]

    for rs in random_seeds:
        seed(rs)
        np.random.seed(rs)
        data = build_dataset(rs)
        test_items = [encode_item(item, attr_ids, value_ids, include_label=False) for item in data]
        test_targets = [item["category"] for item in data]

        for label_rate in label_rates:
            for epoch in range(1, epochs + 1):
                model = CobwebDiscreteTree(alpha=0.5)
                for block in range(1, blocks + 1):
                    shuffle(data)
                    train_items = []
                    for item in data:
                        include_label = np.random.rand() < label_rate
                        train_items.append(encode_item(item, attr_ids, value_ids, include_label))
                    model.fit(train_items, 1, True)
                    acc, mean_prob_b = evaluate(model, test_items, test_targets, category_attr, cat_id_A, cat_id_B)
                    rows.append(
                        {
                            "seed": rs,
                            "epoch": epoch,
                            "block": block,
                            "label_rate": label_rate,
                            "accuracy": acc,
                            "pred_share_B": mean_prob_b,
                        }
                    )

    df = pd.DataFrame(rows)
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(results_dir / "exp_category_label_feedback_discrete.csv"), index=False)


if __name__ == "__main__":
    run()
