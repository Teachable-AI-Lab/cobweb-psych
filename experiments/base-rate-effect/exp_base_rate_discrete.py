from random import seed, shuffle
from pathlib import Path
import numpy as np
import pandas as pd
from cobweb.cobweb_discrete import CobwebDiscreteTree

# Discrete base-rate experiment inspired by Homa & Vosburgh (1976) and Medin & Edelson (1988).
# We bias category frequency during training and measure how Cobweb's predictions favor the
# frequent category on a balanced test set.

attribute_values = {
    "shape": ["square", "triangle"],
    "color": ["red", "blue"],
    "size": ["small", "large"],
    "category": ["A", "B"],
}


def make_mappings():
    attr_ids = {name: idx for idx, name in enumerate(attribute_values.keys())}
    value_ids = {}
    value_ids_reverse = {}
    next_val_id = 0
    for attr, vals in attribute_values.items():
        value_ids[attr] = {}
        value_ids_reverse[attr] = {}
        for v in vals:
            value_ids[attr][v] = next_val_id
            value_ids_reverse[attr][next_val_id] = v
            next_val_id += 1
    return attr_ids, value_ids, value_ids_reverse


def sample_features(cat: str, rng: np.random.Generator, noise: float = 0.15):
    # Prototypes for each category; noise flips features to create variation.
    prototype = {
        "A": {"shape": "square", "color": "red", "size": "small"},
        "B": {"shape": "triangle", "color": "blue", "size": "large"},
    }[cat]
    features = {}
    for attr in ["shape", "color", "size"]:
        if rng.random() < noise:
            # flip to the other option
            options = attribute_values[attr]
            features[attr] = options[1] if prototype[attr] == options[0] else options[0]
        else:
            features[attr] = prototype[attr]
    return features


def build_train(rs: int, base_rate: float, n_total: int = 240, noise: float = 0.15):
    rng = np.random.default_rng(rs)
    n_a = int(n_total * base_rate)
    n_b = n_total - n_a
    items = []
    items.extend({**sample_features("A", rng, noise), "category": "A"} for _ in range(n_a))
    items.extend({**sample_features("B", rng, noise), "category": "B"} for _ in range(n_b))
    return items


def build_test(rs: int, n_per_class: int = 400, noise: float = 0.15):
    rng = np.random.default_rng(rs + 13)
    items = []
    targets = []
    for cat in ["A", "B"]:
        for _ in range(n_per_class):
            items.append(sample_features(cat, rng, noise))
            targets.append(cat)
    return items, targets


def encode_items(raw_items, attr_ids, value_ids, include_category: bool = True):
    encoded = []
    for item in raw_items:
        entry = {}
        for attr in ["shape", "color", "size"]:
            entry[attr_ids[attr]] = {value_ids[attr][item[attr]]: 1.0}
        if include_category:
            entry[attr_ids["category"]] = {value_ids["category"][item["category"]]: 1.0}
        encoded.append(entry)
    return encoded


def run():
    random_seeds = [1, 32, 64, 128, 356]
    base_rates = [0.5, 0.65, 0.8, 0.9]
    blocks = 12
    epochs = 3
    rows = []

    attr_ids, value_ids, value_ids_reverse = make_mappings()
    category_attr = attr_ids["category"]
    cat_id_A = value_ids["category"]["A"]
    cat_id_B = value_ids["category"]["B"]

    for rs in random_seeds:
        seed(rs)
        np.random.seed(rs)
        test_raw, test_targets = build_test(rs)
        test_items = encode_items(test_raw, attr_ids, value_ids, include_category=False)

        for base_rate in base_rates:
            train_raw = build_train(rs, base_rate)
            for epoch in range(1, epochs + 1):
                model = CobwebDiscreteTree(alpha=0.5)
                for block in range(1, blocks + 1):
                    shuffle(train_raw)
                    train_items = encode_items(train_raw, attr_ids, value_ids, include_category=True)
                    model.fit(train_items, 1, True)

                    correct = 0
                    share_b_probs = []
                    for feat, target in zip(test_items, test_targets):
                        pred = model.predict_probs(feat, 100, True, False)[1]
                        cat_probs = pred.get(category_attr, {})
                        prob_a = cat_probs.get(cat_id_A, 0.0)
                        prob_b = cat_probs.get(cat_id_B, 0.0)
                        pred_label = "A" if prob_a >= prob_b else "B"
                        if pred_label == target:
                            correct += 1
                        share_b_probs.append(prob_b)

                    rows.append(
                        {
                            "seed": rs,
                            "epoch": epoch,
                            "block": block,
                            "base_rate": base_rate,
                            "accuracy": correct / len(test_items),
                            "pred_share_B": float(np.mean(share_b_probs)),
                        }
                    )

    df = pd.DataFrame(rows)
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(results_dir / "exp_base_rate_discrete.csv"), index=False)


if __name__ == "__main__":
    run()
