from cobweb.cobweb_discrete import CobwebDiscreteTree
from random import seed, shuffle
import numpy as np
import pandas as pd
from pathlib import Path

# Discrete specific-instance effect: one color exemplar is oversampled.

colors_A = [f"c{i}" for i in range(6)]  # c0 will be oversampled
colors_B = [f"c{i}" for i in range(6, 12)]
all_colors = colors_A + colors_B


def make_mappings():
    attr_ids = {"color": 0, "category": 1}
    value_ids = {
        "color": {c: i for i, c in enumerate(all_colors)},
        "category": {"A": 0, "B": 1},
    }
    return attr_ids, value_ids


def build_train(rs, base_per_color=20, freq_multiplier=5):
    rng = np.random.default_rng(rs)
    items = []
    # Category A colors
    for color in colors_A:
        reps = base_per_color * (freq_multiplier if color == colors_A[0] else 1)
        for _ in range(reps):
            items.append((color, "A"))
    # Category B colors
    for color in colors_B:
        for _ in range(base_per_color):
            items.append((color, "B"))
    rng.shuffle(items)
    return items


def build_tests():
    return [
        ("freq_A", colors_A[0]),
        ("rare_A", colors_A[1]),
        ("rare_B", colors_B[0]),
    ]


def encode_item(color, attr_ids, value_ids, include_label: bool):
    entry = {attr_ids["color"]: {value_ids["color"][color]: 1.0}}
    if include_label:
        category = "A" if color in colors_A else "B"
        entry[attr_ids["category"]] = {value_ids["category"][category]: 1.0}
    return entry


def evaluate(model, test_items, attr_ids, value_ids):
    results = []
    for name, color in test_items:
        feat = {attr_ids["color"]: {value_ids["color"][color]: 1.0}}
        pred = model.predict_probs(feat, 1000, False, False)[1]
        cat_probs = pred.get(attr_ids["category"], {})
        prob_a = cat_probs.get(value_ids["category"]["A"], 0.0)
        prob_b = cat_probs.get(value_ids["category"]["B"], 0.0)
        results.append((name, color, prob_a, prob_b))
    return results


def run():
    random_seeds = [1, 32, 64, 128, 356]
    blocks = 10
    epochs = 4
    rows = []

    attr_ids, value_ids = make_mappings()

    for rs in random_seeds:
        seed(rs)
        np.random.seed(rs)
        train_items = build_train(rs)
        tests = build_tests()

        for epoch in range(1, epochs + 1):
            model = CobwebDiscreteTree(alpha=0.5)
            for block in range(1, blocks + 1):
                shuffle(train_items)
                encoded = []
                for color, cat in train_items:
                    entry = encode_item(color, attr_ids, value_ids, include_label=True)
                    encoded.append(entry)
                model.fit(encoded, 1, True)

                evals = evaluate(model, tests, attr_ids, value_ids)
                for name, color, prob_a, prob_b in evals:
                    rows.append(
                        {
                            "seed": rs,
                            "epoch": epoch,
                            "block": block,
                            "stimulus": name,
                            "color": color,
                            "prob_A": prob_a,
                            "prob_B": prob_b,
                        }
                    )

    df = pd.DataFrame(rows)
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(results_dir / "exp_specific_instance_discrete.csv"), index=False)


if __name__ == "__main__":
    run()
