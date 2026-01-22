from cobweb.cobweb_continuous import CobwebContinuousTree
from itertools import combinations
from random import seed, shuffle
import numpy as np
import pandas as pd


def flip_bits(center, idxs):
    vec = center.copy()
    for idx in idxs:
        vec[idx] = 1.0 - vec[idx]
    return vec


def build_train(distance_counts, dim, center_value):
    center = np.full(dim, center_value, dtype=float)
    combos_cache = {d: list(combinations(range(dim), d)) for d in distance_counts if d > 0}
    items = []
    for dist, count in distance_counts.items():
        if dist == 0:
            items.extend([center.copy() for _ in range(count)])
            continue
        options = combos_cache[dist]
        for i in range(count):
            idxs = options[i % len(options)]
            items.append(flip_bits(center, idxs))
    return items


def build_tests(dim):
    tests = []
    for cat, center_value in enumerate([0.0, 1.0]):
        center = np.full(dim, center_value, dtype=float)
        for dist in range(dim + 1):
            idxs = () if dist == 0 else list(combinations(range(dim), dist))[0]
            tests.append({
                "features": flip_bits(center, idxs),
                "category": "A" if cat == 0 else "B",
                "distance": dist,
            })
    return tests


def label_vec(cat):
    vec = np.zeros(2)
    vec[cat] = 1.0
    return vec


def run():
    dim = 6
    distance_counts = {0: 6, 1: 18, 2: 12, 3: 6}
    random_seeds = [1, 32, 64, 128, 356]
    blocks = 10
    epochs = 4
    tests = build_tests(dim)
    rows = []

    for rs in random_seeds:
        seed(rs)
        np.random.seed(rs)
        train_items = []
        for cat, center_value in enumerate([0.0, 1.0]):
            features = build_train(distance_counts, dim, center_value)
            for feat in features:
                train_items.append((feat, label_vec(cat)))

        for epoch in range(1, epochs + 1):
            model = CobwebContinuousTree(dim, 2, alpha=0.6, prior_var=0.25)
            for block in range(1, blocks + 1):
                shuffle(train_items)
                for feat, label in train_items:
                    model.ifit(feat, label)

                for test_idx, test_item in enumerate(tests):
                    pred_vec = model.predict(test_item["features"], np.zeros(2), 30, False)
                    rows.append({
                        "seed": rs,
                        "epoch": epoch,
                        "block": block,
                        "stimulus": test_idx + 1,
                        "category": test_item["category"],
                        "distance": test_item["distance"],
                        "pred_A": float(pred_vec[0]),
                        "pred_B": float(pred_vec[1]),
                    })

    df = pd.DataFrame(rows)
    df.to_csv("exp_central_tendency_continuous.csv", index=False)


if __name__ == "__main__":
    run()
