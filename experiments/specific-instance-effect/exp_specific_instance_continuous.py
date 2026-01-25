from cobweb.cobweb_continuous import CobwebContinuousTree
from random import seed, shuffle
import numpy as np
import pandas as pd
from pathlib import Path

# Continuous specific-instance effect (Nosofsky-style): one exemplar is oversampled.


def label_vec(cat):
    vec = np.zeros(2)
    vec[cat] = 1.0
    return vec


def build_train(rs, special_reps=120, base_per_class=60, spread=0.12):
    rng = np.random.default_rng(rs)
    items = []
    items.extend((rng.normal(0.0, spread, 2), label_vec(0)) for _ in range(base_per_class))
    items.extend((rng.normal(1.0, spread, 2), label_vec(1)) for _ in range(base_per_class))
    special = np.array([0.2, 0.2])
    items.extend((special.copy(), label_vec(0)) for _ in range(special_reps))
    return [tuple_item for tuple_item in items]


def build_tests():
    return [
        ("special", np.array([0.2, 0.2])),
        ("special_perturbed", np.array([0.25, 0.25])),
        ("typical_A", np.array([0.0, 0.0])),
        ("typical_B", np.array([1.0, 1.0])),
        ("boundary", np.array([0.5, 0.5])),
    ]


def run():
    random_seeds = [1, 32, 64, 128, 356]
    blocks = 10
    epochs = 4
    rows = []
    tests = build_tests()

    for rs in random_seeds:
        seed(rs)
        np.random.seed(rs)
        train_items = build_train(rs)

        for epoch in range(1, epochs + 1):
            model = CobwebContinuousTree(2, 2, alpha=0.6, prior_var=0.2)
            for block in range(1, blocks + 1):
                shuffle(train_items)
                for feat, label in train_items:
                    model.ifit(feat, label)

                for stim_name, feat in tests:
                    pred_vec = model.predict(feat, np.zeros(2), 30, False)
                    rows.append({
                        "seed": rs,
                        "epoch": epoch,
                        "block": block,
                        "stimulus": stim_name,
                        "pred_A": float(pred_vec[0]),
                        "pred_B": float(pred_vec[1]),
                    })

    df = pd.DataFrame(rows)
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(results_dir / "exp_specific_instance_continuous.csv"), index=False)


if __name__ == "__main__":
    run()
