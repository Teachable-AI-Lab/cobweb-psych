from cobweb.cobweb_continuous import CobwebContinuousTree
from random import seed, shuffle
import numpy as np
import pandas as pd
from pathlib import Path


def label_vec(idx):
    vec = np.zeros(2)
    vec[idx] = 1.0
    return vec


def build_train(rs, base_rate, n_total=240, spread=0.18):
    rng = np.random.default_rng(rs)
    n_a = int(n_total * base_rate)
    n_b = n_total - n_a
    feats = []
    feats.extend((rng.normal(0.0, spread, 2), label_vec(0)) for _ in range(n_a))
    feats.extend((rng.normal(1.0, spread, 2), label_vec(1)) for _ in range(n_b))
    return [tuple_item for tuple_item in feats]


def sample_stream(rs, n=240, spread=0.18):
    rng = np.random.default_rng(rs + 99)
    mix_labels = rng.choice([0, 1], size=n, p=[0.6, 0.4])
    feats = []
    for lbl in mix_labels:
        center = 0.0 if lbl == 0 else 1.0
        feats.append(rng.normal(center, spread, 2))
    return feats


def predicted_share(model, feats):
    chosen = 0
    for feat in feats:
        pred = model.predict(feat, np.zeros(2), 30, False)
        if int(np.argmax(pred)) == 1:
            chosen += 1
    return chosen / len(feats)


def run():
    random_seeds = [1, 32, 64, 128, 356]
    base_rates = [0.5, 0.65, 0.8]
    blocks = 12
    epochs = 4
    rows = []

    for rs in random_seeds:
        seed(rs)
        np.random.seed(rs)
        stream_feats = sample_stream(rs)

        for base_rate in base_rates:
            train_items = build_train(rs, base_rate)

            for epoch in range(1, epochs + 1):
                model = CobwebContinuousTree(2, 2, alpha=0.6, prior_var=0.2)
                for block in range(1, blocks + 1):
                    shuffle(train_items)
                    for feat, label in train_items:
                        model.ifit(feat, label)

                    share_b = predicted_share(model, stream_feats)
                    rows.append({
                        "seed": rs,
                        "epoch": epoch,
                        "block": block,
                        "base_rate": base_rate,
                        "pred_share_B": share_b,
                        "match_gap": share_b - (1 - base_rate),
                    })

    df = pd.DataFrame(rows)
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(results_dir / "exp_probability_matching.csv"), index=False)


if __name__ == "__main__":
    run()
