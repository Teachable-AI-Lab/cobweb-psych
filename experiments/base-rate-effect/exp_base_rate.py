from cobweb.cobweb_continuous import CobwebContinuousTree
from random import seed, shuffle
import numpy as np
import pandas as pd


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


def build_test(rs, n_per_class=120, spread=0.18):
    rng = np.random.default_rng(rs + 13)
    feats = []
    targets = []
    feats.extend(rng.normal(0.0, spread, size=(n_per_class, 2)))
    feats.extend(rng.normal(1.0, spread, size=(n_per_class, 2)))
    targets.extend([0] * n_per_class)
    targets.extend([1] * n_per_class)
    return list(feats), targets


def accuracy(model, feats, targets):
    correct = 0
    for feat, target in zip(feats, targets):
        pred = model.predict(feat, np.zeros(2), 30, False)
        if int(np.argmax(pred)) == target:
            correct += 1
    return correct / len(feats)


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
        test_feats, test_targets = build_test(rs)
        probe_feats = test_feats[:]

        for base_rate in base_rates:
            train_items = build_train(rs, base_rate)

            for epoch in range(1, epochs + 1):
                model = CobwebContinuousTree(2, 2, alpha=0.6, prior_var=0.2)
                for block in range(1, blocks + 1):
                    shuffle(train_items)
                    for feat, label in train_items:
                        model.ifit(feat, label)

                    acc = accuracy(model, test_feats, test_targets)
                    share_b = predicted_share(model, probe_feats)
                    rows.append({
                        "seed": rs,
                        "epoch": epoch,
                        "block": block,
                        "base_rate": base_rate,
                        "accuracy": acc,
                        "pred_share_B": share_b,
                    })

    df = pd.DataFrame(rows)
    df.to_csv("exp_base_rate.csv", index=False)


if __name__ == "__main__":
    run()
