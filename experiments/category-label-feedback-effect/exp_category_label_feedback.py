from cobweb.cobweb_continuous import CobwebContinuousTree
from random import seed, shuffle
import numpy as np
import pandas as pd


def label_vec(idx):
    vec = np.zeros(2)
    vec[idx] = 1.0
    return vec


def build_dataset(rs, n_per_class=80, spread=0.15):
    rng = np.random.default_rng(rs)
    feats = []
    labels = []
    feats.append(rng.normal(0.0, spread, size=(n_per_class, 2)))
    feats.append(rng.normal(1.0, spread, size=(n_per_class, 2)))
    features = np.vstack(feats)
    labels = [0] * n_per_class + [1] * n_per_class
    return list(zip(features, labels))


def accuracy(model, tests, targets):
    correct = 0
    for feat, target in zip(tests, targets):
        pred = model.predict(feat, np.zeros(2), 30, False)
        if int(np.argmax(pred)) == target:
            correct += 1
    return correct / len(tests)


def run():
    random_seeds = [1, 32, 64, 128, 356]
    label_rates = [1.0, 0.5, 0.1]
    blocks = 12
    epochs = 4
    rows = []

    for rs in random_seeds:
        seed(rs)
        np.random.seed(rs)
        data = build_dataset(rs)
        feats = [feat for feat, _ in data]
        targets = [label for _, label in data]

        for label_rate in label_rates:
            for epoch in range(1, epochs + 1):
                model = CobwebContinuousTree(2, 2, alpha=0.6, prior_var=0.2)
                for block in range(1, blocks + 1):
                    shuffle(data)
                    for feat, target in data:
                        lbl = label_vec(target) if np.random.rand() < label_rate else np.zeros(2)
                        model.ifit(feat, lbl)

                    acc = accuracy(model, feats, targets)
                    rows.append({
                        "seed": rs,
                        "epoch": epoch,
                        "block": block,
                        "label_rate": label_rate,
                        "accuracy": acc,
                    })

    df = pd.DataFrame(rows)
    df.to_csv("exp_category_label_feedback.csv", index=False)


if __name__ == "__main__":
    run()
