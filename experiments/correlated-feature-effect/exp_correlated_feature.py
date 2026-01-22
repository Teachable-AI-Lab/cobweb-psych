from cobweb.cobweb_continuous import CobwebContinuousTree
from random import seed, shuffle
import numpy as np
import pandas as pd


def label_vec(idx):
    vec = np.zeros(2)
    vec[idx] = 1.0
    return vec


def dataset(structure):
    items = []
    if structure == "xor":
        samples = [
            (np.array([0.0, 0.0]), 0),
            (np.array([0.0, 1.0]), 1),
            (np.array([1.0, 0.0]), 1),
            (np.array([1.0, 1.0]), 0),
        ]
    else:  # separable
        samples = [
            (np.array([0.0, 0.0]), 0),
            (np.array([0.0, 1.0]), 0),
            (np.array([1.0, 0.0]), 1),
            (np.array([1.0, 1.0]), 1),
        ]
    # replicate to give the model more observations
    for feat, label in samples:
        for _ in range(6):
            jitter = np.random.normal(0.0, 0.05, 2)
            items.append((feat + jitter, label_vec(label)))
    test = [feat for feat, _ in samples]
    targets = [label for _, label in samples]
    return items, test, targets


def accuracy(model, tests, targets):
    correct = 0
    for feat, target in zip(tests, targets):
        pred_vec = model.predict(feat, np.zeros(2), 30, False)
        if int(np.argmax(pred_vec)) == target:
            correct += 1
    return correct / len(tests)


def run():
    random_seeds = [1, 32, 64, 128, 356]
    blocks = 15
    epochs = 4
    rows = []

    for structure in ["separable", "xor"]:
        for rs in random_seeds:
            seed(rs)
            np.random.seed(rs)
            train_items, tests, targets = dataset(structure)

            for epoch in range(1, epochs + 1):
                model = CobwebContinuousTree(2, 2, alpha=0.6, prior_var=0.2)
                for block in range(1, blocks + 1):
                    shuffle(train_items)
                    for feat, label in train_items:
                        model.ifit(feat, label)

                    acc = accuracy(model, tests, targets)
                    rows.append({
                        "structure": structure,
                        "seed": rs,
                        "epoch": epoch,
                        "block": block,
                        "accuracy": acc,
                    })

    df = pd.DataFrame(rows)
    df.to_csv("exp_correlated_feature.csv", index=False)


if __name__ == "__main__":
    run()
