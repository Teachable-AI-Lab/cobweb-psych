from cobweb.cobweb_continuous import CobwebContinuousTree
from random import seed, shuffle
import numpy as np
import pandas as pd
from pathlib import Path

structure = {
    "animal": {
        "dog": ["shepherd", "beagle"],
        "cat": ["siamese", "persian"],
    },
    "vehicle": {
        "car": ["sedan", "suv"],
        "boat": ["sailboat", "canoe"],
    },
}

basic_centers = {
    "dog": np.array([-1.0, 0.2, 0.0]),
    "cat": np.array([-0.8, -0.2, 0.2]),
    "car": np.array([1.0, 0.0, 0.2]),
    "boat": np.array([1.2, 0.6, -0.1]),
}

sub_offsets = {
    "shepherd": np.array([0.05, 0.1, 0.05]),
    "beagle": np.array([-0.05, -0.05, 0.0]),
    "siamese": np.array([0.0, 0.05, -0.05]),
    "persian": np.array([0.05, -0.05, 0.05]),
    "sedan": np.array([-0.05, 0.0, -0.05]),
    "suv": np.array([0.05, 0.15, 0.1]),
    "sailboat": np.array([0.0, 0.1, 0.15]),
    "canoe": np.array([0.05, -0.1, -0.05]),
}

levels = {
    "super": sorted(list(structure.keys())),
    "basic": sorted(list(basic_centers.keys())),
    "sub": sorted(list(sub_offsets.keys())),
}


def label_vec(level, name):
    size = len(levels[level])
    vec = np.zeros(size)
    vec[levels[level].index(name)] = 1.0
    return vec


def build_dataset(rs, n_per_sub=30, spread=0.12):
    rng = np.random.default_rng(rs)
    items = []
    for super_name, basic_map in structure.items():
        for basic_name, sub_list in basic_map.items():
            base_center = basic_centers[basic_name]
            for sub_name in sub_list:
                for _ in range(n_per_sub):
                    feat = base_center + sub_offsets[sub_name] + rng.normal(0.0, spread, 3)
                    items.append({
                        "features": feat,
                        "super": super_name,
                        "basic": basic_name,
                        "sub": sub_name,
                    })
    return items


def accuracy(model, items, level):
    correct = 0
    for item in items:
        target_idx = levels[level].index(item[level])
        pred = model.predict(item["features"], np.zeros(len(levels[level])), 30, False)
        if int(np.argmax(pred)) == target_idx:
            correct += 1
    return correct / len(items)


def run():
    random_seeds = [1, 32, 64, 128, 356]
    blocks = 10
    epochs = 4
    rows = []

    for rs in random_seeds:
        seed(rs)
        np.random.seed(rs)
        data = build_dataset(rs)

        for epoch in range(1, epochs + 1):
            models = {
                "super": CobwebContinuousTree(3, len(levels["super"]), alpha=0.6, prior_var=0.25),
                "basic": CobwebContinuousTree(3, len(levels["basic"]), alpha=0.6, prior_var=0.25),
                "sub": CobwebContinuousTree(3, len(levels["sub"]), alpha=0.6, prior_var=0.25),
            }

            for block in range(1, blocks + 1):
                shuffle(data)
                for item in data:
                    feat = item["features"]
                    models["super"].ifit(feat, label_vec("super", item["super"]))
                    models["basic"].ifit(feat, label_vec("basic", item["basic"]))
                    models["sub"].ifit(feat, label_vec("sub", item["sub"]))

                for level in ["super", "basic", "sub"]:
                    acc = accuracy(models[level], data, level)
                    rows.append({
                        "seed": rs,
                        "epoch": epoch,
                        "block": block,
                        "level": level,
                        "accuracy": acc,
                    })

    df = pd.DataFrame(rows)
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(results_dir / "exp_basic_level.csv"), index=False)


if __name__ == "__main__":
    run()
