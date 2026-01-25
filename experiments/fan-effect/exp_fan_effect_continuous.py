from cobweb.cobweb_continuous import CobwebContinuousTree
from random import seed, shuffle
import numpy as np
import pandas as pd
from pathlib import Path

concepts = ["c1", "c2", "c3", "c4", "c5"]
facts_by_concept = {
    "c1": ["f1"],
    "c2": ["f2", "f3"],
    "c3": ["f4", "f5", "f6", "f7"],
    # fan of 8
    "c4": [f"f{8 + i}" for i in range(8)],
    # fan of 16
    "c5": [f"f{16 + i}" for i in range(16)],
}
all_facts = [fact for facts in facts_by_concept.values() for fact in facts]


def fact_vec(name):
    vec = np.zeros(len(all_facts))
    vec[all_facts.index(name)] = 1.0
    return vec


def build_train(rs, instances_per_fact=20, spread=0.1):
    rng = np.random.default_rng(rs)
    items = []
    for idx, concept in enumerate(concepts):
        base = np.array([idx * 0.6, idx * 0.4])
        indicator = np.zeros(len(concepts))
        indicator[idx] = 1.0
        for fact in facts_by_concept[concept]:
            for _ in range(instances_per_fact):
                feat = np.concatenate([base + rng.normal(0.0, spread, 2), indicator])
                items.append((feat, fact_vec(fact), concept, fact))
    return items


def build_tests():
    tests = []
    for idx, concept in enumerate(concepts):
        base = np.array([idx * 0.6, idx * 0.4])
        indicator = np.zeros(len(concepts))
        indicator[idx] = 1.0
        for fact in facts_by_concept[concept]:
            feat = np.concatenate([base, indicator])
            tests.append((feat, concept, fact))
    return tests


def run():
    random_seeds = [1, 32, 64, 128, 356]
    blocks = 12
    epochs = 4
    rows = []
    tests = build_tests()

    for rs in random_seeds:
        seed(rs)
        np.random.seed(rs)
        train_items = build_train(rs)

        for epoch in range(1, epochs + 1):
            model = CobwebContinuousTree(2 + len(concepts), len(all_facts), alpha=0.6, prior_var=0.2)
            for block in range(1, blocks + 1):
                shuffle(train_items)
                for feat, label, _, _ in train_items:
                    model.ifit(feat, label)

                for feat, concept, fact in tests:
                    pred = model.predict(feat, np.zeros(len(all_facts)), 30, False)
                    rows.append({
                        "seed": rs,
                        "epoch": epoch,
                        "block": block,
                        "concept": concept,
                        "fan_size": len(facts_by_concept[concept]),
                        "fact": fact,
                        "prob": float(pred[all_facts.index(fact)]),
                    })

    df = pd.DataFrame(rows)
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(results_dir / "exp_fan_effect_continuous.csv"), index=False)


if __name__ == "__main__":
    run()
