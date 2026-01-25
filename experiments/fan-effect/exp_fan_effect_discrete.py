from cobweb.cobweb_discrete import CobwebDiscreteTree
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


def make_mappings():
    attr_ids = {"concept": 0, "fact": 1}
    value_ids = {
        "concept": {c: i for i, c in enumerate(concepts)},
        "fact": {f: i for i, f in enumerate(all_facts)},
    }
    return attr_ids, value_ids


def build_train(rs, instances_per_fact=20):
    rng = np.random.default_rng(rs)
    items = []
    for idx, concept in enumerate(concepts):
        for fact in facts_by_concept[concept]:
            for _ in range(instances_per_fact):
                items.append(({"concept": concept, "fact": fact}, concept, fact))
    return items


def build_tests():
    tests = []
    for concept in concepts:
        for fact in facts_by_concept[concept]:
            tests.append(({"concept": concept}, concept, fact))
    return tests


def encode_item_discrete(item, attr_ids, value_ids, include_label=False):
    entry = {}
    for attr, val in item.items():
        entry[attr_ids[attr]] = {value_ids[attr][val]: 1.0}
    if include_label:
        # include fact label as the 'fact' attribute
        pass
    return entry


def evaluate(model, test_items_encoded, test_targets, category_attr, cat_id_A=None, cat_id_B=None):
    # For fan effect, compute probability assigned to true fact value
    probs = []
    for feat_enc, true_fact_idx in zip(test_items_encoded, test_targets):
        pred = model.predict_probs(feat_enc, 1000, False, False)[1]
        fact_probs = pred.get(category_attr, {})
        prob_true = fact_probs.get(true_fact_idx, 0.0)
        probs.append(float(prob_true))
    return float(np.mean(probs))


def run():
    random_seeds = [1, 32, 64, 128, 356]
    blocks = 12
    epochs = 4
    rows = []

    attr_ids, value_ids = make_mappings()
    category_attr = attr_ids["fact"]

    for rs in random_seeds:
        seed(rs)
        np.random.seed(rs)
        train_items = build_train(rs)
        tests = build_tests()

        test_items_encoded = [encode_item_discrete(t[0], attr_ids, value_ids, include_label=False) for t in tests]
        test_true_idxs = [value_ids["fact"][t[2]] for t in tests]

        for epoch in range(1, epochs + 1):
            model = CobwebDiscreteTree(alpha=0.5)
            for block in range(1, blocks + 1):
                shuffle(train_items)
                train_encoded = [
                    {**encode_item_discrete(feat, attr_ids, value_ids, include_label=False),
                     category_attr: {value_ids["fact"][label]: 1.0}}
                    for feat, _, label in train_items
                ]
                model.fit(train_encoded, 1, True)

                # compute per-test-item probabilities for the true fact
                per_item_probs = []
                for feat_enc, true_idx in zip(test_items_encoded, test_true_idxs):
                    pred = model.predict_probs(feat_enc, 1000, False, False)[1]
                    fact_probs = pred.get(category_attr, {})
                    prob_true = fact_probs.get(true_idx, 0.0)
                    per_item_probs.append(float(prob_true))

                mean_prob = float(np.mean(per_item_probs))
                for i, ((feat_dict, concept, fact), true_idx) in enumerate(zip(tests, test_true_idxs)):
                    rows.append({
                        "seed": rs,
                        "epoch": epoch,
                        "block": block,
                        "concept": concept,
                        "fan_size": len(facts_by_concept[concept]),
                        "fact": fact,
                        "prob": per_item_probs[i],
                        "mean_prob_true_fact": mean_prob,
                    })

    df = pd.DataFrame(rows)
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(results_dir / "exp_fan_effect_discrete.csv"), index=False)


if __name__ == "__main__":
    run()
