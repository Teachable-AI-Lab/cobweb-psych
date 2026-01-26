from random import seed, shuffle
from pathlib import Path
import numpy as np
import pandas as pd
import json
from cobweb.cobweb_discrete import CobwebDiscreteTree

# Category-label / feedback effect (Posner & Keele, 1968; Homa & Cultice, 1984).
# Goal: Show that labeled/feedback training accelerates prototype learning and increases 
# prototype endorsement compared to unsupervised exposure.
#
# Citation: Posner, M. I., & Keele, S. W. (1968). On the genesis of abstract ideas.
#           Journal of Experimental Psychology, 77(3, Pt.1), 353–363.

# Random seed for reproducibility
RANDOM_SEED = 12345

attribute_values = {
    "shape": ["circle", "triangle"],
    "color": ["red", "blue"],
    "fill": ["solid", "striped"],
    "category": ["A", "B"],
}


def make_mappings():
    """Create attribute and value ID mappings for Cobweb discrete encoding."""
    attr_ids = {name: idx for idx, name in enumerate(attribute_values.keys())}
    value_ids = {}
    for attr, vals in attribute_values.items():
        value_ids[attr] = {v: i for i, v in enumerate(vals)}
    return attr_ids, value_ids


def sample_features(cat: str, rng: np.random.Generator, noise: float = 0.1):
    """
    Sample features from a category prototype with distortion.
    
    Posner & Keele (1968) style: prototype + noise distortions.
    
    Args:
        cat: Category ("A" or "B")
        rng: Random number generator
        noise: Probability of flipping each feature (distortion level)
    
    Returns:
        Dictionary of feature values
    """
    # Define category prototypes
    prototype = {
        "A": {"shape": "circle", "color": "red", "fill": "solid"},
        "B": {"shape": "triangle", "color": "blue", "fill": "striped"},
    }[cat]
    
    features = {}
    for attr in ["shape", "color", "fill"]:
        if rng.random() < noise:
            # Flip to opposite value (distortion)
            options = attribute_values[attr]
            features[attr] = options[1] if prototype[attr] == options[0] else options[0]
        else:
            features[attr] = prototype[attr]
    return features


def build_dataset(rs: int, n_per_class: int = 150, distortion_levels=(0.0, 0.05, 0.15, 0.30)):
    """
    Build training/test datasets with multiple distortion levels.
    
    Posner & Keele (1968): Prototype (0.0), Low (0.05), Medium (0.15), High (0.30) distortion.
    
    Args:
        rs: Random seed
        n_per_class: Number of exemplars per category per distortion level
        distortion_levels: Tuple of noise probabilities (Posner & Keele style)
    
    Returns:
        Dictionary mapping distortion level to list of items
    """
    rng = np.random.default_rng(rs)
    data_by_dist = {}
    for dist in distortion_levels:
        items = []
        for cat in ["A", "B"]:
            for _ in range(n_per_class):
                items.append({**sample_features(cat, rng, dist), "category": cat})
        data_by_dist[dist] = items
    return data_by_dist


def encode_item(raw_item, attr_ids, value_ids, include_label: bool):
    """
    Encode item for Cobweb discrete tree.
    
    Args:
        raw_item: Dictionary with feature values and category
        attr_ids: Attribute ID mapping
        value_ids: Value ID mapping
        include_label: Whether to include category label (supervised vs unsupervised)
    
    Returns:
        Encoded item for Cobweb
    """
    entry = {}
    for attr in ["shape", "color", "fill"]:
        entry[attr_ids[attr]] = {value_ids[attr][raw_item[attr]]: 1.0}
    if include_label:
        entry[attr_ids["category"]] = {value_ids["category"][raw_item["category"]]: 1.0}
    return entry


def evaluate(model, test_items, test_targets, category_attr, cat_id_A, cat_id_B):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained Cobweb model
        test_items: Encoded test items
        test_targets: True category labels
        category_attr: Category attribute ID
        cat_id_A: Category A value ID
        cat_id_B: Category B value ID
    
    Returns:
        Tuple of (accuracy, mean_prob_B)
    """
    correct = 0
    share_b = []
    for feat, target in zip(test_items, test_targets):
        pred = model.predict(feat, 100, True)[1]
        prob_a = pred.get(cat_id_A, 0.0)
        prob_b = pred.get(cat_id_B, 0.0)
        pred_label = "A" if prob_a >= prob_b else "B"
        if pred_label == target:
            correct += 1
        share_b.append(prob_b)
    return correct / len(test_items), float(np.mean(share_b))


def run():
    """
    Run the category-label / feedback effect experiment.
    
    Citation: Posner, M. I., & Keele, S. W. (1968). On the genesis of abstract ideas.
              Journal of Experimental Psychology, 77(3, Pt.1), 353–363.
    
    Expected result: Higher label rates (feedback) should produce better classification 
    accuracy and stronger prototype endorsement, especially on high-distortion test items.
    """
    random_seeds = [RANDOM_SEED + i * 31 for i in range(5)]  # 5 seeds for robustness
    label_rates = [1.0, 0.75, 0.5, 0.25, 0.1, 0.0]  # Feedback manipulation
    distortion_levels = (0.0, 0.05, 0.15, 0.30)  # Posner & Keele style
    blocks = 12
    epochs = 3
    rows = []

    attr_ids, value_ids = make_mappings()
    category_attr = attr_ids["category"]
    cat_id_A = value_ids["category"]["A"]
    cat_id_B = value_ids["category"]["B"]

    for rs in random_seeds:
        seed(rs)
        np.random.seed(rs)
        data_by_dist = build_dataset(rs, n_per_class=1, distortion_levels=distortion_levels)

        # Build test sets per distortion level
        tests_by_dist = {
            dist: ([encode_item(item, attr_ids, value_ids, include_label=False) for item in items], 
                   [item["category"] for item in items])
            for dist, items in data_by_dist.items()
        }

        # Flatten training data across distortion levels (balanced sampling from all distortions)
        flat_data = [(item, dist) for dist, items in data_by_dist.items() for item in items]

        for label_rate in label_rates:
            for epoch in range(1, epochs + 1):
                model = CobwebDiscreteTree(alpha=0.5)
                for block in range(1, blocks + 1):
                    shuffle(flat_data)
                    train_items = []
                    for item, _dist in flat_data:
                        include_label = np.random.rand() < label_rate
                        train_items.append(encode_item(item, attr_ids, value_ids, include_label))
                    for train_item in train_items:
                        print(train_item)
                        model.ifit(train_item)

                    # Evaluate per distortion level
                    for dist, (test_items, test_targets) in tests_by_dist.items():
                        acc, mean_prob_b = evaluate(model, test_items, test_targets, category_attr, cat_id_A, cat_id_B)
                        rows.append(
                            {
                                "seed": rs,
                                "epoch": epoch,
                                "block": block,
                                "label_rate": label_rate,
                                "distortion": dist,
                                "accuracy": acc,
                                "pred_share_B": mean_prob_b,
                            }
                        )

    df = pd.DataFrame(rows)
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV results
    df.to_csv(str(results_dir / "exp_category_label_feedback_discrete.csv"), index=False)
    
    # Save experiment metadata
    metadata = {
        "experiment": "category_label_feedback_effect",
        "citation": "Posner, M. I., & Keele, S. W. (1968). On the genesis of abstract ideas. JEP, 77(3, Pt.1), 353-363.",
        "random_seed": RANDOM_SEED,
        "seeds_used": random_seeds,
        "label_rates": label_rates,
        "distortion_levels": list(distortion_levels),
        "num_blocks": blocks,
        "num_epochs": epochs,
        "expected_effect": "Higher label rates (feedback) produce better accuracy and prototype endorsement"
    }
    
    with open(results_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    run()
