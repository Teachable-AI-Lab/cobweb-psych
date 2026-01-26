from cobweb.cobweb_discrete import CobwebDiscreteTree
from random import seed, shuffle
import numpy as np
import pandas as pd
import json
from pathlib import Path

# Fan Effect (Anderson, 1974; Reder & Ross, 1983)
# Goal: Show that retrieval time and errors increase with fan size (number of associations).
#
# Citation: Anderson, J. R. (1974). Retrieval of propositional information from long-term memory.
#           Cognitive Psychology, 6(4), 451-474.

# Random seed for reproducibility
RANDOM_SEED = 12345

persons = ["Hippie", "Captain", "Giant", "Fireman", "Doctor"]
locations = ["Park", "Church", "Bank", "Beach", "Cave", "Dungeon", "Castle", "Forest", "Mountain", 
             "SpaceStation", "Lab", "Cinema", "Mall", "Zoo", "School", "Library", "Stadium", "Airport"]

facts_by_person = {
    "Hippie": ["Park"], # Fan 1
    "Captain": ["Church", "Bank"], # Fan 2
    "Giant": ["Beach", "Cave", "Dungeon", "Castle"], # Fan 4
    "Fireman": locations[0:8], # Fan 8 (reuses some)
    # Note: reusing locations causes interference for location too, which is valid.
    # To isolate person-fan, we use unique locations here (Anderson 1974 used both balanced and random).
}

# Distribute unique locations
all_locs_pool = locations[:]
facts_by_person = {}
idx = 0
for person, fan in [("Hippie", 1), ("Captain", 2), ("Giant", 4), ("Fireman", 8)]:
    facts_by_person[person] = []
    for _ in range(fan):
        if idx < len(all_locs_pool):
            facts_by_person[person].append(all_locs_pool[idx])
            idx += 1
        else:
            # Fallback if we run out (shouldn't with 1+2+4+8 = 15 < 18)
            facts_by_person[person].append("GenericPlace")

all_facts = []
for p, locs in facts_by_person.items():
    for l in locs:
        all_facts.append((p, "in", l))

def make_mappings():
    # We will encode as (Subject, Relation, Object)
    # Attributes: "subject", "relation", "object"
    # Values: All persons, "in", all locations
    
    all_subjects = list(facts_by_person.keys())
    all_objects = [l for locs in facts_by_person.values() for l in locs]
    
    attr_ids = {"subject": 0, "relation": 1, "object": 2}
    value_ids = {
        "subject": {s: i for i, s in enumerate(all_subjects)},
        "relation": {"in": 0},
        "object": {o: i for i, o in enumerate(all_objects)},
    }
    return attr_ids, value_ids


def build_train(rs, instances_per_fact=20):
    # Anderson 1974: Subjects study sentences.
    items = []
    for p, r, l in all_facts:
        for _ in range(instances_per_fact):
            items.append({"subject": p, "relation": r, "object": l})
    return items


def build_tests():
    # Test items: The probes.
    # We query: Subject + Relation -> Object?
    # Or just "Is this sentence true?" (Recognition)
    # Cobweb predicts attributes. We can model recall: Given Subject+Relation, predict Object.
    # Or verification: Rate probability of (S, R, O).
    # Verification is Anderson's task (Reaction time).
    # We'll use Probability of the Fact as proxy for activation / speed.
    return [{"subject": p, "relation": r, "object": l} for p, r, l in all_facts]


def encode_item_discrete(item, attr_ids, value_ids, mask_object=False):
    entry = {}
    
    # Subject
    entry[attr_ids["subject"]] = {value_ids["subject"][item["subject"]]: 1.0}
    
    # Relation
    entry[attr_ids["relation"]] = {value_ids["relation"][item["relation"]]: 1.0}
    
    # Object (included unless masked for prediction task)
    if not mask_object:
        entry[attr_ids["object"]] = {value_ids["object"][item["object"]]: 1.0}
        
    return entry


def run():
    """
    Run the fan effect experiment.
    
    Citation: Anderson, J. R. (1974). Retrieval of propositional information from 
              long-term memory. Cognitive Psychology, 6(4), 451-474.
    
    Stimuli: "Person is in Location" sentences.
    Fan size controlled by number of locations per person.
    """
    random_seeds = [RANDOM_SEED + i * 31 for i in range(5)]
    blocks = 12
    epochs = 4
    rows = []

    attr_ids, value_ids = make_mappings()
    object_attr = attr_ids["object"]

    for rs in random_seeds:
        seed(rs)
        np.random.seed(rs)
        train_raw = build_train(rs, instances_per_fact=20)
        tests_raw = build_tests()

        for epoch in range(1, epochs + 1):
            model = CobwebDiscreteTree(alpha=0.5)
            for block in range(1, blocks + 1):
                shuffle(train_raw)
                train_encoded = [encode_item_discrete(item, attr_ids, value_ids) for item in train_raw]
                model.fit(train_encoded, 1, True)

                # Evaluate: Probability of the correct object given Subject+Relation
                for item in tests_raw:
                    # Query: Subject, Relation. Predict: Object.
                    query_enc = encode_item_discrete(item, attr_ids, value_ids, mask_object=True)
                    target_obj_idx = value_ids["object"][item["object"]]
                    
                    # Predict
                    # Returns dictionary: {attr_id: {val_id: prob}}
                    pred_dict = model.predict(query_enc, 100, False)
                    object_attr_id = attr_ids["object"]
                    
                    # Get probability distribution for object attribute
                    obj_prob_dist = pred_dict.get(object_attr_id, {})
                    prob_true = obj_prob_dist.get(target_obj_idx, 0.0)
                    
                    fan_size = len(facts_by_person[item["subject"]])
                    
                    rows.append({
                        "seed": rs,
                        "epoch": epoch,
                        "block": block,
                        "person": item["subject"],
                        "fan_size": fan_size,
                        "location": item["object"],
                        "prob": prob_true,
                    })

    df = pd.DataFrame(rows)
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(str(results_dir / "exp_fan_effect_discrete.csv"), index=False)
    
    # Calculate fan effect
    final_block = df[df["block"] == blocks]
    fan_summary = final_block.groupby("fan_size", as_index=False)["prob"].mean()
    
    metadata = {
        "experiment": "fan_effect",
        "citation": "Anderson (1974)",
        "fan_sizes": [1, 2, 4, 8],
        "mean_prob_by_fan": fan_summary.to_dict(orient="records"),
        "expected_effect": "Retrieval probability decreases with increasing fan size"
    }
    
    with open(results_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    run()
