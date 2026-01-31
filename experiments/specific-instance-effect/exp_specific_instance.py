from cobweb.cobweb_discrete import CobwebDiscreteTree
from random import seed, shuffle, randint
import numpy as np
import pandas as pd
from pathlib import Path
import json

# Specific-Instance / Exemplar-Strength Effect
# Replicating Hayes-Roth & Hayes-Roth (1977) Design
#
# Reference: Hayes-Roth, B., & Hayes-Roth, F. (1977). Concept learning and the 
#            recognition and classification of exemplars. 
#            Journal of Verbal Learning and Verbal Behavior.
#
# Stimuli Structure:
# - 3 Relevant Dimensions (Age, Education, Marital Status)
# - 2 Irrelevant/Distractor Dimensions (Surname, Hobby)
# - Values: 1-4 (Discrete)
# - Categories: Club 1 (Proto 111), Club 2 (Proto 222)

def generate_hayes_roth_stimuli(rng_seed):
    """
    Generates the specific stimulus set with Hayes-Roth properties.
    Includes 3 relevant features and 2 irrelevant distractors.
    
    Prototypes:
    - Class 1: 1 1 1 (never shown or rare)
    - Class 2: 2 2 2 (never shown or rare)
    
    Manipulation:
    - Orthogonal manipulation of Distance-to-Prototype vs Frequency.
    """
    
    # Local RNG for stability of irrelevant features per seed
    local_rng = np.random.RandomState(rng_seed)
    
    def get_distractors():
        # Generates 2 random values for the irrelevant dimensions (1-4)
        return [local_rng.randint(1, 5), local_rng.randint(1, 5)]

    # We define the specific logical forms from the study
    # Structure: [Rel1, Rel2, Rel3]
    
    exemplars = []
    
    # --- Category 1 (Club 1) ---
    # Center: 1 1 1
    
    # 1. The Prototype (Distance 0) - Never Shown
    exemplars.append({
        "relevant": [1, 1, 1], "class": 1, "id": "A_Proto", "freq": 0, "type": "prototype",
        "irrelevant": get_distractors()
    })
    
    # 2. Frequent Exemplar (Distance 1) - Shown Often (e.g. 10 times)
    # Transformation: 112
    exemplars.append({
        "relevant": [1, 1, 2], "class": 1, "id": "A_Freq_Dist1", "freq": 10, "type": "frequent",
        "irrelevant": get_distractors()
    })
    
    # 3. Rare Exemplar (Distance 1) - Shown Once
    # Transformation: 121 (Equidistant to 111 as 112 is)
    exemplars.append({
        "relevant": [1, 2, 1], "class": 1, "id": "A_Rare_Dist1", "freq": 1, "type": "rare",
        "irrelevant": get_distractors()
    })
    
    # 4. Context Fillers (Distance 1 or 2)
    # 211
    exemplars.append({
        "relevant": [2, 1, 1], "class": 1, "id": "A_Fill1", "freq": 2, "type": "filler",
        "irrelevant": get_distractors()
    })
    # 131 (Dist 2)
    exemplars.append({
        "relevant": [1, 3, 1], "class": 1, "id": "A_Fill2", "freq": 1, "type": "filler",
        "irrelevant": get_distractors()
    })
    
    # --- Category 2 (Club 2) ---
    # Center: 2 2 2
    
    # Prototype B (Distance 0) - Never Shown
    exemplars.append({
        "relevant": [2, 2, 2], "class": 2, "id": "B_Proto", "freq": 0, "type": "prototype",
        "irrelevant": get_distractors()
    })
    
    # Distinctive B Exemplars
    # 221 (Dist 1)
    exemplars.append({
        "relevant": [2, 2, 1], "class": 2, "id": "B_1", "freq": 5, "type": "filler",
        "irrelevant": get_distractors()
    })
    # 212 (Dist 1)
    exemplars.append({
        "relevant": [2, 1, 2], "class": 2, "id": "B_2", "freq": 5, "type": "filler",
        "irrelevant": get_distractors()
    })
    # 322 (Dist 1)
    exemplars.append({
        "relevant": [3, 2, 2], "class": 2, "id": "B_3", "freq": 5, "type": "filler",
        "irrelevant": get_distractors()
    })
    
    return exemplars

def encode_full_item(item, include_class=True):
    """
    Encodes item including relevant and irrelevant features.
    Map:
    1: Rel1
    2: Rel2
    3: Rel3
    4: Irr1
    5: Irr2
    6: Class
    """
    encoded = {}
    
    # Relevant Features
    for i, val in enumerate(item["relevant"]):
        encoded[i+1] = {val: 1.0}
        
    # Irrelevant Features
    for i, val in enumerate(item["irrelevant"]):
        encoded[i+4] = {val: 1.0}
        
    # Class
    if include_class:
        encoded[6] = {item["class"]: 1.0}
        
    return encoded

def execute_hayes_roth_simulation(number_of_seeds=20):
    results = []
    
    for s in range(number_of_seeds):
        seed(s)
        np.random.seed(s)
        
        # Generated Stimuli for this subject/seed
        stimuli = generate_hayes_roth_stimuli(s)
        
        # 1. Build Training List
        training_sequence = []
        for item in stimuli:
            count = item["freq"]
            for _ in range(count):
                training_sequence.append(item)
                
        # 2. Learning Phase
        tree = CobwebDiscreteTree(alpha=0.25)
        
        # Train for multiple epochs to simulate learning to criterion
        epochs = 3 
        for e in range(epochs):
            shuffle(training_sequence)
            for item in training_sequence:
                encoded = encode_full_item(item, include_class=True)
                tree.fit([encoded])
                
        # 3. Test Phase
        # We test on Specific Items (Old and New)
        # Specifically comparing A_Freq (Old, Freq=High) vs A_Rare (Old, Freq=Low) vs A_Proto (New, Freq=0)
        
        target_ids = ["A_Freq_Dist1", "A_Rare_Dist1", "A_Proto"]
        
        for item in stimuli:
            if item["id"] not in target_ids:
                continue
                
            # -- Recognition Test --
            # "Old/New" judgment. Modeled as instance familiarity (log likelihood of the features).
            # We treat the instance (features+distractors) as the probe.
            # Usually recognition probes don't have the category label visible? 
            # In H-R 77, recognition is "Have you seen this person?". 
            # So we query P(Features).
            
            # Note: For Cobweb, log_prob usually calculates joint probability of the leaf.
            # We'll use the log_prob of the *entire* instance (minus class maybe?).
            # The class label is part of the concept 'Person', but in recognition test, 
            # strict recognition is usually feature-based. However, H-R participants learned names/clubs.
            # We'll stick to Feature Likelihood.
            
            test_instance_features = encode_full_item(item, include_class=False)
            recognition_score = tree.log_prob(test_instance_features, 100, False)
            
            # -- Classification Test --
            # "Which Club?". Predict Class.
            
            probs = tree.predict(test_instance_features, 100, False)
            
            class_prob = 0.0
            correct_class = item["class"]
            if 6 in probs and correct_class in probs[6]:
                class_prob = probs[6][correct_class]
            
            results.append({
                "seed": s,
                "stimulus_id": item["id"],
                "type": item["type"],
                "frequency": item["freq"],
                "recognition_score": recognition_score, # Log Likelihood
                "classification_accuracy": class_prob   # Probability of Correct Class
            })
            
    return results

def run_and_save_results():
    results = execute_hayes_roth_simulation(number_of_seeds=30)
    df = pd.DataFrame(results)
    
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp_specific_instance_discrete.csv"
    
    df.to_csv(out_path, index=False)
    print(f"Hayes-Roth Simulation Results saved to {out_path}")
    
    metadata = {
            "experiment": "Hayes-Roth & Hayes-Roth (1977) Replication",
            "description": "Specific Instance Effect using 5-feature discrete stimuli.",
            "manipulation": "Orthogonal Distance (from Prototype) vs Frequency.",
            "dimensions": "3 Relevant + 2 Irrelevant (Distractors)",
            "key_comparison": "A_Freq (Dist 1, High Freq) vs A_Proto (Dist 0, Zero Freq)"
        }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    run_and_save_results()
