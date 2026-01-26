from cobweb.cobweb_continuous import CobwebContinuousTree
from random import seed, shuffle, gauss
import numpy as np
import pandas as pd
from pathlib import Path
import json

# Category-label / feedback effect (Posner & Keele, 1968; Homa & Cultice, 1984).
# Stimuli: 9-dot random patterns (18 continuous dimensions).
# Effect: Labeled training (feedback) improves prototype learning vs unsupervised.
# 
# Citation: Posner, M. I., & Keele, S. W. (1968). On the genesis of abstract ideas.
#           JEP, 77(3, Pt.1), 353-363.

RANDOM_SEED = 12345

def generate_prototype(dims=18):
    # Determine bounds 0-100?
    return [np.random.uniform(0, 100) for _ in range(dims)]

def generate_distortion(prototype, level):
    # Level is standard deviation of noise or similar
    # Posner & Keele levels: 7.7, 9.7 bit distortions.
    # We'll use Gaussian noise sigma.
    # Level 1 (Low): sigma=5
    # Level 2 (High): sigma=20
    if level == 0:
        return list(prototype)
    return [p + gauss(0, level) for p in prototype]

def encode_item(vector, category_val=None):
    # vector: list of 18 floats in 0-100 range
    # category_val: 0.0 (A) or 1.0 (B), or None
    feat_arr = np.array(vector, dtype=float)
    
    if category_val is not None:
        if category_val == 0.0:
            label_arr = np.array([1.0, 0.0]) # A
        else:
            label_arr = np.array([0.0, 1.0]) # B
    else:
        label_arr = np.zeros(2) # Unknown/Unsupervised? Or should pass null?
        # If unsupervised, we probably shouldn't be confident in label.
        # But ifit requires 2 args. 
        # In feedback=0.0 case (unsupervised), maybe we just don't pass label info 
        # but the API forces a vector?
        # If I pass [0, 0], it might mean "don't know".
        pass
        
    return feat_arr, label_arr

def run():
    random_seeds = [RANDOM_SEED + i * 31 for i in range(5)]
    # Label rates: 1.0 (Supervised), 0.0 (Unsupervised)
    label_rates = [1.0, 0.0] 
    
    # 2 Categories: A and B
    # Prototypes
    
    epochs = 4
    blocks = 1
    rows = []
    
    # Define distortion sigmas
    dist_levels = {"Prototype": 0, "Low": 10, "High": 30}
    
    for rs in random_seeds:
        seed(rs)
        np.random.seed(rs)
        
        proto_A = generate_prototype()
        proto_B = generate_prototype()
        
        # Train set: 10 Low distortion items per category
        train_A = [generate_distortion(proto_A, dist_levels["Low"]) for _ in range(10)]
        train_B = [generate_distortion(proto_B, dist_levels["Low"]) for _ in range(10)]
        
        # Test set: Prototype, Low, High items
        tests = []
        # A items
        tests.append((proto_A, "A", "Prototype"))
        tests.append((generate_distortion(proto_A, dist_levels["Low"]), "A", "Low"))
        tests.append((generate_distortion(proto_A, dist_levels["High"]), "A", "High"))
        # B items
        tests.append((proto_B, "B", "Prototype"))
        tests.append((generate_distortion(proto_B, dist_levels["Low"]), "B", "Low"))
        tests.append((generate_distortion(proto_B, dist_levels["High"]), "B", "High"))
        
        for rate in label_rates:
            # 18 dims, 2 labels
            model = CobwebContinuousTree(18, 2, alpha=0.5)
            
            # Prepare training list
            training_data = []
            for item in train_A:
                training_data.append((item, 0.0))
            for item in train_B:
                training_data.append((item, 1.0))
            shuffle(training_data)
            
            for epoch in range(1, epochs+1):
                # Train
                for item, cat_val in training_data:
                    # Apply label rate
                    if np.random.rand() < rate:
                        x, y = encode_item(item, cat_val)
                    else:
                        # Unsupervised: Pass None to get zero vector or similar?
                        # Or maybe we just pass feature vector and "missing" label?
                        # Using [0, 0] as "no label" signal.
                        x, _ = encode_item(item, cat_val)
                        y = np.zeros(2)
                        
                    model.ifit(x, y)
                    
                # Evaluate
                for vec, label_str, dist_name in tests:
                    # Predict category 
                    x, _ = encode_item(vec, None)
                    prediction = model.predict(x, np.zeros(2), 100, False)
                    
                    prob_A = float(prediction[0])
                    prob_B = float(prediction[1])
                    
                    target = 0.0 if label_str == "A" else 1.0
                    acc = 1 if (prob_B > 0.5 and target == 1.0) or (prob_A > 0.5 and target == 0.0) else 0
                    
                    rows.append({
                        "seed": rs,
                        "label_rate": rate,
                        "distortion": dist_name,
                        "true_category": label_str,
                        "prob_A": prob_A,
                        "prob_B": prob_B,
                        "accuracy": acc,
                        "epoch": epoch
                    })
                    
    df = pd.DataFrame(rows)
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(results_dir / "exp_category_label_feedback_continuous.csv"), index=False)

    metadata = {
        "experiment": "category_label_feedback_continuous",
        "description": "Posner & Keele (1968) random dot patterns",
        "expected_effect": "Feedback (1.0 label rate) improves accuracy and prototype abstraction."
    }
    with open(results_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    run()
