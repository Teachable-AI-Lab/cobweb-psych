from cobweb.cobweb_continuous import CobwebContinuousTree
from random import seed, shuffle, gauss, choice
import numpy as np
import pandas as pd
from pathlib import Path
import json

# Category-label / feedback effect (Posner & Keele, 1968; Homa & Cultice, 1984).
# Replication of Homa & Cultice (1984) methodology.
#
# Stimuli: 9-dot random patterns (18 continuous dimensions) in 50x50 grid.
# Categories: 3 (A, B, C) of unequal size (3, 6, 9 exemplars).
# Conditions: Feedback (Corrective vs None) x Distortion (Low, Mixed, High).

RANDOM_SEED = 12345

def generate_prototype(dims=18):
    # 9 dots * 2 coords = 18 dims. 50x50 grid.
    return [np.random.uniform(0, 50) for _ in range(dims)]

def generate_distortion(prototype, level_name):
    # Homa & Cultice (1984) specify 'Average Euclidean Displacement' per dot.
    # For a 2D Gaussian displacement (X, Y ~ N(0, sigma^2)), the Mean Distance (Rayleigh mean) is sigma * sqrt(pi/2) ~= 1.2533 * sigma.
    # Therefore, sigma = Mean_Displacement / 1.2533.
    
    mean_displacement = 0.0
    if level_name == 0 or level_name == "Prototype":
        return list(prototype)
    
    if level_name == 1 or level_name == "Low":
        mean_displacement = 1.10
    elif level_name == 2 or level_name == "Medium":
        mean_displacement = 2.90
    elif level_name == 3 or level_name == "High":
        mean_displacement = 4.80
        
    # Scale sigma
    sigma = mean_displacement / 1.2533
    
    # Apply displacement to each coordinate
    # Note: Text says "displaced... about X units". Assuming isotropic Gaussian noise.
    return [p + gauss(0, sigma) for p in prototype]

def encode_item(vector, category_val=None):
    # vector: list of 18 floats
    # category_val: 0 (A), 1 (B), 2 (C), or None
    feat_arr = np.array(vector, dtype=float)
    label_arr = np.zeros(3) # 3 categories
    
    if category_val is not None and category_val != -1: 
        label_arr[int(category_val)] = 1.0
        
    return feat_arr, label_arr

def score_unsupervised_alignment(leaf_assignments, true_labels):
    """
    Scores unsupervised clustering by mapping each Leaf to its majority True Label.
    Homa's 'Maximizing' scoring logic: "Working backward... maximized for three identifiable categories"
    This approach calculates 'Cluster Purity' relative to the targets.
    
    leaf_assignments: List of leaf node objects (or IDs)
    true_labels: List of int (0, 1, 2)
    """
    if not leaf_assignments: return 0.0
    
    # Map Leaf -> Counter of True Labels
    leaf_map = {}
    for leaf, label in zip(leaf_assignments, true_labels):
        if leaf not in leaf_map:
            leaf_map[leaf] = {0:0, 1:0, 2:0}
        leaf_map[leaf][label] += 1
        
    # Greedily assign Leaf -> Predicted Label
    # Note: Ideally we find optimal permutation (Hungarian Algorithm), 
    # but Homa says "maximized for each trial". A simple majority vote per cluster 
    # approximates the best a subject could identify "This pile is A".
    
    correct = 0
    total = len(true_labels)
    
    for leaf, counts in leaf_map.items():
        # Get the label that this leaf captures most
        # Logic: If a leaf has {A:5, B:1}, we count 5 correct.
        best_label_count = max(counts.values())
        correct += best_label_count
        
    return correct / total

def run():
    # 24 subjects per condition (Paper says 192 total, 8 conditions -> 24/cond)
    random_seeds = [RANDOM_SEED + i * 31 for i in range(24)]
    
    feedback_conditions = [1.0, 0.0] # 1=Feedback, 0=No Feedback
    distortion_conditions = ["Low", "Medium", "High", "Mixed"]
    
    cat_sizes = [3, 6, 9]
    
    epochs = 8 
    rows = []
    
    for subject_idx, rs in enumerate(random_seeds):
        seed(rs)
        np.random.seed(rs)
        
        # 1. Generate Prototypes (A, B, C) and Unrelated (U1, U2)
        # 5 total prototypes. First 3 used for categories, last 2 for unrelated.
        # "Balanced across category size... Latin square"
        # We rotate assignment of prototypes to sizes (3, 6, 9) based on subject index (mod 3).
        all_protos = [generate_prototype() for _ in range(5)]
        
        group_idx = subject_idx % 3
        if group_idx == 0:
            learning_indices = [0, 1, 2]
        elif group_idx == 1:
            learning_indices = [1, 2, 0]
        else:
            learning_indices = [2, 0, 1]
        
        protos = [all_protos[i] for i in learning_indices]
        unrelated_protos = all_protos[3:]
        
        for dist_cond in distortion_conditions:
            for fb_rate in feedback_conditions:
                
                # --- Construct Training Set ---
                training_items = [] # list of {"vector":..., "cat":..., "level":...}
                old_items_by_cat = {0: [], 1: [], 2: []}
                
                for cat_idx, size in enumerate(cat_sizes):
                    p = protos[cat_idx]
                    
                    levels_to_gen = []
                    if dist_cond == "Mixed":
                        # "3-instance cat: 1L, 1M, 1H. 6-inst: 2L, 2M, 2H..."
                        k = size // 3
                        levels_to_gen = ["Low"]*k + ["Medium"]*k + ["High"]*k
                    else:
                        levels_to_gen = [dist_cond] * size
                        
                    for lvl in levels_to_gen:
                        item_vec = generate_distortion(p, lvl)
                        
                        item_obj = {
                            "vector": item_vec, 
                            "cat": cat_idx, 
                            "level": lvl
                        }
                        
                        training_items.append(item_obj)
                        old_items_by_cat[cat_idx].append(item_obj)
                
                # --- Model Training ---
                # Homa Analysis: Feedback vs No Feedback
                model = CobwebContinuousTree(18, 3, alpha=0.1) 
                
                for epoch in range(1, epochs + 1):
                    current_order = list(training_items)
                    shuffle(current_order)
                    
                    # Store for scoring
                    if fb_rate == 0:
                        epoch_leaves = []
                        epoch_truths = []
                    
                    correct_count_fb = 0
                    
                    for item in current_order:
                        # Feed to Model
                        vec = item["vector"]
                        cat = item["cat"]
                        
                        if fb_rate > 0:
                            # Feedback Condition
                            x, y = encode_item(vec, cat)
                            
                            # Predict (Score)
                            pred = model.predict(x, np.zeros(3), 100, True)
                            pred_cat = np.argmax(pred) if np.sum(pred) > 0 else -1
                            if pred_cat == cat: correct_count_fb += 1
                            
                            # Train
                            model.ifit(x, y)
                            
                        else:
                            # No Feedback Condition
                            x, _ = encode_item(vec, None) # No label
                            y_dummy = np.zeros(3)
                            
                            # Predict (Categorize for maximized scoring)
                            # We use .get_leaf() to get the concept node
                            leaf = model.get_leaf(x, y_dummy)
                            epoch_leaves.append(leaf)
                            epoch_truths.append(cat)
                            
                            # Train (Unsupervised)
                            model.ifit(x, y_dummy)
                            
                    # Calculate Accuracy
                    if fb_rate > 0:
                        acc = correct_count_fb / 18.0
                    else:
                        acc = score_unsupervised_alignment(epoch_leaves, epoch_truths)
                        
                    rows.append({
                        "seed": rs,
                        "feedback": "Feedback" if fb_rate > 0 else "No Feedback",
                        "learning_distortion": dist_cond,
                        "phase": "learning",
                        "epoch": epoch,
                        "accuracy": acc,
                        "stim_type": "learning_set"
                    })

                # --- Construct Transfer Set (80 items) ---
                transfer_list = []
                
                # A) 9 Old Patterns
                for c_idx in range(3):
                    olds = old_items_by_cat[c_idx]
                    selected = []
                    
                    if dist_cond == "Mixed":
                        # "Equally represented by L, M, H" -> 1 of each
                        by_level = {"Low": [], "Medium": [], "High": []}
                        for o in olds: by_level[o["level"]].append(o)
                        
                        # Grab 1 from each
                        try:
                            # Homa says "drawn equally". Since we generated balanced, this works.
                            # Use random choice if multiple exist
                            selected.append(choice(by_level["Low"]))
                            selected.append(choice(by_level["Medium"]))
                            selected.append(choice(by_level["High"]))
                        except IndexError:
                            # Fallback if generation failed logic (shouldn't happen)
                            selected = olds[:3]
                    else:
                        # Drawn equally from available (all same level)
                        shuffle(olds)
                        selected = olds[:3]
                        
                    for s in selected:
                        transfer_list.append((s["vector"], c_idx, "Old"))
                        
                # B) 6 Prototypes (2 copies of each)
                for c_idx in range(3):
                    p = protos[c_idx]
                    transfer_list.append((p, c_idx, "Prototype"))
                    transfer_list.append((p, c_idx, "Prototype"))
                    
                # C) 45 New Patterns (15/cat: 5L, 5M, 5H)
                for c_idx in range(3):
                    p = protos[c_idx]
                    for _ in range(5): transfer_list.append((generate_distortion(p, "Low"), c_idx, "New_Low"))
                    for _ in range(5): transfer_list.append((generate_distortion(p, "Medium"), c_idx, "New_Medium"))
                    for _ in range(5): transfer_list.append((generate_distortion(p, "High"), c_idx, "New_High"))

                # D) 20 Unrelated Patterns (10 from U1, 10 from U2)
                # "10 each... 6 were low level, 6 were medium level, and 8 were high-level" (Total 20)
                # Wait, "10 each" usually implies symmetric composition. 
                # If total is 20 and counts are 6,6,8, split is 3,3,4 per proto.
                for u_idx in range(2):
                    p = unrelated_protos[u_idx]
                    for _ in range(3): transfer_list.append((generate_distortion(p, "Low"), -1, "Unrelated"))
                    for _ in range(3): transfer_list.append((generate_distortion(p, "Medium"), -1, "Unrelated"))
                    for _ in range(4): transfer_list.append((generate_distortion(p, "High"), -1, "Unrelated"))
                    
                shuffle(transfer_list)
                
                # --- Transfer Evaluation ---
                # Homa scoring: Maximize number correct given three identifiable categories.
                # For Feedback subjects: They rely on learned labels A/B/C.
                # For No-Feedback: We should technically do the maximizing again?
                # Homa: "If subject appeared to switch... transfer was scored to maximize."
                # As Cobweb is a consistent model, for No-Feed we continue using Unsupervised Scoring?
                # However, usually we test if the concepts align with True categories.
                # Let's use the same Predict vs Unsupervised logic.
                
                if fb_rate > 0:
                    # Supervised Prediction
                    for vec, true_cat, s_type in transfer_list:
                        x, _ = encode_item(vec, None)
                        pred = model.predict(x, np.zeros(3), 100, False)
                        pred_cat = np.argmax(pred)
                        
                        is_correct = 0
                        if true_cat != -1 and pred_cat == true_cat:
                            is_correct = 1
                            
                        rows.append({
                            "seed": rs,
                            "feedback": "Feedback",
                            "learning_distortion": dist_cond,
                            "phase": "transfer",
                            "epoch": epochs + 1,
                            "accuracy": is_correct,
                            "stim_type": s_type
                        })
                else:
                    # Unsupervised Scoring for Transfer
                    # Collect all transfer items, cluster them, then score purity?
                    # Homa implies the "consistency" is from learning to transfer.
                    # We will treat the transfer simply as: Do the transfer items cluster 
                    # into the nodes established for A, B, C?
                    
                    # Store all assignments
                    trans_leaves = []
                    trans_truths = []
                    row_metadata = []
                    
                    for vec, true_cat, s_type in transfer_list:
                        x, _ = encode_item(vec, None)
                        # We do NOT train on transfer
                        y_dummy = np.zeros(3)
                        leaf = model.get_leaf(x, y_dummy)
                        
                        trans_leaves.append(leaf)
                        trans_truths.append(true_cat)
                        row_metadata.append(s_type)

                    # We need to assign labels to leaves based on... what?
                    # The text: "Scored to maximize number correct... if subject appeared to switch use of category labels".
                    # This implies optimizing the map on the Transfer Set itself? 
                    # Or using the map from Learning?
                    # Usually "Transfer performance scored to maximize" means optimizing the mapping on the transfer block.
                    # So we call score_unsupervised_alignment on the transfer set data.
                    # Note: Unrelated items (-1) should arguably be excluded from the purity map optimization 
                    # OR treated as a 4th category? "Three identifiable categories were required".
                    # So we filter out -1 for the mapping calculation.
                    
                    # Filter for scoring (Only A/B/C items)
                    valid_indices = [i for i, t in enumerate(trans_truths) if t != -1]
                    valid_leaves = [trans_leaves[i] for i in valid_indices]
                    valid_truths = [trans_truths[i] for i in valid_indices]
                    
                    # Establish Mapping (Greedy Majority Vote)
                    leaf_to_label = {}
                    leaf_counts = {}
                    for lf, lab in zip(valid_leaves, valid_truths):
                        if lf not in leaf_counts: leaf_counts[lf] = {0:0, 1:0, 2:0}
                        leaf_counts[lf][lab] += 1
                        
                    for lf, counts in leaf_counts.items():
                        leaf_to_label[lf] = max(counts, key=counts.get)
                        
                    # Now score ALL items based on this map (ignoring unrelated for "Accuracy" stats usually)
                    for i, s_type in enumerate(row_metadata):
                        true_cat = trans_truths[i]
                        leaf = trans_leaves[i]
                        
                        # Predicted Label
                        pred_cat = leaf_to_label.get(leaf, -1) # -1 if leaf only contained noise or empty?
                        
                        is_correct = 0
                        if true_cat != -1:
                            if pred_cat == true_cat:
                                is_correct = 1
                        # Save
                        rows.append({
                            "seed": rs,
                            "feedback": "No Feedback",
                            "learning_distortion": dist_cond,
                            "phase": "transfer",
                            "epoch": epochs + 1,
                            "accuracy": is_correct,
                            "stim_type": s_type
                        })

    df = pd.DataFrame(rows)
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(results_dir / "exp_category_label_feedback_continuous.csv"), index=False)

    metadata = {
        "experiment": "Category Label Feedback (Homa & Cultice 1984)",
        "revisions": "Adjusted sigma for mean displacement; Implemented Maximized Scoring for No-Feedback; N=24/cond; Latin Square Proto Balancing.",
    }
    with open(results_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    run()
