from cobweb.cobweb_continuous import CobwebContinuousTree
from random import shuffle, seed
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from pathlib import Path

# Medin & Schaffer (1978) - Context Theory / Central Tendency
# Updated to use CobwebContinuousTree

class MedinDataset(object):
    def __init__(self, random_seed=None, stimulus_type='geometric'):
        
        # Dimensions and Values
        if stimulus_type == 'geometric':
            # Binary features mapped to 0/1 float
            # Dims: Color, Form, Size, Number
            self.attribute_values = {
                'color': ['red', 'blue'],
                'form': ['triangle', 'circle'],
                'size': ['large', 'small'],
                'number': [1, 2],
            }
        elif stimulus_type == 'face':
            # Dims: eye height, eye separation, nose length, mouth height
            # Values are discrete but numeric: 
            # eye height: 2.5 vs 5.0
            # eye sep: 1.5 vs 3.5
            # nose len: 1.5 vs 3.0
            # mouth height: 1.5 vs 3.0
            self.attribute_values = {
                'eye height': [2.5, 5], 
                'eye separation': [1.5, 3.5],
                'nose length': [1.5, 3.0],
                'mouth height': [1.5, 3.0],
            }
            
        self.stimulus_type = stimulus_type
        
        # Stimuli Definitions (Medin & Schaffer 1978)
        # Format: [Stimulus ID, Dim1(0/1), Dim2(0/1), Dim3(0/1), Dim4(0/1), Class(A/B)]
        # Note: Values are indices into the attribute_values lists above.
        
        training_stimuli_ls = [
            [4, 1, 1, 1, 0, 'A'], 
            [7, 1, 0, 1, 0, 'A'],
            [15, 1, 0, 1, 1, 'A'], 
            [13, 1, 1, 0, 1, 'A'],
            [5, 0, 1, 1, 1, 'A'], 
            [12, 1, 1, 0, 0, 'B'],
            [2, 0, 1, 1, 0, 'B'], 
            [14, 0, 0, 0, 1, 'B'],
            [10, 0, 0, 0, 0, 'B']
        ]
        
        transfer_stimuli_ls = [
            [1, 1, 0, 0, 1, 'A'], 
            [3, 1, 0, 0, 0, 'B'],
            [6, 1, 1, 1, 1, 'A'], 
            [8, 0, 0, 1, 0, 'B'],
            [9, 0, 1, 0, 1, 'A'], 
            [11, 0, 0, 1, 1, 'A'],
            [16, 0, 1, 0, 0, 'B']
        ]
        
        test_stimuli_ls = training_stimuli_ls + transfer_stimuli_ls
        
        if random_seed is not None:
            seed(random_seed)
            np.random.seed(random_seed)
            
        # Convert to continuous vector format
        # vector = [val1, val2, val3, val4]
        # label = [1, 0] if A, [0, 1] if B
        
        self.train_objs = self.encode_list(training_stimuli_ls)
        self.test_objs = self.encode_list(test_stimuli_ls)
        
        shuffle(self.train_objs)

    def encode_list(self, lss):
        objs = []
        for row in lss:
            stim_id = row[0]
            indices = row[1:5] # 4 dimensional indices
            cls = row[5]
            
            vec = []
            keys = list(self.attribute_values.keys())
            
            for i, idx in enumerate(indices):
                dim_name = keys[i]
                # Map 0/1 index to actual value
                val = self.attribute_values[dim_name][idx]
                # For geometric, 'red' isn't float. We map indices to 0.0, 1.0 directly or use continuous val if numeric.
                # If geometric: map to 0.0 or 1.0 (arbitrary numeric coding)
                if self.stimulus_type == 'geometric':
                    vec.append(float(idx))
                else: 
                    # Face: values are already numeric floats
                    vec.append(float(val))
                    
            # Label
            label_vec = np.zeros(2)
            if cls == 'A':
                label_vec[0] = 1.0
            else:
                label_vec[1] = 1.0
                
            objs.append({
                "id": stim_id,
                "x": np.array(vec, dtype=float),
                "y": label_vec,
                "cls_str": cls
            })
            
        return objs

class Experiment(object):
    def __init__(self, random_seeds=[1, 32, 64, 128, 356], epochs=5, stimulus_type='geometric', rank_method='ordinal'):
        if stimulus_type == 'geometric':
            self.observed_probs = [0.59, 0.84, 0.69, 0.78, 0.81,
                                   0.94, 0.88, 0.66, 0.5, 0.97, 0.62,
                                   0.84, 0.88, 0.88, 0.81, 0.84]
        elif stimulus_type == 'face':
            self.observed_probs = [0.72, 0.72, 0.44, 0.97, 0.72,
                                   0.98, 0.97, 0.77, 0.27, 0.95, 0.39,
                                   0.67, 0.81, 0.97, 0.92, 0.91]
                                   
        self.stimulus_type = stimulus_type
        self.rank_method = rank_method
        self.observed_ranking = rankdata(self.observed_probs, method=rank_method)
        self.random_seeds = random_seeds
        self.epochs = epochs
        
        self.df = self.run_batch()
        
    def run_batch(self):
        all_rows = []
        for s in self.random_seeds:
            for e in range(self.epochs):
                rows = self.run_iteration(s, e + 1)
                all_rows.extend(rows)
        return pd.DataFrame(all_rows)

    def run_iteration(self, seed_val, iteration):
        # 4 dimensions, 2 classes
        # Alpha adjusted for continuous
        tree = CobwebContinuousTree(4, 2, alpha=0.5)
        
        dataset = MedinDataset(random_seed=seed_val, stimulus_type=self.stimulus_type)
        
        # Train
        train_objs = dataset.train_objs
        for item in train_objs:
            tree.ifit(item['x'], item['y'])
            
        # Test
        rows = []
        test_objs = dataset.test_objs
        
        # We need to map prediction back to Stimulus ID order 1..16
        # The test_objs list has them but not in 1..16 order potentially?
        # MedinDataset constructs them in a specific order then shuffles unless we suppressed it?
        # Actually dataset shuffles self.test_objs in constructor? No, wait.
        # "test_stimuli_complete" was shuffled in original code.
        # We will iterate and store, then sort by ID later if needed, or just store ID.
        
        pred_probs = {} # ID -> Prob(Class A)
        
        for item in test_objs:
            # Predict
            # predict(x, empty_y, max_nodes, greedy)
            start_t = time.time()
            probs = tree.predict(item['x'], np.zeros(2), 200, False)
            end_t = time.time()
            
            p_A = probs[0]
            
            # Map ground truth class to 0 or 1 index
            gt_idx = 0 if item['cls_str'] == 'A' else 1
            
            # Probability assigned to the Correct Class
            p_correct = probs[gt_idx]
            
            rows.append({
                "stimulus": item['id'],
                "seed": seed_val,
                "iteration": iteration,
                "predicted_prob_correct": p_correct,
                "reaction_time": (end_t - start_t) * 1000
            })
            
            pred_probs[item['id']] = p_correct
            
        # Add correlation metrics for this run
        # Need vector of preds aligned with observed_probs (which are ordered by ID 1..16)
        pred_vec = []
        for i in range(1, 17):
            if i in pred_probs:
                pred_vec.append(pred_probs[i])
            else:
                pred_vec.append(0.0) # Should not happen if all 16 are present
                
        corr_prob = np.corrcoef(self.observed_probs, pred_vec)[0, 1]
        
        # Add correlation to rows
        for r in rows:
            r["correlation_prob"] = corr_prob
            r["observed_prob"] = self.observed_probs[r["stimulus"]-1]
            
        return rows
        
    def save_csv(self):
        results_dir = Path(__file__).resolve().parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        fname = f"exp_medin_continuous_{self.stimulus_type}.csv"
        self.df.to_csv(str(results_dir / fname), index=False)
        print(f"Saved results to {results_dir / fname}")
        
    def print_stats(self):
        mean_corr = self.df.groupby(["seed", "iteration"])["correlation_prob"].mean().mean()
        print(f"Mean Correlation: {mean_corr:.4f}")

# ===============================================

if __name__ == "__main__":
    # seeds = [int(i) for i in np.arange(0, 1000, 50)]
    seeds = [0, 42, 123, 999, 2026]
    
    # Run
    exp = Experiment(random_seeds=seeds, epochs=5, stimulus_type='face') 
    exp.save_csv()
    exp.print_stats()

