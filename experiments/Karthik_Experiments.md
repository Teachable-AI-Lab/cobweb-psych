# Cobweb Psychological Categorization Effects: Summary Report

This repository contains implementations of canonical human categorization effects using the Cobweb incremental clustering algorithm. Each experiment replicates seminal findings from cognitive psychology to validate Cobweb's psychological plausibility as a model of human category learning.

## Overview

**Random Seed:** 12345 (used across all experiments for reproducibility)

**Model:** CobwebDiscreteTree and CobwebContinuousTree (from cobweb library)

**Language:** Python 3.10+

**Dependencies:** numpy, pandas, matplotlib, seaborn, scikit-learn

---

## Experiments Implemented

### 1. Specific-Instance / Frequency Bias Effect

**Location:** `experiments/specific-instance-effect/`

**Citation:** Nosofsky, R. M. (1988). Similarity, frequency, and category representations. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 14*(1), 54-65.

**Goal:** Demonstrate that oversampled exemplars bias classification decisions locally, showing exemplar-based memory effects in continuous space.

**Design:**
- **Model:** CobwebContinuousTree
- **Stimuli:** 12 color exemplars (Munsell chips) defined in continuous 2D space (dim1, dim2)
- **Manipulation:** One exemplar (A1) from category A is presented 5× more frequently than other exemplars
- **Test:** Classification probability of all exemplars

**Expected Effect:** Higher P(Category A) for items closer to the frequent exemplar (A1) compared to items equidistant from the prototype but near rare exemplars.

**Output Files:**
- `results/exp_specific_instance_continuous.csv` - Classification probabilities
- `results/specific_instance_curve_continuous.png` - **Graph:** Line plot of P(Category A) vs. Distance to Frequent Exemplar (A1). Curve should peak near the frequent exemplar.
- `results/metadata.json` - Experiment parameters and summary

**Model Evaluation:** Check if P(A) is higher for A1 than for equidistant rare neighbors.

---

### 2. Category-Label / Feedback Effect

**Location:** `experiments/category-label-feedback-effect/`

**Citation:** Posner, M. I., & Keele, S. W. (1968). On the genesis of abstract ideas. *Journal of Experimental Psychology, 77*(3, Pt.1), 353–363.

**Goal:** Show that supervised learning with category labels (feedback) accelerates prototype learning and improves generalization compared to unsupervised exposure.

**Design:**
- **Model:** CobwebContinuousTree
- **Stimuli:** 9-dot random patterns encoded as 18-dimensional continuous feature vectors
- **Distortion levels:** Prototype, Low Distortion, High Distortion
- **Label rates:** 1.0 (Feedback) vs 0.0 (No Feedback)

**Expected Effect:** 
1. Generalization gradient: Performance peaks at Prototype (0 distortion) and declines with distortion.
2. Feedback Effect: 100% Label Rate (Feedback) produces higher accuracy across all distortion levels than 0% (Unsupervised).

**Output Files:**
- `results/exp_category_label_feedback_continuous.csv` - Accuracy by label rate and distortion
- `results/category_label_feedback_curve_continuous.png` - **Graph:** Line plot of Endorsement Probability (Accuracy) vs. Distortion Level. Separate lines for Feedback vs No Feedback conditions.
- `results/metadata.json` - Experiment parameters

**Model Evaluation:** Compare accuracy curves; Feedback condition should be strictly higher.

---

### 3. Base-Rate & Inverse Base-Rate Effect

**Location:** `experiments/base-rate-effect/`

**Citation:** Medin, D. L., & Edelson, S. M. (1988). Problem structure and the use of base-rate information from experience. *Journal of Experimental Psychology: General, 117*(1), 68-85.

**Goal:** Reproduce the inverse base-rate effect where on ambiguous test trials, participants choose the rare category despite lower training base rates.

**Design:**
- **Model:** CobwebDiscreteTree
- **Cover Story:** Medical Diagnosis (Symptoms -> Disease)
- **Training:** 
  - AB → Common Disease (Frequent, e.g., 40 trials) using I+PC symptoms
  - AC → Rare Disease (Infrequent, e.g., 10 trials) using I+PR symptoms
- **Critical Test:** BC (Ambiguous) → Tests conflict between PC (predicts Common) and PR (predicts Rare)

**Expected Effect:** On ambiguous BC test trials, the model should prefer the **Rare** category (Inverse Base-Rate Effect) because the cue for Rare (C) is perfectly predictive, whereas the cue for Common (B) is contextualized by A.

**Output Files:**
- `results/exp_base_rate_discrete.csv` - Predictions by test type
- `results/base_rate_ibre_bar_discrete.png` - **Graph:** Grouped bar chart showing proportion of "Common" vs "Rare" responses for AB, AC, and BC stimuli. BC should show Rare > Common.
- `results/metadata.json` - IBRE rate and parameters

**Model Evaluation:** `ibre_rate_observed` should show P(Rare) > P(Common) for BC items.

---

### 4. Correlated-Feature / XOR Difficulty

**Location:** `experiments/correlated-feature-effect/`

**Citation:** Medin, D. L., Altom, M. W., Edelson, S. M., & Freko, D. (1982). Correlated symptoms and simulated medical classification. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 8*(1), 37-50.

**Goal:** Demonstrate that configural (XOR) category structures are harder to learn than linearly separable structures.

**Design:**
- **Model:** CobwebDiscreteTree
- **Stimuli:** 4 binary symptoms (Case Studies)
- **Structure:**
  - **Correlated (XOR):** Symptoms 1 & 2 form an XOR pattern (00->A, 01->B, 10->B, 11->A)
  - **Uncorrelated (Separable):** Symptom 1 predicts category independently of Symptom 2
- **Training:** Equal frequency exposure (8 reps per pattern)

**Expected Effect:** Configural (XOR) structure results in lower classification accuracy than Uncorrelated structure.

**Output Files:**
- `results/exp_correlated_feature_discrete.csv` - Accuracy by structure
- `results/correlated_feature_bar_discrete.png` - **Graph:** Bar chart comparing mean accuracy for Correlated vs Uncorrelated conditions. Uncorrelated bar should be higher.
- `results/metadata.json` - Final accuracies stats

**Model Evaluation:** `final_accuracy_separable` > `final_accuracy_xor`.

---

### 5. Fan Effect

**Location:** `experiments/fan-effect/`

**Citation:** Anderson, J. R. (1974). Retrieval of propositional information from long-term memory. *Cognitive Psychology, 6*(4), 451-474.

**Goal:** Show that retrieval interference increases with the number of associations (fan size) attached to a concept.

**Design:**
- **Model:** CobwebDiscreteTree
- **Stimuli:** "Person is in Location" sentences (Subject-Relation-Object triplets)
- **Fan Manipulation:** Number of distinct locations associated with a Person (Fan Size = 1, 2, 4, 8)
- **Task:** Retrieve "Object" given "Subject" + "Relation"

**Expected Effect:** Retrieval probability decreases as Fan Size increases. Simulated reaction time (1/Prob) increases linearly with Fan Size.

**Output Files:**
- `results/exp_fan_effect_discrete.csv` - Retrieval probabilities
- `results/fan_effect_rt_curve_discrete.png` - **Graph:** Line plot of Simulated Reaction Time (1/Probability) vs Fan Size. Should show an upward-sloping linear trend.
- `results/metadata.json` - Regression statistics

**Model Evaluation:** Positive correlation between Fan Size and Simulated RT.

---

## How to Run Experiments

### Individual Experiment

```bash
cd experiments/<effect-name>/
python exp_<effect>_discrete.py  # or exp_<effect>_continuous.py
```

### Generate Visualizations

```bash
cd experiments/<effect-name>/results/
python visualize.py
```

## References

Anderson, J. R. (1974). Retrieval of propositional information from long-term memory. *Cognitive Psychology, 6*(4), 451-474.

Medin, D. L., Altom, M. W., Edelson, S. M., & Freko, D. (1982). Correlated symptoms and simulated medical classification. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 8*(1), 37-50.

Medin, D. L., & Edelson, S. M. (1988). Problem structure and the use of base-rate information from experience. *Journal of Experimental Psychology: General, 117*(1), 68-85.

Nosofsky, R. M. (1988). Similarity, frequency, and category representations. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 14*(1), 54-65.

Posner, M. I., & Keele, S. W. (1968). On the genesis of abstract ideas. *Journal of Experimental Psychology, 77*(3, Pt.1), 353–363.

---

## Notes

- All experiments use `alpha=0.5` for CobwebDiscreteTree