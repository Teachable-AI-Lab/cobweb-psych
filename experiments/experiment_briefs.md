# Cognitive Experiment Briefs

## 1. Fan Effect

**Based on:** Anderson (1974); Anderson (1991); Reder and Ross (1983)

### The Effect
The **Fan Effect** describes the phenomenon where the time required to retrieve a specific fact from memory increases as the number of associations ("fan") connected to the concepts in that fact increases. It provides evidence for limited-capacity activation spreading in semantic networks.

### Stimuli
The stimuli typically consist of simple subject-predicate sentences (e.g., "The Hippie is in the Park").
- **Dimensions:** Concepts are usually distinct entities (Persons) and Locations/Predicates.
- **Structure:** The experiment design strictly controls the number of times each Person and each Location appears across the study set.
    - *Example:* "The Hippie" might appear in 1 sentence (Fan 1) or 3 sentences (Fan 3).

### Training Regime
1.  **Study Phase:** Participants memorize a list of sentences (facts) until they can recall them perfectly. The critical manipulation is the "Fan Size" (1, 2, or 3) of the concepts involved.
2.  **Test Phase:** Participants are presented with probes (Sentences) and must quickly decide if the sentence is "Old" (studied) or "New" (unstudied).

### Expected Effect in Humans
-   **Reaction Time (RT):** RT increases linearly with the Fan size of the concepts in the probe. A sentence containing a Fan-3 person and Fan-3 location takes significantly longer to verify than a Fan-1/Fan-1 sentence.
-   **Accuracy:** Error rates may also increase slightly with Fan size, though the primary metric is retrieval latency.

---

## 2. Probability Matching Effect

**Based on:** Gluck & Bower (1988)

### The Effect
**Probability Matching** in category learning refers to the tendency of human learners to select category labels with a frequency proportional to their probability of being correct given the cues, rather than "maximizing" (always choosing the most likely category).

### Stimuli
The task is often framed as a **Medical Diagnosis** problem.
-   **Cues:** discrete symptoms (e.g., Bloody Nose, Stomach Cramps, Discolored Gums).
-   **Categories:** Two hypothetical diseases (e.g., Burglitis vs. Midosis).

### Training Regime
1.  **Probabilistic Structure:** Participants are shown patients with specific patterns of 1-4 symptoms.
    -   *Base Rates:* One disease is typically more common (e.g., 250 patients) than the other (e.g., 83 patients).
    -   *Cue Validity:* Symptoms are probabilistically associated with diseases. Some are highly diagnostic, others are weak predictors.
2.  **Procedure:** On each trial, a patient's symptoms are shown. The participant predicts the disease and receives corrective feedback.
3.  **Features:** The experiment typically uses a "configuration" of cues where the optimal decision requires integrating evidence, but humans often rely on simple component cues (Rule-based or Exemplar-based) or match the probabilities.

### Expected Effect in Humans
-   **Matching Behavior:** When presented with a symptom pattern where Disease A has a 70% probability and Disease B has 30%, humans tend to guess "Disease A" roughly 70% of the time and "Disease B" 30% of the time, rather than choosing "Disease A" 100% of the time (which would maximize accuracy).
-   **Base Rate Neglect:** Depending on the specific cue confusions, participants may also underestimate the high base-rate disease in the presence of conflicting diagnostic cues (a phenomenon explored further in Medin & Edelson, 1988).

---

## 3. Specific Instance Effect

**Based on:** Hayes-Roth & Hayes-Roth (1977)

### The Effect
The **Specific Instance Effect** (or Exemplar Strength Effect) demonstrates that category representations are not purely abstract prototypes. Instead, memory for specific training exemplars strongly influences classification and recognition, sometimes overriding prototype-based similarity.

### Stimuli
Descriptions of fictitious people defined by discrete features.
-   **Dimensions:** 3 relevant dimensions (e.g., Age, Education, Marital Status) and irrelevant distractors.
-   **Example Values:** "30 years old", "High School", "Single".
-   **Coding:** Often symbolic (Feature values 1, 2, 3, 4).

### Training Regime
The experiment strictly manipulates **Frequency** and **Distance to Prototype** orthogonally.
1.  **Prototypes:** Two prototypes (e.g., 111 and 222) define the centers of "Club 1" and "Club 2". These Prototypes are *never* shown during training (Frequency = 0).
2.  **Exemplars:**
    -   *Frequent Exemplar:* A specific item (e.g., 112) is shown very often (e.g., 10 times).
    -   *Rare Exemplar:* An item equidistant from the prototype (e.g., 121) is shown rarely (e.g., 1 time).
3.  **Task:** Participants learn to classify individuals into clubs with feedback.

### Expected Effect in Humans
-   **Recognition:** Participants judge the *Frequent Exemplar* as being "more familiar" or "older" than the *Prototype*, even though the Prototype is central to the category cluster.
-   **Classification:** While the Prototype is classified accurately (due to similarity to the cluster center), the *Frequent Exemplar* is often classified with higher confidence or speed than the Prototype or Rare exemplars, demonstrating that token frequency is stored alongside category abstraction.
-   **Dissociation:** The effect highlights that feature combinations (specific instances) are encoded distinct from independent feature probabilities.

---

## 4. Central Tendency Effect

**Based on:** Medin, D. L., & Schaffer, M. M. (1978), Experiment 2 (The Context Theory)

### The Effect
The **Central Tendency** test examines whether learners classify new items based on their logical distance to a category center (Prototype theory) or their similarity to specific stored training examples (Exemplar/Context theory). The classic finding is that learners can classify a "Prototype" (which they have never seen) with higher accuracy and confidence than the specific instances they actually studied, provided the Prototype is highly similar to the training set.

### Stimuli
Geometric forms defined by 4 binary dimensions (e.g., Color, Form, Size, Position).
-   **Structure:** A specific "5-4" or similar structure is used where the categories are not linearly separable by a simple rule.
-   **Notation:** Stimuli are often denoted as binary strings (e.g., `1110`, `0001`).

### Training Regime
1.  **Learning:** Participants view a small subset of possible stimuli (e.g., 6 items) and learn to classify them into Category A or B with feedback.
2.  **Criterion:** Training continues until the participant achieves a perfect run (e.g., 2 consecutive error-free passes).
3.  **Transfer:** Participants classify a mix of Old (training) and New items without feedback.

### Expected Effect in Humans
-   **Prototype Advantage:** The unseen Prototype (e.g., `1111`) is often classified into Category A with very high probability, sometimes exceeding the classification performance of the actual training items.
-   **Context Theory:** This effect is predicted by Exemplar models (like the Context Model) because the Prototype shares the most features with the stored training exemplars, maximizing summed similarity.
