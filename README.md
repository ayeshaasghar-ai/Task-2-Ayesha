# Project 2 — Data Classification Using AI
**DecodeLabs Industrial Training Kit | Batch 2026**
**Submitted by:** Ayesha

---

## 1. Goal
Build a basic classification model using a small dataset, applying the
fundamental supervised learning pipeline: load data, split it, train a
model, and validate its predictions.

## 2. Dataset
Used the classic **Iris benchmark**:
- **150 samples**, perfectly balanced (50 per class)
- **3 classes**: Setosa, Versicolor, Virginica
- **4 features**: Sepal Length, Sepal Width, Petal Length, Petal Width

## 3. Requirement → Implementation Mapping

| Requirement (from brief) | How it's implemented |
|---|---|
| **Load and understand a dataset** | `load_dataset()` loads Iris via scikit-learn, prints shape, first rows, and class distribution |
| **Split data into training and testing sets** | `split_data()` — 80/20 split via `train_test_split`, shuffled + stratified to preserve class balance |
| **Apply a simple classification algorithm** | `train_model()` — K-Nearest Neighbors (KNN) with k=5, following the Instantiate → Fit → Predict scikit-learn workflow |

## 4. IPO Architecture (per the training deck)

| Stage | What happens |
|---|---|
| **Input** | Iris dataset loaded; features scaled with `StandardScaler` (mean=0, variance=1) so no single feature dominates the distance calculation KNN relies on |
| **Process** | Train-test split (80/20) → KNN model trained on the training set |
| **Output** | Predictions validated with a **Confusion Matrix** and **F1 Score** — not just raw accuracy, since accuracy alone can be misleading ("Accuracy Mirage") |

## 5. Why Feature Scaling?
KNN classifies points based on distance to their nearest neighbors. Without
scaling, a feature measured in the 0–1000 range would dominate the distance
calculation over a feature measured in the 0–1 range, even if both are
equally important. `StandardScaler` is fit **only on the training data**
and then applied to both sets, to avoid data leakage from the test set.

## 6. Why Confusion Matrix + F1 Score (not just accuracy)?
In imbalanced datasets, a model can score high accuracy just by always
predicting the majority class. The confusion matrix shows exactly where
the model gets confused (which classes it mixes up), and the F1 score
balances precision and recall — a more honest measure of real performance.

## 7. How to Run
```bash
pip install scikit-learn pandas
python3 classifier.py
```

## 8. Results
- **Accuracy:** 93.33%
- **F1 Score (macro):** 0.93
- The model correctly classified all Setosa and Versicolor test samples;
  a small number of Virginica samples were confused with Versicolor —
  visible directly in the confusion matrix.

## 9. Extensions Beyond the Minimum Spec
- Full classification report (precision/recall/F1 per class), not just
  overall accuracy
- Stratified split to keep class balance consistent between train/test
- A live demo classifying one brand-new, unseen flower measurement
- Modular functions (`load_dataset`, `split_data`, `scale_features`,
  `train_model`, `evaluate_model`) instead of one script block, so
  individual stages can be reused or swapped (e.g. testing a different
  algorithm) later
