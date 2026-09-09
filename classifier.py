"""
Project 2: Data Classification Using AI
DecodeLabs Industrial Training Kit - Batch 2026

Goal: Build a basic classification model using a small dataset (Iris).

Key Requirements covered:
- Load and understand a dataset          -> load_dataset()
- Split data into training and testing   -> train_test_split (80/20, shuffled)
- Apply a simple classification algorithm -> K-Nearest Neighbors (KNN)

Architecture: IPO Framework (per the training deck)
  INPUT   -> Iris dataset + Feature Scaling (StandardScaler)
  PROCESS -> Train-Test Split + KNN Algorithm
  OUTPUT  -> Confusion Matrix + F1 Score (not just raw accuracy,
             since "in imbalanced data, accuracy is a lie")
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)


# ----------------------------------------------------------------------
# STAGE 1 (INPUT): Load and understand the dataset
# ----------------------------------------------------------------------
def load_dataset():
    """
    Loads the classic Iris benchmark:
    150 samples, 3 classes (Setosa, Versicolor, Virginica), 4 features
    (Sepal Length, Sepal Width, Petal Length, Petal Width).
    """
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="species")
    target_names = iris.target_names

    print("=== Dataset Overview ===")
    print(f"Samples: {X.shape[0]} | Features: {X.shape[1]} | Classes: {len(target_names)}")
    print(f"Class names: {list(target_names)}")
    print("\nFirst 5 rows:")
    print(X.head())
    print("\nClass distribution:")
    print(y.value_counts().sort_index().to_string())
    print()

    return X, y, target_names


# ----------------------------------------------------------------------
# STAGE 2 (PROCESS - part A): Train-Test Split
# ----------------------------------------------------------------------
def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits data 80/20 into training and test sets.
    Shuffling (default in train_test_split) removes order bias,
    and stratify=y keeps class proportions balanced in both sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}\n")
    return X_train, X_test, y_train, y_test


# ----------------------------------------------------------------------
# STAGE 2 (PROCESS - part B): Feature Scaling ("The Gatekeeper Rule")
# ----------------------------------------------------------------------
def scale_features(X_train, X_test):
    """
    StandardScaler transforms features to mean=0, variance=1.
    KNN relies on distance calculations, so unscaled features
    (e.g. one column ranging 0-1000 vs another 0-1) would bias
    the "nearest neighbor" vote toward the larger-scale feature.
    Important: fit the scaler ONLY on training data, then transform
    both sets, to avoid data leakage from the test set.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ----------------------------------------------------------------------
# STAGE 3 (PROCESS - part C): Train the KNN Classifier
# ----------------------------------------------------------------------
def train_model(X_train_scaled, y_train, k=5):
    """
    Instantiate -> Fit -> (Predict happens separately)
    K-Nearest Neighbors: classifies a new point by majority vote
    among its 'k' closest neighbors in feature space.
    k=5 is a reasonable default (avoids k=1 overfitting to noise
    and avoids a very large k that underfits / becomes too generic).
    """
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    return model


# ----------------------------------------------------------------------
# STAGE 4 (OUTPUT): Validate with Confusion Matrix + F1 Score
# ----------------------------------------------------------------------
def evaluate_model(model, X_test_scaled, y_test, target_names):
    predictions = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")
    cm = confusion_matrix(y_test, predictions)

    print("=== Output Validation ===")
    print(f"Accuracy : {acc:.2%}")
    print(f"F1 Score (macro): {f1:.4f}")

    print("\nConfusion Matrix:")
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual: {n}" for n in target_names],
        columns=[f"Pred: {n}" for n in target_names],
    )
    print(cm_df)

    print("\nFull Classification Report:")
    print(classification_report(y_test, predictions, target_names=target_names))

    return predictions, acc, f1, cm


# ----------------------------------------------------------------------
# MAIN PIPELINE (Input -> Process -> Output)
# ----------------------------------------------------------------------
def main():
    # INPUT
    X, y, target_names = load_dataset()

    # PROCESS: split -> scale -> train
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    model = train_model(X_train_scaled, y_train, k=5)

    # OUTPUT: evaluate
    evaluate_model(model, X_test_scaled, y_test, target_names)

    # Quick demo: classify one brand-new (unseen) flower measurement
    print("\n=== Demo: Classifying a new sample ===")
    new_sample = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2]], columns=X.columns
    )  # typical Setosa measurements
    new_sample_scaled = scaler.transform(new_sample)
    prediction = model.predict(new_sample_scaled)
    print(f"Input measurements: {new_sample.iloc[0].tolist()}")
    print(f"Predicted species: {target_names[prediction[0]]}")


if __name__ == "__main__":
    main()
