# Breast Cancer Decision Tree Classification

This project uses the Breast Cancer Wisconsin dataset from Scikit-learn to classify tumors as:

- Benign (non-cancerous)
- Malignant (cancerous)

The main goal of this project is to compare different Decision Tree algorithms and determine the best-performing model.

---

## Dataset Information

The dataset was imported from:

`sklearn.datasets.load_breast_cancer()`

### Dataset Details:
- 569 samples
- 30 features
- Target classes:
  - Benign
  - Malignant

### Important Features:
- mean radius
- mean texture
- mean perimeter
- mean area
- mean smoothness

---

## What is Gini?

Gini impurity is used in decision trees to measure how mixed the classes are in a node.

### Formula:

`Gini = 1 - Σ(pi²)`

Where:

- pi = probability of each class
- Lower Gini value means better class separation

---

## What is Entropy?

Entropy measures uncertainty in decision trees.

### Formula:

`Entropy = -Σ pi log2(pi)`

Lower entropy means better classification.

---

## Models Used

- Decision Tree (Gini)
- Decision Tree (Entropy)
- Gini with max_depth=3
- Entropy with max_depth=3

---

## Overfitting Problem

Initial models had:

- Training Accuracy: 100%
- Lower test accuracy

This indicated overfitting.

Solution:

- Limited tree depth using `max_depth=3`

---

## Results

| Model | Training Accuracy | Test Accuracy |
|--------|-------------------|---------------|
| Gini | 1.00 | 0.923 |
| Entropy | 1.00 | 0.959 |
| Gini (max_depth=3) | 0.969 | 0.964 |
| Entropy (max_depth=3) | 0.979 | 0.964 |

---

## Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab

---

## Project Goal

To compare decision tree algorithms and reduce overfitting for better classification performance.
