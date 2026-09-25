STEP 1 — Create the project folder

Create a folder named:

Assignment3_DryBean


STEP 2 — Create the project structure

Inside the folder, create:

Assignment3_DryBean/
├── Assignment3_DryBean.ipynb
├── data/
├── figures/
├── results/
├── README.md
├── requirements.txt
└── .gitignore


STEP 3 — Add the dataset

Place the downloaded dataset:

Dry_Bean_Dataset.xlsx

inside:

Assignment3_DryBean/data/


STEP 4 — Open Jupyter Notebook

Open:

Assignment3_DryBean.ipynb

from the Assignment3_DryBean project folder.


STEP 5 — Import the required libraries

Use:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.manifold import TSNE

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


STEP 6 — Create the figures folder

Use:

os.makedirs("../figures", exist_ok=True)


STEP 7 — Create the results folder

Use:

os.makedirs("../results", exist_ok=True)


STEP 8 — Load the dataset

Use:

df = pd.read_excel("../data/Dry_Bean_Dataset.xlsx")


STEP 9 — Display the first five rows

Use:

df.head()


STEP 10 — Check the dataset dimensions

Use:

df.shape

Initial dataset:

(13611, 17)

This means:
13,611 observations
17 columns
16 numerical features
1 target column


STEP 11 — Display column names

Use:

df.columns

The target column is:

Class


STEP 12 — Check data types

Use:

df.dtypes


STEP 13 — Check for missing values

Use:

df.isnull().sum()

The dataset contains no missing values.


STEP 14 — Check duplicate records

Use:

df.duplicated().sum()

There are 68 duplicate records.


STEP 15 — Remove duplicate records

Use:

df = df.drop_duplicates().reset_index(drop=True)


STEP 16 — Check the new dataset size

Use:

df.shape

Final dataset:

(13543, 17)


STEP 17 — Check class distribution

Use:

df["Class"].value_counts()

Expected class distribution:

DERMASON    3546
SIRA        2636
SEKER       2027
HOROZ       1928
CALI        1630
BARBUNYA    1322
BOMBAY       522


STEP 18 — Plot class distribution

Use:

plt.figure(figsize=(10, 6))

df["Class"].value_counts().plot(kind="bar")

plt.title("Dry Bean Class Distribution")
plt.xlabel("Bean Class")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("../figures/class_distribution.png", dpi=300)
plt.show()


STEP 19 — Separate features and target

Use:

X = df.drop("Class", axis=1)
y = df["Class"]


STEP 20 — Check feature dimensions

Use:

X.shape

Expected:

(13543, 16)


STEP 21 — Encode the target classes

Use:

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


STEP 22 — Display encoded class names

Use:

label_encoder.classes_

Expected:

['BARBUNYA' 'BOMBAY' 'CALI' 'DERMASON' 'HOROZ' 'SEKER' 'SIRA']


STEP 23 — Split the dataset

Use:

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


STEP 24 — Check training and testing dimensions

Use:

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

Expected:

Training data: (10834, 16)
Testing data: (2709, 16)


STEP 25 — Standardize the features

Use:

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


STEP 26 — Create a stratified sample for t-SNE

Use:

X_tsne, _, y_tsne, _ = train_test_split(
    X_train_scaled,
    y_train,
    train_size=5000,
    random_state=42,
    stratify=y_train
)


STEP 27 — Apply t-SNE

Use:

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    max_iter=1000,
    init="pca"
)

X_tsne_2d = tsne.fit_transform(X_tsne)


STEP 28 — Plot t-SNE

Use:

plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    X_tsne_2d[:, 0],
    X_tsne_2d[:, 1],
    c=y_tsne,
    cmap="tab10",
    s=10,
    alpha=0.7
)

plt.title("t-SNE Visualization of Dry Bean Classes")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")

plt.tight_layout()

plt.savefig("../figures/tsne_dry_bean.png", dpi=300)

plt.show()


STEP 29 — Create the three SVM models

Use:

linear_svm = SVC(
    kernel="linear",
    random_state=42
)

poly_svm = SVC(
    kernel="poly",
    degree=3,
    random_state=42
)

rbf_svm = SVC(
    kernel="rbf",
    random_state=42
)


STEP 30 — Train and evaluate Linear SVM

Use:

start_time = time.perf_counter()

linear_svm.fit(X_train_scaled, y_train)

linear_training_time = time.perf_counter() - start_time

start_time = time.perf_counter()

y_pred_linear = linear_svm.predict(X_test_scaled)

linear_prediction_time = time.perf_counter() - start_time

linear_accuracy = accuracy_score(y_test, y_pred_linear)

linear_precision = precision_score(
    y_test,
    y_pred_linear,
    average="weighted",
    zero_division=0
)

linear_recall = recall_score(
    y_test,
    y_pred_linear,
    average="weighted",
    zero_division=0
)

linear_f1 = f1_score(
    y_test,
    y_pred_linear,
    average="weighted",
    zero_division=0
)

print("Linear SVM")
print("Accuracy:", linear_accuracy)
print("Precision:", linear_precision)
print("Recall:", linear_recall)
print("F1 Score:", linear_f1)


STEP 31 — Train and evaluate Polynomial SVM

Use:

start_time = time.perf_counter()

poly_svm.fit(X_train_scaled, y_train)

poly_training_time = time.perf_counter() - start_time

start_time = time.perf_counter()

y_pred_poly = poly_svm.predict(X_test_scaled)

poly_prediction_time = time.perf_counter() - start_time

poly_accuracy = accuracy_score(y_test, y_pred_poly)

poly_precision = precision_score(
    y_test,
    y_pred_poly,
    average="weighted",
    zero_division=0
)

poly_recall = recall_score(
    y_test,
    y_pred_poly,
    average="weighted",
    zero_division=0
)

poly_f1 = f1_score(
    y_test,
    y_pred_poly,
    average="weighted",
    zero_division=0
)

print("Polynomial SVM")
print("Accuracy:", poly_accuracy)
print("Precision:", poly_precision)
print("Recall:", poly_recall)
print("F1 Score:", poly_f1)


STEP 32 — Train and evaluate RBF SVM

Use:

start_time = time.perf_counter()

rbf_svm.fit(X_train_scaled, y_train)

rbf_training_time = time.perf_counter() - start_time

start_time = time.perf_counter()

y_pred_rbf = rbf_svm.predict(X_test_scaled)

rbf_prediction_time = time.perf_counter() - start_time

rbf_accuracy = accuracy_score(y_test, y_pred_rbf)

rbf_precision = precision_score(
    y_test,
    y_pred_rbf,
    average="weighted",
    zero_division=0
)

rbf_recall = recall_score(
    y_test,
    y_pred_rbf,
    average="weighted",
    zero_division=0
)

rbf_f1 = f1_score(
    y_test,
    y_pred_rbf,
    average="weighted",
    zero_division=0
)

print("RBF SVM")
print("Accuracy:", rbf_accuracy)
print("Precision:", rbf_precision)
print("Recall:", rbf_recall)
print("F1 Score:", rbf_f1)


STEP 33 — Create SVM comparison table

Use:

svm_results = pd.DataFrame({
    "Model": [
        "SVM - Linear",
        "SVM - Polynomial",
        "SVM - RBF"
    ],
    "Accuracy": [
        linear_accuracy,
        poly_accuracy,
        rbf_accuracy
    ],
    "Precision": [
        linear_precision,
        poly_precision,
        rbf_precision
    ],
    "Recall": [
        linear_recall,
        poly_recall,
        rbf_recall
    ],
    "F1_Score": [
        linear_f1,
        poly_f1,
        rbf_f1
    ],
    "Training_Time": [
        linear_training_time,
        poly_training_time,
        rbf_training_time
    ],
    "Prediction_Time": [
        linear_prediction_time,
        poly_prediction_time,
        rbf_prediction_time
    ]
})

svm_results


STEP 34 — Save SVM results

Use:

svm_results.to_csv(
    "../results/SVM_Comparison.csv",
    index=False
)


STEP 35 — Generate SVM confusion matrices

Use:

svm_predictions = {
    "Linear SVM": y_pred_linear,
    "Polynomial SVM": y_pred_poly,
    "RBF SVM": y_pred_rbf
}

for name, predictions in svm_predictions.items():

    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.colorbar()

    plt.tight_layout()

    filename = name.lower().replace(" ", "_")

    plt.savefig(
        f"../figures/confusion_matrix_{filename}.png",
        dpi=300
    )

    plt.show()


STEP 36 — Plot SVM kernel comparison

Use:

plt.figure(figsize=(10, 6))

plt.bar(
    svm_results["Model"],
    svm_results["Accuracy"]
)

plt.title("SVM Kernel Accuracy Comparison")
plt.xlabel("SVM Model")
plt.ylabel("Accuracy")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "../figures/svm_kernel_comparison.png",
    dpi=300
)

plt.show()


STEP 37 — Define Decision Tree parameters

Use:

max_depth_values = [5, 10, 15, 20]

min_samples_split_values = [2, 10, 20]

This creates 12 Decision Tree configurations.


STEP 38 — Train and evaluate all Decision Tree configurations

Use:

dt_results = []

for max_depth in max_depth_values:

    for min_samples_split in min_samples_split_values:

        dt_model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )

        start_time = time.perf_counter()

        dt_model.fit(X_train, y_train)

        training_time = time.perf_counter() - start_time

        start_time = time.perf_counter()

        y_pred_dt = dt_model.predict(X_test)

        prediction_time = time.perf_counter() - start_time

        accuracy = accuracy_score(
            y_test,
            y_pred_dt
        )

        precision = precision_score(
            y_test,
            y_pred_dt,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred_dt,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred_dt,
            average="weighted",
            zero_division=0
        )

        dt_results.append({
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1,
            "Training_Time": training_time,
            "Prediction_Time": prediction_time
        })


STEP 39 — Create Decision Tree results DataFrame

Use:

dt_results_df = pd.DataFrame(dt_results)

dt_results_df


STEP 40 — Save Decision Tree results

Use:

dt_results_df.to_csv(
    "../results/Decision_Tree_Comparison.csv",
    index=False
)


STEP 41 — Plot Decision Tree depth comparison

Use:

plt.figure(figsize=(10, 6))

for split_value in min_samples_split_values:

    subset = dt_results_df[
        dt_results_df["min_samples_split"] == split_value
    ]

    plt.plot(
        subset["max_depth"],
        subset["Accuracy"],
        marker="o",
        label=f"min_samples_split={split_value}"
    )

plt.title("Decision Tree Accuracy by Maximum Depth")
plt.xlabel("Maximum Depth")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "../figures/decision_tree_depth_comparison.png",
    dpi=300
)

plt.show()


STEP 42 — Plot Decision Tree F1 comparison

Use:

plt.figure(figsize=(10, 6))

for split_value in min_samples_split_values:

    subset = dt_results_df[
        dt_results_df["min_samples_split"] == split_value
    ]

    plt.plot(
        subset["max_depth"],
        subset["F1_Score"],
        marker="o",
        label=f"min_samples_split={split_value}"
    )

plt.title("Decision Tree F1 Score Comparison")
plt.xlabel("Maximum Depth")
plt.ylabel("F1 Score")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "../figures/decision_tree_f1_comparison.png",
    dpi=300
)

plt.show()


STEP 43 — Select the Decision Tree configuration for detailed evaluation

Use:

max_depth = 15
min_samples_split = 20

This tested configuration produced:

Accuracy  = 0.908822
Precision = 0.908626
Recall    = 0.908822
F1 Score  = 0.908552

Train the selected Decision Tree:

final_dt = DecisionTreeClassifier(
    max_depth=15,
    min_samples_split=20,
    random_state=42
)

start_time = time.perf_counter()

final_dt.fit(X_train, y_train)

final_dt_training_time = time.perf_counter() - start_time

start_time = time.perf_counter()

final_dt_pred = final_dt.predict(X_test)

final_dt_prediction_time = time.perf_counter() - start_time


STEP 44 — Generate final Decision Tree confusion matrix

Use:

cm_dt = confusion_matrix(
    y_test,
    final_dt_pred
)

plt.figure(figsize=(8, 6))

plt.imshow(
    cm_dt,
    interpolation="nearest",
    cmap="Blues"
)

plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.colorbar()

plt.tight_layout()

plt.savefig(
    "../figures/confusion_matrix_decision_tree.png",
    dpi=300
)

plt.show()


STEP 45 — Create the final model comparison

Use:

final_comparison = pd.DataFrame({
    "Model": [
        "SVM - Linear",
        "SVM - Polynomial",
        "SVM - RBF",
        "Decision Tree"
    ],

    "Accuracy": [
        0.921004,
        0.913990,
        0.921004,
        0.908822
    ],

    "Precision": [
        0.921098,
        0.921894,
        0.921411,
        0.908626
    ],

    "Recall": [
        0.921004,
        0.913990,
        0.921004,
        0.908822
    ],

    "F1_Score": [
        0.921001,
        0.915492,
        0.921078,
        0.908552
    ],

    "Training_Time": [
        1.384946,
        2.169969,
        2.004742,
        0.714068
    ],

    "Prediction_Time": [
        0.481592,
        0.786913,
        2.029755,
        0.006538
    ],

    "max_depth": [
        np.nan,
        np.nan,
        np.nan,
        15
    ],

    "min_samples_split": [
        np.nan,
        np.nan,
        np.nan,
        20
    ]
})

final_comparison


STEP 46 — Save final comparison and create final chart

Save the final comparison:

final_comparison.to_csv(
    "../results/Final_Model_Comparison.csv",
    index=False
)

Create the final chart:

plt.figure(figsize=(10, 6))

plt.bar(
    final_comparison["Model"],
    final_comparison["Accuracy"]
)

plt.title("Final Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "../figures/final_model_comparison.png",
    dpi=300
)

plt.show()


STEP 47 — Finalize the project and update GitHub

Your final project should contain:

Assignment3_DryBean/
│
├── Assignment3_DryBean.ipynb
│
├── data/
│   └── Dry_Bean_Dataset.xlsx
│
├── figures/
│   ├── class_distribution.png
│   ├── tsne_dry_bean.png
│   ├── confusion_matrix_linear_svm.png
│   ├── confusion_matrix_polynomial_svm.png
│   ├── confusion_matrix_rbf_svm.png
│   ├── svm_kernel_comparison.png
│   ├── decision_tree_depth_comparison.png
│   ├── decision_tree_f1_comparison.png
│   ├── confusion_matrix_decision_tree.png
│   └── final_model_comparison.png
│
├── results/
│   ├── SVM_Comparison.csv
│   ├── Decision_Tree_Comparison.csv
│   └── Final_Model_Comparison.csv
│
├── README.md
├── requirements.txt
└── .gitignore


requirements.txt should contain:

pandas
numpy
matplotlib
scikit-learn
openpyxl
jupyter


FINAL RESULTS

SVM - Linear:
Accuracy  = 0.921004
Precision = 0.921098
Recall    = 0.921004
F1 Score  = 0.921001

SVM - Polynomial:
Accuracy  = 0.913990
Precision = 0.921894
Recall    = 0.913990
F1 Score  = 0.915492

SVM - RBF:
Accuracy  = 0.921004
Precision = 0.921411
Recall    = 0.921004
F1 Score  = 0.921078

Decision Tree:
Accuracy  = 0.908822
Precision = 0.908626
Recall    = 0.908822
F1 Score  = 0.908552

Decision Tree parameters:
max_depth = 15
min_samples_split = 20


FINAL GITHUB CHECKLIST

1. Save the notebook.
2. Make sure all notebook cells run without errors.
3. Make sure the figures folder contains all generated PNG files.
4. Make sure the results folder contains all CSV files.
5. Make sure README.md is present.
6. Make sure requirements.txt is present.
7. Open GitHub Desktop.
8. Select Assignment3_DryBean.
9. Go to Changes.
10. Check the changed files.
11. Enter commit message:

Complete Assignment 3 documentation and analysis

12. Click Commit to main.
13. Click Push origin.
14. Open the GitHub repository.
15. Check that README.md displays correctly.
16. Check that the notebook is present.
17. Check that figures are present.
18. Check that results are present.
AUTHOR: VYSHNAVI PERUMAL
