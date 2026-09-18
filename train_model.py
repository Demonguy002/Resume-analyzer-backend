import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# PATHS
# ============================================================

DATASET = os.path.join(
    "dataset",
    "UpdatedResumeDataSet_clean.csv"
)

MODEL_DIR = "models"
EVALUATION_DIR = "evaluation"

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "resume_classifier.joblib"
)

CONFUSION_MATRIX_PNG = os.path.join(
    EVALUATION_DIR,
    "confusion_matrix.png"
)

CONFUSION_MATRIX_CSV = os.path.join(
    EVALUATION_DIR,
    "confusion_matrix.csv"
)

CLASSIFICATION_REPORT = os.path.join(
    EVALUATION_DIR,
    "classification_report.txt"
)

METRICS_JSON = os.path.join(
    EVALUATION_DIR,
    "metrics.json"
)

PREDICTIONS_CSV = os.path.join(
    EVALUATION_DIR,
    "test_predictions.csv"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVALUATION_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("\n" + "=" * 60)
print("RESUME SCREENING MODEL TRAINING")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATASET)

df["Category"] = (
    df["Category"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["Resume"] = (
    df["Resume"]
    .fillna("")
    .astype(str)
)

# Remove invalid rows
df = df[
    (df["Category"] != "") &
    (df["Resume"].str.strip() != "")
].copy()

print(f"Total resumes     : {len(df)}")
print(f"Total categories  : {df['Category'].nunique()}")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    df["Resume"],
    df["Category"],
    test_size=0.20,
    random_state=42,
    stratify=df["Category"]
)

print(f"Training samples   : {len(X_train)}")
print(f"Testing samples    : {len(X_test)}")


# ============================================================
# MODEL PIPELINE
# ============================================================

print("\nCreating TF-IDF + LinearSVC model...")

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            sublinear_tf=True,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.98,
            max_features=150000,
            strip_accents="unicode"
        )
    ),

    (
        "classifier",
        LinearSVC(
            C=1.0,
            class_weight="balanced"
        )
    )
])


# ============================================================
# TRAIN
# ============================================================

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# TEST
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 60)
print(f"TEST ACCURACY : {accuracy * 100:.2f}%")
print(f"PRECISION      : {precision * 100:.2f}%")
print(f"RECALL         : {recall * 100:.2f}%")
print(f"F1 SCORE       : {f1 * 100:.2f}%")
print("=" * 60)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print("\nClassification Report:\n")
print(report)

with open(
    CLASSIFICATION_REPORT,
    "w",
    encoding="utf-8"
) as f:

    f.write("RESUME SCREENING CLASSIFICATION REPORT\n")
    f.write("=" * 60 + "\n\n")

    f.write(
        f"Test Accuracy : {accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Precision     : {precision * 100:.2f}%\n"
    )

    f.write(
        f"Recall        : {recall * 100:.2f}%\n"
    )

    f.write(
        f"F1 Score      : {f1 * 100:.2f}%\n\n"
    )

    f.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nGenerating confusion matrix...")

labels = model.classes_

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)


# Save confusion matrix as CSV

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

cm_df.to_csv(
    CONFUSION_MATRIX_CSV
)


# Save confusion matrix as PNG

fig, ax = plt.subplots(
    figsize=(20, 18)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

disp.plot(
    ax=ax,
    xticks_rotation=90,
    values_format="d",
    cmap="Blues",
    colorbar=True
)

plt.title(
    f"Resume Category Confusion Matrix\n"
    f"Test Accuracy: {accuracy * 100:.2f}%"
)

plt.xlabel(
    "Predicted Category"
)

plt.ylabel(
    "Actual Category"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PNG,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {CONFUSION_MATRIX_PNG}"
)


# ============================================================
# TEST PREDICTIONS CSV
# ============================================================

predictions_df = pd.DataFrame({
    "Actual_Category": y_test.values,
    "Predicted_Category": y_pred
})

predictions_df.to_csv(
    PREDICTIONS_CSV,
    index=False
)

print(
    f"Saved: {PREDICTIONS_CSV}"
)


# ============================================================
# METRICS JSON
# ============================================================

metrics = {
    "dataset": "UpdatedResumeDataSet_clean.csv",

    "total_samples": int(len(df)),

    "training_samples": int(len(X_train)),

    "testing_samples": int(len(X_test)),

    "number_of_categories": int(
        df["Category"].nunique()
    ),

    "test_accuracy": round(
        accuracy * 100,
        2
    ),

    "weighted_precision": round(
        precision * 100,
        2
    ),

    "weighted_recall": round(
        recall * 100,
        2
    ),

    "weighted_f1_score": round(
        f1 * 100,
        2
    ),

    "model": "LinearSVC",

    "C": 1.0,

    "tfidf_ngram_range": "1-2",

    "test_size": 0.20,

    "random_state": 42
}

with open(
    METRICS_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

print(
    f"Saved: {METRICS_JSON}"
)


# ============================================================
# FINAL MODEL
# ============================================================

print("\nTraining final model on complete dataset...")

model.fit(
    df["Resume"],
    df["Category"]
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE,
    compress=3
)

print(
    f"\nSaved model: {MODEL_FILE}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Accuracy       : {accuracy * 100:.2f}%"
)

print(
    f"Precision      : {precision * 100:.2f}%"
)

print(
    f"Recall         : {recall * 100:.2f}%"
)

print(
    f"F1 Score       : {f1 * 100:.2f}%"
)

print("\nGenerated files:")

print("models/")
print("  └── resume_classifier.joblib")

print("\nevaluation/")
print("  ├── confusion_matrix.png")
print("  ├── confusion_matrix.csv")
print("  ├── classification_report.txt")
print("  ├── metrics.json")
print("  └── test_predictions.csv")

print("\nModel is ready for prediction.")
print("=" * 60)
