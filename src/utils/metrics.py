
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def compute_metrics(labels, predictions):
    """Computes standard evaluation metrics."""
    accuracy = float(accuracy_score(labels, predictions))
    precision = float(precision_score(labels, predictions, average="macro", zero_division=0))
    recall = float(recall_score(labels, predictions, average="macro", zero_division=0))
    macro_f1 = float(f1_score(labels, predictions, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(labels, predictions, average="weighted", zero_division=0))

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
    }


def get_confusion_matrix(labels, predictions, num_classes=None):
    """Computes the confusion matrix."""
    labels_list = list(range(num_classes)) if num_classes is not None else None
    return confusion_matrix(labels, predictions, labels=labels_list)


def log_robustness_result(csv_path, dataset_name, corruption, severity, metrics):
    """Appends or creates a robustness result entry."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "dataset": dataset_name.upper(),
        "corruption": corruption,
        "severity": int(severity),
        **metrics,
    }

    df_row = pd.DataFrame([row])
    if csv_path.exists():
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, mode="w", header=True, index=False)
