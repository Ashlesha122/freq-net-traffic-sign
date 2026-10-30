
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(cm, title="Confusion Matrix", save_path="results/figures/confusion_matrix.pdf"):
    """
    Plots and saves a confusion matrix.
    Saves in PDF format per course requirements (also saves PNG for convenience).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    # Normalize confusion matrix for better color distribution across unbalanced classes
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    sns.heatmap(
        cm_norm,
        cmap="Blues",
        cbar=True,
        square=True,
        xticklabels=False,
        yticklabels=False,
    )

    plt.title(title, fontsize=14, pad=12)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.tight_layout()

    # Save as PDF (required by guidelines)
    pdf_path = save_path.with_suffix(".pdf")
    plt.savefig(pdf_path, format="pdf", dpi=300)

    # Also save PNG for easy viewing in editors
    png_path = save_path.with_suffix(".png")
    plt.savefig(png_path, format="png", dpi=200)

    plt.close()
    return pdf_path


def plot_degradation_curves(df, dataset_name="BTSD", save_dir="results/figures"):
    """
    Plots performance degradation curves (Accuracy and F1 vs. Severity)
    for all corruptions and saves them as PDF and PNG.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    corruptions = df["corruption"].unique()

    # --- 1. Accuracy vs Severity Curve ---
    plt.figure(figsize=(12, 7))
    for corr in corruptions:
        sub = df[df["corruption"] == corr].sort_values("severity")
        plt.plot(sub["severity"], sub["accuracy"], marker='o', linewidth=2, label=corr)

    plt.title(f"Accuracy Degradation under Environmental Corruptions ({dataset_name})", fontsize=14, pad=12)
    plt.xlabel("Severity Level (1 to 5)", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.xticks([1, 2, 3, 4, 5])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=10)
    plt.tight_layout()

    acc_pdf = save_dir / f"{dataset_name.lower()}_accuracy_degradation.pdf"
    plt.savefig(acc_pdf, format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(save_dir / f"{dataset_name.lower()}_accuracy_degradation.png", format="png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- 2. Macro F1 vs Severity Curve ---
    plt.figure(figsize=(12, 7))
    for corr in corruptions:
        sub = df[df["corruption"] == corr].sort_values("severity")
        plt.plot(sub["severity"], sub["macro_f1"], marker='s', linewidth=2, label=corr)

    plt.title(f"Macro F1-Score Degradation under Environmental Corruptions ({dataset_name})", fontsize=14, pad=12)
    plt.xlabel("Severity Level (1 to 5)", fontsize=12)
    plt.ylabel("Macro F1-Score", fontsize=12)
    plt.xticks([1, 2, 3, 4, 5])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=10)
    plt.tight_layout()

    f1_pdf = save_dir / f"{dataset_name.lower()}_f1_degradation.pdf"
    plt.savefig(f1_pdf, format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(save_dir / f"{dataset_name.lower()}_f1_degradation.png", format="png", dpi=200, bbox_inches="tight")
    plt.close()

    return acc_pdf, f1_pdf
