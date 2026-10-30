
import argparse
import sys
from pathlib import Path

# Add src/ to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from src.dataset import TrafficSignDataset
    from src.model import FreqNet
    from src.corruptions.apply_corruptions import CORRUPTIONS, apply_corruption
    from src.utils.metrics import compute_metrics, get_confusion_matrix, log_robustness_result
    from src.utils.visualization import plot_confusion_matrix, plot_degradation_curves
except ModuleNotFoundError:
    from dataset import TrafficSignDataset
    from model import FreqNet
    from corruptions.apply_corruptions import CORRUPTIONS, apply_corruption
    from utils.metrics import compute_metrics, get_confusion_matrix, log_robustness_result
    from utils.visualization import plot_confusion_matrix, plot_degradation_curves


DATASET_DIRS = {
    "gtsrb": "GTSRB",
    "btsd": "BTSD",
    "ctsd": "CTSD",
}


def get_num_classes(csv_path):
    dataframe = pd.read_csv(csv_path)
    return dataframe["label"].nunique()


def evaluate_dataloader(model, dataloader, device):
    """Evaluates model over a standard TrafficSignDataset dataloader."""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for spatial, frequency, labels in dataloader:
            spatial = spatial.to(device)
            frequency = frequency.to(device)

            outputs = model(spatial, frequency)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.tolist())

    return all_labels, all_predictions


def evaluate_with_corruption(model, raw_images, labels, dataset_helper, corruption_name, severity, device, batch_size=64):
    """
    Applies corruption on-the-fly to loaded test images and evaluates FreqNet.
    Fast and avoids saving 180,000 files to disk while computing exact metrics.
    """
    model.eval()
    all_predictions = []

    num_samples = len(raw_images)
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_imgs = raw_images[i:i + batch_size]

            spatial_tensors = []
            freq_tensors = []

            for img in batch_imgs:
                # 1. Apply corruption at specified severity
                corrupted_img = apply_corruption(img, corruption_name, severity)

                # 2. Get spatial tensor (ImageNet normalized)
                sp = dataset_helper.spatial_transform(corrupted_img)

                # 3. Get frequency tensor (FFT log-magnitude min-max normalized)
                fr = dataset_helper._compute_frequency(corrupted_img)

                spatial_tensors.append(sp)
                freq_tensors.append(fr)

            spatial_batch = torch.stack(spatial_tensors).to(device)
            freq_batch = torch.stack(freq_tensors).to(device)

            outputs = model(spatial_batch, freq_batch)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_predictions.extend(preds)

    return labels, all_predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate FreqNet on clean baseline and environmental corruptions.")
    parser.add_argument("--dataset", required=True, choices=["gtsrb", "btsd", "ctsd"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--robustness", action="store_true", help="Run full robustness evaluation (15 corruptions x 5 severities)")
    args = parser.parse_args()

    # Paths
    dataset_name = args.dataset.upper()
    dataset_dir = DATASET_DIRS[args.dataset]
    base_dir = Path("data/processed") / dataset_dir
    test_csv = base_dir / "testing.csv"
    train_csv = base_dir / "training.csv"

    if not test_csv.exists():
        print(f"Testing CSV not found: {test_csv}")
        return

    num_classes = get_num_classes(train_csv)
    checkpoint_path = Path("checkpoints") / f"{args.dataset}_model.pth"

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train the model first using src/train.py")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dataset: {dataset_name} ({num_classes} classes)")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {checkpoint_path}")

    # Build model and load trained weights
    model = FreqNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Helper dataset object for transforms
    clean_dataset = TrafficSignDataset(test_csv, base_dir)
    clean_loader = DataLoader(clean_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ---------------------------------------------------------
    # 1. Clean Baseline Evaluation
    # ---------------------------------------------------------
    print("\n--- Evaluating Clean Baseline ---")
    clean_labels, clean_preds = evaluate_dataloader(model, clean_loader, device)
    clean_metrics = compute_metrics(clean_labels, clean_preds)

    for k, v in clean_metrics.items():
        print(f"{k.capitalize():12s}: {v:.4f}")

    # Save baseline metrics
    baseline_dir = Path("results/baseline")
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_df = pd.DataFrame([{
        "dataset": dataset_name,
        **clean_metrics,
    }])
    baseline_csv = baseline_dir / f"{args.dataset}_results.csv"
    baseline_df.to_csv(baseline_csv, index=False)
    print(f"Baseline results saved to {baseline_csv}")

    # Save clean confusion matrix (PDF + PNG)
    cm_clean = get_confusion_matrix(clean_labels, clean_preds, num_classes=num_classes)
    figures_dir = Path("results/figures")
    clean_cm_path = figures_dir / f"{args.dataset}_confusion_matrix_clean.pdf"
    plot_confusion_matrix(cm_clean, title=f"Clean Baseline Confusion Matrix ({dataset_name})", save_path=clean_cm_path)
    print(f"Clean confusion matrix saved to {clean_cm_path}")

    # ---------------------------------------------------------
    # 2. Full Robustness Evaluation (if --robustness is passed)
    # ---------------------------------------------------------
    if not args.robustness:
        print("\nTip: Pass --robustness to run all 15 environmental corruptions at 5 severity levels.")
        return

    print("\n=======================================================")
    print(f"RUNNING ROBUSTNESS TESTING: 15 CORRUPTIONS x 5 SEVERITIES")
    print("=======================================================")

    # Load all raw test images into memory once for high speed
    test_df = pd.read_csv(test_csv)
    raw_images = []
    labels = []
    print(f"Preloading {len(test_df)} test images for evaluation...")
    for _, row in test_df.iterrows():
        img_path = base_dir / row["image_path"]
        raw_images.append(Image.open(img_path).convert("RGB"))
        labels.append(int(row["label"]))

    robustness_log_path = Path("results/robustness_logs") / f"{args.dataset}_robustness.csv"
    # Remove existing log to write fresh results
    if robustness_log_path.exists():
        robustness_log_path.unlink()

    records = []

    for corr_name in CORRUPTIONS.keys():
        print(f"\nEvaluating Corruption: {corr_name}")
        for severity in range(1, 6):
            corr_labels, corr_preds = evaluate_with_corruption(
                model=model,
                raw_images=raw_images,
                labels=labels,
                dataset_helper=clean_dataset,
                corruption_name=corr_name,
                severity=severity,
                device=device,
                batch_size=args.batch_size,
            )

            metrics = compute_metrics(corr_labels, corr_preds)
            log_robustness_result(robustness_log_path, dataset_name, corr_name, severity, metrics)

            records.append({
                "dataset": dataset_name,
                "corruption": corr_name,
                "severity": severity,
                **metrics,
            })

            print(f"  Severity {severity}/5 -> Accuracy: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f}")

            # At maximum severity (5), plot the confusion matrix (PDF + PNG)
            if severity == 5:
                cm_sev5 = get_confusion_matrix(corr_labels, corr_preds, num_classes=num_classes)
                cm_path = figures_dir / f"{args.dataset}_cm_{corr_name}_sev5.pdf"
                plot_confusion_matrix(
                    cm_sev5,
                    title=f"{corr_name.replace('_', ' ').title()} (Severity 5) Confusion Matrix",
                    save_path=cm_path,
                )

    print(f"\nAll 75 robustness evaluations completed!")
    print(f"Full results logged to: {robustness_log_path}")

    # ---------------------------------------------------------
    # 3. Plot Performance Degradation Curves
    # ---------------------------------------------------------
    print("\nGenerating performance degradation curves (PDF format)...")
    rob_df = pd.DataFrame(records)
    acc_pdf, f1_pdf = plot_degradation_curves(rob_df, dataset_name=dataset_name, save_dir=figures_dir)
    print(f"Accuracy curves saved to: {acc_pdf}")
    print(f"Macro F1 curves saved to: {f1_pdf}")


if __name__ == "__main__":
    main()
