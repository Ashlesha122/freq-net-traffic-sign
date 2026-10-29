
import argparse
import sys
from pathlib import Path

# Add src/ to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from src.dataset import TrafficSignDataset
    from src.model import FreqNet
except ModuleNotFoundError:
    from dataset import TrafficSignDataset
    from model import FreqNet


# Dataset folder names use uppercase
DATASET_DIRS = {
    "gtsrb": "GTSRB",
    "btsd": "BTSD",
    "ctsd": "CTSD",
}


def get_num_classes(csv_path):
    dataframe = pd.read_csv(csv_path)
    return dataframe["label"].nunique()


def evaluate(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_labels = []

    # No gradient computation needed during evaluation
    with torch.no_grad():
        for spatial, frequency, labels in dataloader:
            spatial = spatial.to(device)
            frequency = frequency.to(device)

            outputs = model(spatial, frequency)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.tolist())

    return all_labels, all_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["gtsrb", "btsd", "ctsd"])
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    # Build paths
    dataset_dir = DATASET_DIRS[args.dataset]
    base_dir = Path("data/processed") / dataset_dir
    test_csv = base_dir / "testing.csv"
    train_csv = base_dir / "training.csv"

    if not test_csv.exists():
        print(f"Testing CSV not found: {test_csv}")
        return

    # num_classes must match what the model was trained with (from training CSV)
    num_classes = get_num_classes(train_csv)

    # Load the saved checkpoint
    checkpoint_path = Path("checkpoints") / f"{args.dataset}_model.pth"
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Run train.py first")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Device: {device}")

    # Build model and load trained weights
    model = FreqNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Load test data
    dataset = TrafficSignDataset(test_csv, base_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Run evaluation
    labels, predictions = evaluate(model, dataloader, device)

    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="macro", zero_division=0)
    recall = recall_score(labels, predictions, average="macro", zero_division=0)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, predictions, average="weighted", zero_division=0)

    print(f"\nAccuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    # Save results to CSV
    results_dir = Path("results/baseline")
    results_dir.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame([{
        "dataset": args.dataset.upper(),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
    }])

    results_csv = results_dir / f"{args.dataset}_results.csv"
    results.to_csv(results_csv, index=False)
    print(f"\nResults saved to {results_csv}")


if __name__ == "__main__":
    main()
