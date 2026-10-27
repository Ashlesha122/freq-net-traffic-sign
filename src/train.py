
import argparse
import sys
from pathlib import Path

# Add src/ to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.traffic_sign_dataset import TrafficSignDataset
from models.freq_net import FreqNet


# Dataset folder names use uppercase
DATASET_DIRS = {
    "gtsrb": "GTSRB",
    "btsd": "BTSD",
    "ctsd": "CTSD",
}


def get_num_classes(csv_path):
    dataframe = pd.read_csv(csv_path)
    return dataframe["label"].nunique()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for spatial, frequency, labels in dataloader:
        spatial = spatial.to(device)
        frequency = frequency.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(spatial, frequency)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        # Count correct predictions
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["gtsrb", "btsd", "ctsd"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    # Build paths for whatever dataset was requested
    dataset_dir = DATASET_DIRS[args.dataset]
    base_dir = Path("data/processed") / dataset_dir
    csv_path = base_dir / "training.csv"

    if not csv_path.exists():
        print(f"Training CSV not found: {csv_path}")
        return

    # Figure out how many classes this dataset has from the CSV itself
    num_classes = get_num_classes(csv_path)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Classes: {num_classes}")

    # Load training data
    dataset = TrafficSignDataset(csv_path, base_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Set up model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = FreqNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        loss, accuracy = train_one_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch}/{args.epochs} - loss: {loss:.4f} - accuracy: {accuracy:.4f}")

    # Save the trained model
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{args.dataset}_model.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nModel saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
