
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# ImageNet mean and std — standard values used for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TrafficSignDataset(Dataset):
    """Loads traffic sign images and returns spatial + frequency tensors."""

    def __init__(self, csv_path, base_dir):
        self.base_dir = Path(base_dir)
        self.dataframe = pd.read_csv(csv_path)

        # Normalize spatial images the standard way
        self.spatial_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = self.base_dir / row["image_path"]
        label = int(row["label"])

        image = Image.open(image_path).convert("RGB")

        # Spatial tensor — just convert to tensor and normalize
        spatial = self.spatial_transform(image)

        # Frequency tensor — compute FFT magnitude spectrum per channel
        frequency = self._compute_frequency(image)

        return spatial, frequency, label

    def _compute_frequency(self, image):
        # Convert image to numpy array, shape (64, 64, 3), values 0-255
        pixels = np.array(image, dtype=np.float32)

        channels = []
        for c in range(3):
            channel = pixels[:, :, c]

            # 2D FFT on this color channel
            fft = np.fft.fft2(channel)

            # Shift so low frequencies are in the center
            fft_shifted = np.fft.fftshift(fft)

            # Take magnitude and log-scale it to compress the range
            magnitude = np.abs(fft_shifted)
            magnitude = np.log(1 + magnitude)

            channels.append(magnitude)

        # Stack channels into (3, 64, 64) and convert to tensor
        freq_array = np.stack(channels, axis=0)
        freq_tensor = torch.from_numpy(freq_array)

        # Min-max normalize to [0, 1] independently from spatial pixel statistics
        min_val = freq_tensor.min()
        max_val = freq_tensor.max()
        freq_tensor = (freq_tensor - min_val) / (max_val - min_val + 1e-8)

        return freq_tensor


if __name__ == "__main__":
    # Quick test — try loading a GTSRB sample if data exists
    csv_path = Path("data/processed/GTSRB/training.csv")
    base_dir = Path("data/processed/GTSRB")

    if csv_path.exists():
        dataset = TrafficSignDataset(csv_path, base_dir)
        print(f"Dataset size: {len(dataset)}")

        spatial, frequency, label = dataset[0]
        print(f"Spatial shape: {spatial.shape}")
        print(f"Frequency shape: {frequency.shape}")
        print(f"Label: {label}")
    else:
        print("GTSRB training CSV not found — run data preparation first")
