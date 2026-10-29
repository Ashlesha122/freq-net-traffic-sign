
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """One conv -> batchnorm -> relu -> maxpool block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        return self.block(x)


class StreamEncoder(nn.Module):
    """Three ConvBlocks followed by global average pooling and a linear layer."""

    def __init__(self):
        super().__init__()

        # Three conv blocks: 3 -> 32 -> 64 -> 128
        self.conv1 = ConvBlock(3, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)

        # Global average pooling collapses spatial dims to 1x1
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 128)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Pool from (batch, 128, h, w) down to (batch, 128, 1, 1)
        x = self.pool(x)

        # Flatten to (batch, 128)
        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        x = self.fc(x)
        x = self.relu(x)
        return x


class FreqNet(nn.Module):
    """Dual-stream model that combines spatial and frequency features."""

    def __init__(self, num_classes):
        super().__init__()

        # Two separate encoders — they don't share weights
        self.spatial_encoder = StreamEncoder()
        self.frequency_encoder = StreamEncoder()

        # Fuse the two 128-d outputs into one 128-d vector, then classify
        self.fusion = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, spatial, frequency):
        spatial_features = self.spatial_encoder(spatial)
        frequency_features = self.frequency_encoder(frequency)

        # Concatenate along feature dim -> (batch, 256)
        combined = torch.cat([spatial_features, frequency_features], dim=1)

        x = self.fusion(combined)
        x = torch.relu(x)
        logits = self.classifier(x)

        # Return raw logits — softmax is handled by CrossEntropyLoss
        return logits


if __name__ == "__main__":
    model = FreqNet(num_classes=43)

    # Fake batch of 4 images, 3 channels, 64x64
    spatial = torch.randn(4, 3, 64, 64)
    frequency = torch.randn(4, 3, 64, 64)

    output = model(spatial, frequency)
    print(f"Output shape: {output.shape}")