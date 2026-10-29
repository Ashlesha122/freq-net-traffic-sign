
# Re-export from the datasets package for convenience
try:
    from src.datasets.traffic_sign_dataset import TrafficSignDataset
except ModuleNotFoundError:
    from datasets.traffic_sign_dataset import TrafficSignDataset
