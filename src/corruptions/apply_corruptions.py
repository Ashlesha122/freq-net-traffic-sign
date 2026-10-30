
import argparse
import io
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import scipy.ndimage




def gaussian_noise(image: Image.Image, severity: int = 1) -> Image.Image:
    # Adds Gaussian noise with variance increasing with severity
    scales = [0.06, 0.10, 0.15, 0.22, 0.30]
    scale = scales[severity - 1]

    arr = np.array(image, dtype=np.float32) / 255.0
    noise = np.random.normal(0, scale, arr.shape)
    corrupted = np.clip(arr + noise, 0, 1.0) * 255.0
    return Image.fromarray(corrupted.astype(np.uint8))


def shot_noise(image: Image.Image, severity: int = 1) -> Image.Image:
    # Poisson noise simulating photon noise in low-light camera sensors
    scales = [250, 100, 50, 25, 12]
    c = scales[severity - 1]

    arr = np.array(image, dtype=np.float32) / 255.0
    corrupted = np.clip(np.random.poisson(arr * c) / float(c), 0, 1.0) * 255.0
    return Image.fromarray(corrupted.astype(np.uint8))


def impulse_noise(image: Image.Image, severity: int = 1) -> Image.Image:
    # Salt-and-pepper noise
    amounts = [0.03, 0.06, 0.10, 0.18, 0.28]
    amount = amounts[severity - 1]

    arr = np.array(image, dtype=np.uint8).copy()
    mask = np.random.rand(*arr.shape[:2])

    # Pepper (black) and Salt (white)
    arr[mask < (amount / 2.0)] = 0
    arr[mask > (1.0 - amount / 2.0)] = 255
    return Image.fromarray(arr)


def speckle_noise(image: Image.Image, severity: int = 1) -> Image.Image:
    # Multiplicative noise: x + x * noise
    scales = [0.08, 0.14, 0.20, 0.28, 0.38]
    scale = scales[severity - 1]

    arr = np.array(image, dtype=np.float32) / 255.0
    noise = np.random.normal(0, scale, arr.shape)
    corrupted = np.clip(arr + arr * noise, 0, 1.0) * 255.0
    return Image.fromarray(corrupted.astype(np.uint8))


def defocus_blur(image: Image.Image, severity: int = 1) -> Image.Image:
    # Defocus blur simulated by Gaussian blur kernels
    sigmas = [0.8, 1.3, 1.9, 2.7, 3.6]
    sigma = sigmas[severity - 1]

    arr = np.array(image, dtype=np.float32)
    corrupted = cv2.GaussianBlur(arr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return Image.fromarray(np.clip(corrupted, 0, 255).astype(np.uint8))


def motion_blur(image: Image.Image, severity: int = 1) -> Image.Image:
    # Linear motion blur kernel
    kernel_sizes = [3, 5, 7, 9, 13]
    size = kernel_sizes[severity - 1]

    kernel = np.zeros((size, size))
    kernel[int((size - 1) / 2), :] = np.ones(size)
    kernel = kernel / size

    arr = np.array(image, dtype=np.float32)
    corrupted = cv2.filter2D(arr, -1, kernel)
    return Image.fromarray(np.clip(corrupted, 0, 255).astype(np.uint8))


def zoom_blur(image: Image.Image, severity: int = 1) -> Image.Image:
    # Radial blur simulating camera zoom during capture
    zoom_factors = [1.03, 1.06, 1.10, 1.15, 1.22]
    factor = zoom_factors[severity - 1]

    arr = np.array(image, dtype=np.float32)
    h, w = arr.shape[:2]

    # Crop central region and resize back to original dimensions
    crop_h = int(h / factor)
    crop_w = int(w / factor)
    y1 = (h - crop_h) // 2
    x1 = (w - crop_w) // 2

    zoomed = cv2.resize(arr[y1:y1 + crop_h, x1:x1 + crop_w], (w, h), interpolation=cv2.INTER_LINEAR)
    corrupted = 0.5 * arr + 0.5 * zoomed
    return Image.fromarray(np.clip(corrupted, 0, 255).astype(np.uint8))


def fog(image: Image.Image, severity: int = 1) -> Image.Image:
    # Low-frequency smooth noise blended to mimic atmospheric fog
    opacities = [0.20, 0.35, 0.50, 0.65, 0.80]
    opacity = opacities[severity - 1]

    arr = np.array(image, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]

    # Generate smooth noise map
    raw_noise = np.random.rand(h, w).astype(np.float32)
    fog_map = cv2.GaussianBlur(raw_noise, (21, 21), 6)[..., None]
    fog_map = (fog_map - fog_map.min()) / (fog_map.max() - fog_map.min() + 1e-8)

    # Blend original image with fog color (light gray/white)
    corrupted = (1.0 - opacity) * arr + opacity * (0.8 + 0.2 * fog_map)
    return Image.fromarray((np.clip(corrupted, 0, 1.0) * 255.0).astype(np.uint8))


def snow(image: Image.Image, severity: int = 1) -> Image.Image:
    # Random snowflake flecks + subtle brightness shift
    snow_densities = [0.03, 0.06, 0.10, 0.16, 0.24]
    density = snow_densities[severity - 1]

    arr = np.array(image, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]

    # Create random snowflake points
    snow_mask = (np.random.rand(h, w, 1) < density).astype(np.float32)
    # Blur flakes slightly to look like falling snow
    snow_mask = cv2.GaussianBlur(snow_mask, (3, 3), 0.8)[..., None]

    # Slightly wash out image contrast and add white flakes
    corrupted = arr * 0.85 + 0.10 + snow_mask * 0.65
    return Image.fromarray((np.clip(corrupted, 0, 1.0) * 255.0).astype(np.uint8))


def frost(image: Image.Image, severity: int = 1) -> Image.Image:
    # Ice crystal patterns and frost layer
    intensities = [0.15, 0.28, 0.42, 0.58, 0.72]
    intensity = intensities[severity - 1]

    arr = np.array(image, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]

    # Fine high-frequency noise for frost crystallization
    crystal = cv2.GaussianBlur(np.random.rand(h, w).astype(np.float32), (5, 5), 1.2)[..., None]
    corrupted = (1.0 - intensity) * arr + intensity * (0.65 + 0.35 * crystal)
    return Image.fromarray((np.clip(corrupted, 0, 1.0) * 255.0).astype(np.uint8))


def brightness(image: Image.Image, severity: int = 1) -> Image.Image:
    # Strong sunlight / over-exposure
    factors = [1.2, 1.4, 1.6, 1.8, 2.1]
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factors[severity - 1])


def contrast(image: Image.Image, severity: int = 1) -> Image.Image:
    # Degraded contrast (overcast / washed out lighting)
    factors = [0.75, 0.55, 0.40, 0.25, 0.10]
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factors[severity - 1])


def pixelate(image: Image.Image, severity: int = 1) -> Image.Image:
    # Downsample and nearest-neighbor upsample to simulate low resolution
    sizes = [48, 36, 24, 16, 10]
    down_size = sizes[severity - 1]

    small = image.resize((down_size, down_size), Image.BILINEAR)
    return small.resize(image.size, Image.NEAREST)


def elastic_transform(image: Image.Image, severity: int = 1) -> Image.Image:
    # Elastic distortion / warp
    alphas = [4.0, 7.0, 10.0, 14.0, 18.0]
    alpha = alphas[severity - 1]
    sigma = 3.0

    arr = np.array(image, dtype=np.float32)
    shape = arr.shape[:2]

    dx = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    corrupted = np.zeros_like(arr)
    for c in range(3):
        corrupted[..., c] = scipy.ndimage.map_coordinates(arr[..., c], indices, order=1, mode='reflect').reshape(shape)

    return Image.fromarray(np.clip(corrupted, 0, 255).astype(np.uint8))


def jpeg_compression(image: Image.Image, severity: int = 1) -> Image.Image:
    # High compression artifacts
    qualities = [35, 25, 15, 9, 4]
    quality = qualities[severity - 1]

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# Dictionary of all 15 corruptions
CORRUPTIONS = {
    "gaussian_noise": gaussian_noise,
    "shot_noise": shot_noise,
    "impulse_noise": impulse_noise,
    "speckle_noise": speckle_noise,
    "defocus_blur": defocus_blur,
    "motion_blur": motion_blur,
    "zoom_blur": zoom_blur,
    "snow": snow,
    "fog": fog,
    "frost": frost,
    "brightness": brightness,
    "contrast": contrast,
    "pixelate": pixelate,
    "elastic_transform": elastic_transform,
    "jpeg_compression": jpeg_compression,
}


def apply_corruption(image: Image.Image, corruption_name: str, severity: int = 1) -> Image.Image:
    """Applies a specific corruption at a given severity (1-5)."""
    if corruption_name not in CORRUPTIONS:
        raise ValueError(f"Unknown corruption '{corruption_name}'. Choose from {list(CORRUPTIONS.keys())}")
    if not (1 <= severity <= 5):
        raise ValueError("Severity must be an integer between 1 and 5.")

    return CORRUPTIONS[corruption_name](image, severity)


# -------------------------------------------------------------
# Batch generation for dataset test split
# -------------------------------------------------------------

def generate_corrupted_dataset(dataset_name: str = "btsd", limit: int = None):
    """
    Applies all 15 corruptions at 5 severities to the test split
    and writes them into data/corrupted/<DATASET>/<corruption>/severity_<sev>/.
    """
    import pandas as pd
    from tqdm import tqdm

    name_upper = dataset_name.upper()
    base_dir = Path("data/processed") / name_upper
    test_csv = base_dir / "testing.csv"

    if not test_csv.exists():
        print(f"Error: {test_csv} not found!")
        return

    df = pd.read_csv(test_csv)
    if limit is not None:
        df = df.iloc[:limit]

    print(f"Generating corrupted dataset for {name_upper} ({len(df)} test images)...")
    out_root = Path("data/corrupted") / name_upper

    for corr_name in CORRUPTIONS.keys():
        print(f"\nProcessing corruption: {corr_name}")
        for severity in range(1, 6):
            save_dir = out_root / corr_name / f"severity_{severity}"
            img_dir = save_dir / "images"
            img_dir.mkdir(parents=True, exist_ok=True)

            corrupted_records = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  sev {severity}", leave=False):
                orig_rel_path = row["image_path"]
                full_path = base_dir / orig_rel_path
                label = row["label"]

                img = Image.open(full_path).convert("RGB")
                corr_img = apply_corruption(img, corr_name, severity)

                out_filename = Path(orig_rel_path).name
                save_path = img_dir / out_filename
                corr_img.save(save_path, quality=95)

                corrupted_records.append({
                    "image_path": (Path("images") / out_filename).as_posix(),
                    "label": label,
                })

            csv_out = save_dir / "testing.csv"
            pd.DataFrame(corrupted_records).to_csv(csv_out, index=False)

    print(f"\nAll 15 corruptions generated under {out_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate corrupted traffic sign test sets.")
    parser.add_argument("--dataset", type=str, default="btsd", choices=["gtsrb", "btsd", "ctsd"])
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images for quick testing")
    args = parser.parse_args()

    generate_corrupted_dataset(args.dataset, limit=args.limit)
