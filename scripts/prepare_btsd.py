
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

RAW_DIR = Path("data/raw/BTSD")
OUTPUT_DIR = Path("data/processed/BTSD")
IMAGE_SIZE = (64, 64)

def process_split(split_name):
    source_dir = RAW_DIR / split_name
    output_images = OUTPUT_DIR / "images" / split_name
    output_images.mkdir(parents=True, exist_ok=True)

    records = []
    skipped = 0

    # Each numbered folder represents a class
    class_folders = sorted(
        [p for p in source_dir.iterdir() if p.is_dir()]
    )

    for class_folder in class_folders:
        if not class_folder.name.isdigit():
            continue

        label = int(class_folder.name)

        image_files = [
            p for p in class_folder.iterdir()
            if p.is_file() and p.suffix.lower() in
            [".ppm", ".png", ".jpg", ".jpeg", ".bmp"]
        ]

        for image_path in tqdm(
            image_files, desc=f"{split_name}/{class_folder.name}"
        ):
            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    image = image.resize(IMAGE_SIZE)

                    output_name = (
                        f"class_{label:05d}_{image_path.stem}.jpg"
                    )
                    saved_path = output_images / output_name
                    image.save(saved_path, quality=95)

                records.append({
                    "image_path": saved_path.relative_to(
                        OUTPUT_DIR
                    ).as_posix(),
                    "label": label
                })

            except Exception as error:
                skipped += 1
                print(f"Skipped {image_path}: {error}")

    csv_path = OUTPUT_DIR / f"{split_name.lower()}.csv"
    pd.DataFrame(
        records, columns=["image_path", "label"]
    ).to_csv(csv_path, index=False)

    print(f"\n{split_name} images processed: {len(records)}")
    print(f"Images skipped: {skipped}")
    print(f"CSV saved to: {csv_path}")

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    process_split("Training")
    process_split("Testing")
