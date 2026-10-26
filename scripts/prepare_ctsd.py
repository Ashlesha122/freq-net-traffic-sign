
from pathlib import Path
import pandas as pd
from PIL import Image

# Project directories
RAW_DIR = Path("data/raw/CTSD")
OUT_DIR = Path("data/processed/CTSD")
IMAGE_SIZE = (64, 64)

# Dataset folders and annotation files
DATASETS = {
    "training": {
        "images": RAW_DIR / "CTSD-Train",
        "annotations": RAW_DIR / "CTSD-Train Annotation" / "TsignRecgTrain4170Annotation.txt",
        "output": OUT_DIR / "images" / "training",
        "csv": OUT_DIR / "training.csv",
    },
    "testing": {
        "images": RAW_DIR / "CTSD-Test",
        "annotations": RAW_DIR / "CTSD-Test Annotation" / "TsignRecgTest1994Annotation.txt",
        "output": OUT_DIR / "images" / "testing",
        "csv": OUT_DIR / "testing.csv",
    },
}


def process_split(config):
    config["output"].mkdir(parents=True, exist_ok=True)
    rows = []
    errors = 0

    with config["annotations"].open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            fields = line.strip().split(";")

            # Remove the empty field caused by the trailing semicolon
            fields = [field for field in fields if field != ""]

            if len(fields) != 8:
                print(f"Skipping invalid annotation at line {line_number}")
                errors += 1
                continue

            filename = fields[0]
            x1, y1, x2, y2, label = map(int, fields[3:8])

            image_path = config["images"] / filename

            if not image_path.exists():
                print(f"Missing image: {image_path}")
                errors += 1
                continue

            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")

                    # Clamp bounding-box coordinates to image boundaries
                    x1 = max(0, min(x1, image.width))
                    x2 = max(0, min(x2, image.width))
                    y1 = max(0, min(y1, image.height))
                    y2 = max(0, min(y2, image.height))

                    if x2 <= x1 or y2 <= y1:
                        print(f"Invalid bounding box: {filename}")
                        errors += 1
                        continue

                    # Crop the traffic sign and resize it
                    cropped = image.crop((x1, y1, x2, y2))
                    cropped = cropped.resize(IMAGE_SIZE)

                    output_path = config["output"] / filename
                    cropped.save(output_path)

                # Paths in CSV are relative to data/processed/CTSD
                relative_path = output_path.relative_to(OUT_DIR).as_posix()
                rows.append({
                    "image_path": relative_path,
                    "label": label,
                })

            except Exception as error:
                print(f"Error processing {filename}: {error}")
                errors += 1

    dataframe = pd.DataFrame(rows, columns=["image_path", "label"])
    dataframe.to_csv(config["csv"], index=False)

    print(f"\nCSV created: {config['csv']}")
    print(f"Images processed: {len(rows)}")
    print(f"Errors: {errors}")
    print(f"Classes found: {dataframe['label'].nunique()}")


if __name__ == "__main__":
    for split_name, config in DATASETS.items():
        print(f"\nProcessing {split_name} dataset...")
        process_split(config)
