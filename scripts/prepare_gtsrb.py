
from pathlib import Path
import pandas as pd
from PIL import Image

RAW_DIR = Path("data/raw/GTSRB/data")
OUT_DIR = Path("data/processed/GTSRB")
IMAGE_SIZE = (64, 64)

DATASETS = {
    "training": {
        "csv": RAW_DIR / "Train.csv",
        "output": OUT_DIR / "images" / "training",
        "save_csv": OUT_DIR / "training.csv",
    },
    "testing": {
        "csv": RAW_DIR / "Test.csv",
        "output": OUT_DIR / "images" / "testing",
        "save_csv": OUT_DIR / "testing.csv",
    },
}


def process_split(config):
    config["output"].mkdir(parents=True, exist_ok=True)
    annotations = pd.read_csv(config["csv"])

    rows = []
    errors = 0

    for _, row in annotations.iterrows():
        relative_path = str(row["Path"]).replace("\\", "/")

        # Original paths begin with /code/
        if relative_path.startswith("/code/"):
            relative_path = relative_path[len("/code/"):]

        image_path = RAW_DIR / relative_path
        label = int(row["ClassId"])

        if not image_path.exists():
            print(f"Missing image: {image_path}")
            errors += 1
            continue

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")

                x1 = max(0, min(int(row["Roi.X1"]), image.width))
                y1 = max(0, min(int(row["Roi.Y1"]), image.height))
                x2 = max(0, min(int(row["Roi.X2"]), image.width))
                y2 = max(0, min(int(row["Roi.Y2"]), image.height))

                if x2 <= x1 or y2 <= y1:
                    print(f"Invalid bounding box: {image_path.name}")
                    errors += 1
                    continue

                image = image.crop((x1, y1, x2, y2))
                image = image.resize(IMAGE_SIZE)

                output_path = config["output"] / image_path.name
                image.save(output_path)

            rows.append({
                "image_path": output_path.relative_to(OUT_DIR).as_posix(),
                "label": label
            })

        except Exception as error:
            print(f"Error processing {image_path}: {error}")
            errors += 1

    dataframe = pd.DataFrame(rows, columns=["image_path", "label"])
    dataframe.to_csv(config["save_csv"], index=False)

    print(f"\nCSV created: {config['save_csv']}")
    print(f"Images processed: {len(rows)}")
    print(f"Errors: {errors}")
    print(f"Classes found: {dataframe['label'].nunique()}")


if __name__ == "__main__":
    for split_name, config in DATASETS.items():
        print(f"\nProcessing {split_name} dataset...")
        process_split(config)
