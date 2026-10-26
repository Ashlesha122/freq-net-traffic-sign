
from pathlib import Path
import pandas as pd
from PIL import Image

DATASETS = {
    "GTSRB": Path("data/processed/GTSRB"),
    "BTSD": Path("data/processed/BTSD"),
    "CTSD": Path("data/processed/CTSD"),
}

IMAGE_SIZE = (64, 64)


def validate_dataset(name, dataset_dir):
    print(f"\n{'=' * 55}")
    print(f"Validating {name}")
    print("=" * 55)

    errors = []
    total_checked = 0
    total_rows = 0

    for split in ["training", "testing"]:
        csv_path = dataset_dir / f"{split}.csv"

        if not csv_path.exists():
            errors.append(f"Missing CSV: {csv_path}")
            print(f"\n{split}: CSV not found")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as error:
            errors.append(f"Cannot read {csv_path}: {error}")
            continue

        print(f"\n{split.capitalize()} rows: {len(df)}")
        total_rows += len(df)

        required_columns = {"image_path", "label"}

        if not required_columns.issubset(df.columns):
            errors.append(
                f"{csv_path} must contain columns: image_path,label"
            )
            continue

        if df.empty:
            errors.append(f"Empty CSV: {csv_path}")
            continue

        if df["image_path"].isna().any():
            errors.append(f"Empty image paths in {csv_path}")
            continue

        if df["label"].isna().any():
            errors.append(f"Empty labels in {csv_path}")
            continue

        if not pd.api.types.is_numeric_dtype(df["label"]):
            errors.append(f"Non-numeric labels in {csv_path}")
            continue

        for relative_path in df["image_path"]:
            image_path = dataset_dir / str(relative_path)

            if not image_path.is_file():
                errors.append(f"Missing image: {image_path}")
                continue

            try:
                # Verify the file is readable and not corrupted.
                with Image.open(image_path) as image:
                    image.verify()

                # Reopen after verify to inspect size and color mode.
                with Image.open(image_path) as image:
                    if image.size != IMAGE_SIZE:
                        errors.append(
                            f"Wrong size: {image_path} ({image.size})"
                        )

                    if image.mode != "RGB":
                        errors.append(
                            f"Wrong color mode: {image_path} ({image.mode})"
                        )

                total_checked += 1

            except Exception as error:
                errors.append(
                    f"Unreadable image: {image_path} ({error})"
                )

    print(f"\nTotal CSV rows: {total_rows}")
    print(f"Images checked successfully: {total_checked}")
    print(f"Errors found: {len(errors)}")

    if errors:
        print("\nFirst 20 errors:")
        for error in errors[:20]:
            print("-", error)

        if len(errors) > 20:
            print(f"... and {len(errors) - 20} more errors")

        return False

    print("PASS: CSVs and processed images passed all checks.")
    return True


if __name__ == "__main__":
    results = {}

    for name, dataset_dir in DATASETS.items():
        results[name] = validate_dataset(name, dataset_dir)

    print(f"\n{'=' * 55}")
    print("FINAL VALIDATION SUMMARY")
    print("=" * 55)

    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{name}: {status}")

    if all(results.values()):
        print("\nAll three datasets passed validation.")
    else:
        print("\nSome datasets failed. Review the errors above.")
