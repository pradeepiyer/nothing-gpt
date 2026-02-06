"""Download Seinfeld scripts dataset from Kaggle."""

import shutil
from pathlib import Path

import kagglehub

DATASET = "thec03u5/seinfeld-chronicles"
RAW_DIR = Path(__file__).parent.parent.parent.parent / "data" / "raw"


def download(output_dir: Path = RAW_DIR) -> Path:
    """Download the Seinfeld scripts CSV from Kaggle and copy to output_dir."""
    dataset_path = Path(kagglehub.dataset_download(DATASET))
    output_dir.mkdir(parents=True, exist_ok=True)

    # The dataset contains a scripts.csv file
    src = dataset_path / "scripts.csv"
    if not src.exists():
        # Check for any CSV file in the download
        csvs = list(dataset_path.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in {dataset_path}")
        src = csvs[0]

    dest = output_dir / "scripts.csv"
    shutil.copy2(src, dest)
    print(f"Downloaded dataset to {dest}")
    return dest


if __name__ == "__main__":
    download()
