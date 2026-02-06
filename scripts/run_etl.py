"""Run the full data pipeline: download → parse → format."""

from nothing_gpt.data.download import download
from nothing_gpt.data.format import format_dataset
from nothing_gpt.data.parse import parse_csv, save_dialogues


def main() -> None:
    print("=== Step 1: Download dataset ===")
    csv_path = download()

    print("\n=== Step 2: Parse dialogues ===")
    episodes = parse_csv(csv_path)
    dialogues_path = save_dialogues(episodes)

    print("\n=== Step 3: Format training data ===")
    format_dataset(dialogues_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
