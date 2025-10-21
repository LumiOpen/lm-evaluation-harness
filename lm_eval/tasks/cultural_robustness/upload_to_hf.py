"""Script to upload cultural robustness data to HuggingFace Hub."""

import argparse
from pathlib import Path

from datasets import DatasetDict, load_dataset


def upload_dataset(username: str, dataset_name: str = "cultural-robustness"):
    """Upload cultural robustness dataset to HuggingFace Hub.

    Args:
        username: HuggingFace username
        dataset_name: Name for the dataset (default: cultural-robustness)
    """
    data_dir = Path(__file__).parent / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Load all files
    specific_files = sorted(str(p) for p in data_dir.glob("specific_*.jsonl"))
    unspecific_files = sorted(str(p) for p in data_dir.glob("unspecific_*.jsonl"))

    if not specific_files or not unspecific_files:
        raise FileNotFoundError("Data files not found")

    print(
        f"Found {len(specific_files)} specific files and {len(unspecific_files)} unspecific files"
    )

    # Load datasets
    specific_dataset = load_dataset("json", data_files={"train": specific_files})
    unspecific_dataset = load_dataset("json", data_files={"train": unspecific_files})

    # Create dataset dict with both splits
    dataset_dict = DatasetDict(
        {
            "specific": specific_dataset["train"],
            "unspecific": unspecific_dataset["train"],
        }
    )

    # Upload to HuggingFace
    repo_id = f"{username}/{dataset_name}"
    print(f"Uploading to {repo_id}...")

    dataset_dict.push_to_hub(
        repo_id,
        private=False,  # Make it public
        commit_message="Upload cultural robustness evaluation dataset",
    )

    print(f"Successfully uploaded to https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload cultural robustness data to HuggingFace Hub"
    )
    parser.add_argument("username", help="Your HuggingFace username")
    parser.add_argument(
        "--dataset-name",
        default="cultural-robustness",
        help="Name for the dataset (default: cultural-robustness)",
    )

    args = parser.parse_args()

    upload_dataset(args.username, args.dataset_name)


if __name__ == "__main__":
    main()
