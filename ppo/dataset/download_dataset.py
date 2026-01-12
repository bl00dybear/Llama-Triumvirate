from datasets import load_dataset
import os


def main():
    dataset_name = "LLM-LAT/harmful-dataset"
    save_path = "./harmful_dataset"

    print(f"Downloading: {dataset_name}...")

    dataset = load_dataset(dataset_name)
    dataset.save_to_disk(save_path)

    print(f"Done. Saved in: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    main()