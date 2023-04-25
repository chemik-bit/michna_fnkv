from pathlib import Path
import pickle


def get_stats(path_to_dataset: Path):
    with open(path_to_dataset, "rb") as f:
        data = pickle.load(f)
    labels = data["labels"]
    total_len = len(labels)
    zeros_count = labels.count(0)
    ones_count = labels.count(1)
    print(f"0: {zeros_count}")
    print(f"1: {ones_count}")
    print(f"total: {total_len}")
    print(f"0%: {zeros_count / total_len}")
    print(f"1%: {ones_count / total_len}")


if __name__ == "__main__":
    val_file = Path("../data/splited_voiced/val/voiced_pairs_paths_00002.pickled")
    get_stats(val_file)
