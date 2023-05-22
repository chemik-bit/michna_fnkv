from pathlib import Path
import shutil
import csv


if __name__ == "__main__":
    results_path = Path("..", "data", "results")
    results_csv = list(results_path.glob("*.csv"))
    created_dirs = []

    # with open(results_path.joinpath(""))
    for result_file in results_csv:
        print(result_file)
        with open(result_file, "r") as f:
            data = csv.DictReader(f)

            for row in data:
                print(f"{row['history_file']} --> {row['configfile']}")
                if row["configfile"][:-5] not in created_dirs:
                    print(row["configfile"][:-5])
                    results_path.joinpath(row["configfile"][:-5]).mkdir()
                    created_dirs.append(row["configfile"][:-5])
                shutil.copy(results_path.joinpath(row["history_file"]), results_path.joinpath(row["configfile"][:-5], row["history_file"]))
