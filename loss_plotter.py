import os
import json
from matplotlib import pyplot as plt


if os.name == "nt":
    from config import WINDOWS_PATHS as PATHS
else:
    from config import CENTOS_PATHS as PATHS


result_dirs = []
for item in PATHS["PATH_RESULTS"].iterdir():
    if not (".") in str(item.name):
        result_dirs.append(item)

for directory in result_dirs:
    json_files = list(directory.glob("*.json"))
    print(f"Processing {directory}")
    directory.joinpath("imgs").mkdir()
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)
            plt.figure()
            plt.plot(data["loss"])
            plt.plot(data["val_loss"], "r")
            plt.savefig(directory.joinpath("imgs", json_file.stem + ".png"))
            plt.close()
