import os
import json
from matplotlib import pyplot as plt
from pathlib import Path

if os.name == "nt":
    from config import WINDOWS_PATHS as PATHS
else:
    from config import CENTOS_PATHS as PATHS


result_dirs = []
#for item in PATHS["PATH_RESULTS"].iterdir():
print(PATHS["PATH_RESULTS"].parent.parent.parent.parent)
for item in PATHS["PATH_RESULTS"].parent.parent.parent.parent.joinpath("./h_svk").iterdir():
    if not (".") in str(item.name):
        result_dirs.append(item)

for directory in result_dirs:
    print(directory)
    json_files = list(directory.glob("*.json"))
    print(f"Processing {directory}")
    directory.joinpath("imgs").mkdir(exist_ok=True)
    for json_file in json_files:
        if not directory.joinpath("imgs", json_file.stem + ".png").exists():
            with open(json_file, "r") as f:
                data = json.load(f)
                plt.figure()
                plt.plot(data["loss"])
                plt.plot(data["val_loss"], "r")
                plt.savefig(directory.joinpath("imgs", json_file.stem + ".png"))
                plt.close()
