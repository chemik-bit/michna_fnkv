import os
import json
from matplotlib import pyplot as plt


if os.name == "nt":
    from config import WINDOWS_PATHS as PATHS
else:
    from config import CENTOS_PATHS as PATHS

json_files = list(PATHS["PATH_RESULTS"].glob("*.json"))

for json_file in json_files:
    with open(json_file, "r") as f:
        data = json.load(f)
        plt.figure()
        plt.plot(data["loss"])
        plt.plot(data["val_loss"], "r")
        plt.savefig(PATHS["PATH_RESULTS"].joinpath("imgs", json_file.stem + ".png"))
        plt.close()
