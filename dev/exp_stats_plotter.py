import os
import json
from matplotlib import pyplot as plt


if os.name == "nt":
    from config import WINDOWS_PATHS as PATHS
else:
    from config import CENTOS_PATHS as PATHS

json_files = list(PATHS["PATH_RESULTS"].glob("*.json"))
",f1755636-1f2d-4b9e-bc04-296cef8227b9.json"
"69add14c-bddb-48f9-be60-e4260517b86b.json"
with open(PATHS["PATH_RESULTS"].joinpath("f1755636-1f2d-4b9e-bc04-296cef8227b9.json"), "r") as f:
    data = json.load(f)
    plt.figure()
    plt.plot(data["val_TP"], label="TP")
    plt.plot(data["val_TN"], "r", label="TN")
    plt.plot(data["val_FP"], "g", label="FP")
    plt.plot(data["val_FN"], "k", label="FN")
    plt.legend()
    plt.show()


