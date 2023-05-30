"""
Module that calculates the correlation between the validation
loss and training loss to evaluate potentially useful models.
If the correlation exceeds 0.5, the name of the file is printed out.
"""
import os
import json
from scipy.stats import pearsonr


if os.name == "nt":
    from config import WINDOWS_PATHS as PATHS
else:
    from config import CENTOS_PATHS as PATHS

for item in PATHS["PATH_RESULTS"].glob("*"):
    if item.is_dir():
        for result in item.glob("*.json"):
            with open(result, "r", encoding="utf-8") as f:
                data = json.load(f)
                if pearsonr(data["loss"], data["val_loss"])[0] > 0.5:
                    print(result)
