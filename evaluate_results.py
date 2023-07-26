"""
Module to check results against Laura Verdes article
"""
import os
import json


LAURA_RESULTS = {"f1": 72.99 / 100,
                 "precision": 80.65 / 100,
                 "recall": 66.67 / 100,
                 "accuracy": 64.42 / 100,
                 "specificity": 58.62 / 100,
                 "auc": 0.626}
LAURA_RESULTS = {"f1": 80.00 / 100,
                 "precision": 74.44 / 100,
                 "recall": 86.45 / 100,
                 "accuracy": 73.93 / 100,
                 "specificity":   54.9 / 100,
                 "auc": 0.707}

def compute_metrics(history: dict, filename):
    for idx, tp in enumerate(history["val_TP"]):
        try:
            results = {"f1": 2 * tp / (2 * tp + history["val_FP"][idx] + history["val_FN"][idx]),
                     "precision": tp / (tp + history["val_FP"][idx]),
                     "recall": tp / (tp + history["val_FN"][idx]),
                     "accuracy": (tp + history["val_TN"][idx]) / (history["val_FP"][idx] + history["val_FN"][idx] + tp + history["val_TN"][idx]),
                     "specificity": history["val_TN"][idx] / (history["val_TN"][idx] + history["val_FP"][idx]),
                     "auc": history["val_AUC"][idx]}
            bool_result = [True if results[key] > LAURA_RESULTS[key] - 0.0 else False for key in results.keys()]
            if all(bool_result):
                epoch_result = [f"{key}: {results[key] > LAURA_RESULTS[key]} " for key in results.keys()]
                print(f"{filename.name} - {epoch_result} - epoch: {idx}")
                print(results)
                print("------------------------------------------------------------------------------------------")
        except ZeroDivisionError:
            pass


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
    print("*********************************************************")
    print("*********************************************************")
    print(f"Processing {directory}")


    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)
            #print(f"evaluating {json_file.name}....")
            compute_metrics(data, json_file)


