import os
import json

LAURA_RESULTS = {"f1": 72.99 / 100,
                 "precision": 80.65 / 100,
                 "recall": 66.607 / 100,
                 "accuracy": 64.42 / 100,
                 "specificity": 58.62 / 100,
                 "auc": 0.626}

def compute_metrics(history: dict):
    for idx, tp in enumerate(history["TP"]):
        results = {"f1": 2 * tp / (2 * tp + history["FP"][idx] + history["FN"][idx]),
                 "precision": tp / (tp + history["FP"][idx]),
                 "recall": tp / (tp + history["FN"][idx]),
                 "accuracy": (tp + history["TN"][idx]) / (history["FP"][idx] + history["FN"][idx] + tp + history["TN"][idx]),
                 "specificity": history["TN"][idx] / (history["TN"][idx] + history["FP"][idx]),
                 "auc": history["AUC"][idx]}
        bool_result = [True if results[key] > LAURA_RESULTS[key] else False for key in results.keys()]
        if all(bool_result):
            epoch_result = [f"{key}: {results[key] > LAURA_RESULTS[key]} " for key in results.keys()]
            print(epoch_result)
if os.name == "nt":
    from config import WINDOWS_PATHS as PATHS
else:
    from config import CENTOS_PATHS as PATHS

json_files = list(PATHS["PATH_RESULTS"].glob("*.json"))

for json_file in json_files:
    with open(json_file, "r") as f:
        data = json.load(f)
        print(f"evaluating {json_file.name}....")
        compute_metrics(data)


