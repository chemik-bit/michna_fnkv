import os
import json
from matplotlib import pyplot as plt


if os.name == "nt":
    from config import WINDOWS_PATHS as PATHS
else:
    from config import CENTOS_PATHS as PATHS

#json_files = list(PATHS["PATH_RESULTS"].joinpath("h_conv_rectangular_noresample_fft25ms_overlap50_lr000001").glob("*.json"))
json_files = list(PATHS["PATH_RESULTS"].parent.parent.parent.parent.joinpath("./h_svk").glob("*.json"))

with open(PATHS["PATH_RESULTS"].parent.parent.parent.parent.joinpath("./h_svk").joinpath("28c3fe00-1239-4286-b4d2-1d4c890e2da0.json"), "r") as f:
    data = json.load(f)
    plt.figure()
    plt.plot(data["val_TP"], label="TP")
    plt.plot(data["val_TN"], "r", label="TN")
    plt.plot(data["val_FP"], "g", label="FP")
    plt.plot(data["val_FN"], "k", label="FN")
    plt.plot([idx + value for idx, value in zip(data["val_FN"], data["val_FP"])], "c", label="Benchmark")
    plt.legend()
    plt.show()

    plt.figure()
    history = data
    results = {"f1": [],
                "precision": [],
                "recall": [],
                "accuracy": [],
                "specificity": [],
                "auc": []}
    for idx, tp in enumerate(data["val_TP"]):
        try:
            results["f1"].append(2 * tp / (2 * tp + history["val_FP"][idx] + history["val_FN"][idx]))
            results["precision"].append(tp / (tp + history["val_FP"][idx]))
            results["recall"].append(tp / (tp + history["val_FN"][idx]))
            results["accuracy"].append((tp + history["val_TN"][idx]) / (history["val_FP"][idx] + history["val_FN"][idx] + tp + history["val_TN"][idx]))
            results["specificity"].append(history["val_TN"][idx] / (history["val_TN"][idx] + history["val_FP"][idx]))
            results["auc"].append(history["val_AUC"][idx])
        except:
            pass
    plt.plot(results["f1"], "r", label="F1")
    plt.plot(results["precision"], "b", label="Precision")
    plt.plot(results["recall"], "k", label="Recall")
    plt.plot(results["accuracy"], "g", label="Acc")
    plt.plot(results["specificity"], "c", label="specificity")
    plt.plot(results["auc"], "m", label="auc")
    plt.legend()
    plt.show()