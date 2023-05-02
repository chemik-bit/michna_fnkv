laura_results = {"f1": 72.99,
                 "precision": 80.65,
                 "recall": 66.67,
                 "accuracy": 64.42,
                 "specificity": 58.62,
                 "auc": 0.626}

TP = 120
TN = 28
FP = 29
FN = 31

our_results = {"f1": 100 * 2 * TP / (2 * TP + TN + FP + FN),
               "precision":100 * TP / (TP + FP),
               "recall": 100 *TP / (TP + FN),
               "accuracy": 100 *(TP + TN) / (TP + TN + FP + FN),
               "specificity": 100 *TN / (TN + FP),
               "auc": 0.628}

for key in our_results:
    if our_results[key] > laura_results[key]:
        print(f"{key} - pass {our_results[key]} vs {laura_results[key]}")
    else:
        print(f"{key} - failed  {our_results[key]} vs {laura_results[key]}")