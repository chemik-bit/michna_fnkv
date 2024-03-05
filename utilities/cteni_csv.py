import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dtypes = {'model' : str, 'benchmark_value' : float, 'TP' : float, 'TN' : float, 'FP' : float, \
          'FN' : float, 'AUC' : float,'training_set' : str, 'val_set' : str, 'loss' : str, \
              'optimizer' : str, 'lr' : float, 'epochs' : float,
       'batch_size' : float, 'balance' : str, 'fft_len' : float, 'fft_overlap' : float, \
           'chunks' : float,'image_size' : str, 'val_ratio' : float, 'resampling': str, \
               'BENCHMARK_AUC' : float, 'VAL_AUC_MAX' : float,'AUC_MAX' : float, 'history_file' : str, 'configfile' : str}
    
    
file = pd.read_csv("/Users/honzamichna/Documents/446-a336-j2.vscht.cz_results_v3.csv", dtype=dtypes)

df = pd.DataFrame(file)
#print(df)
#print(df["configfile"])
df = df[df["configfile"] == "h_svk.yaml"]
df_fp_fn = df[["FN", "FP"]]
df_fp_fn.plot.bar()
