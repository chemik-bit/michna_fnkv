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
df = df[df["configfile"] == "h_svk.yaml"][18:]
df_fp_fn = df[["FN", "FP"]]
#df_fp_fn.plot.bar()
x_ticks = [f"model_{i}" for i in range(1, len(df_fp_fn) + 1)]

# Plot the bar chart
ax = df_fp_fn.plot.bar()

# Set custom x_ticks
ax.set_xticklabels(x_ticks)

# Show the plot
plt.savefig("/Users/honzamichna/Desktop/svk.png", dpi=300, bbox_inches='tight')
plt.show()

