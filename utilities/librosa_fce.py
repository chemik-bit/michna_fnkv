import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

scale_file = "/Users/honzamichna/Documents/GitHub/michna_fnkv/data/wav/svdadult/1/svdadult0363_unhealthy_50000_00000.wav"
# load audio files with librosa
scale, sr = librosa.load(scale_file)

FRAME_SIZE = 1250
HOP_SIZE = 625

S_scale = librosa.stft(scale, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
S_scale.shape
S_scale

type(S_scale[0][0])

#Calculating the spectrogram¶

Y_scale = np.abs(S_scale) ** 2
Y_scale.shape
print("Y_scale", Y_scale)

type(Y_scale[0][0])

def plot_spectrogram(Y, sr, hop_length, y_axis="linear", save_path=None):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(Y, 
                             sr=sr, 
                             hop_length=hop_length, 
                             x_axis="time", 
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)  # Save the figure
    plt.show()  # Display the figure as before

Y_log_scale = librosa.power_to_db(Y_scale)

plot_spectrogram(Y_scale, sr, HOP_SIZE, save_path="Y_scale_spectrogram.png")
plot_spectrogram(Y_log_scale, sr, HOP_SIZE, save_path="Y_log_scale_spectrogram.png")
plot_spectrogram(Y_log_scale, sr, HOP_SIZE, y_axis="log", save_path="Y_log_scale_log_spectrogram.png")

