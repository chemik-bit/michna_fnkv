"""
Conversion sound file to spectrogram
"""
from pathlib import Path
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np


def to_spectrogram(source_path, destination_path):
    """
    Converts sound file (sorce_path) to its spectrogram and save it to destination_path folder.
    Filename is the same as source sound file (but with .png extension).
    :param source_path: path to sound file
    :param destination_path: path to folder where the spectrogram is saved.
    :return: None
    """
    sample_rate, samples = wavfile.read(source_path)

    frequencies, times, spectrogram = signal.spectrogram(samples, fs=sample_rate,
                                                         scaling="spectrum", nfft=None,
                                                         mode="psd", noverlap=128,
                                                         window=np.hamming(256))
    plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram), cmap="viridis")
    plt.axis("off")
    plt.savefig(destination_path.joinpath(source_path.stem + ".png"), dpi=300, format="png",
                bbox_inches='tight', pad_inches=0)
    plt.close("all")


if __name__ == "__main__":
    SOURCE_PATH = Path("../data/mono")
    DESTINATION_PATH = Path("../data/spectrograms")
    for sound_file in SOURCE_PATH.iterdir():
        start = timer()
        to_spectrogram(sound_file, DESTINATION_PATH)
        end = timer()
        print(f"{sound_file.name} conversion: {end-start:.2f} s")
