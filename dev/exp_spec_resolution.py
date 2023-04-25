from pathlib import Path
import numpy as np
from scipy import signal
from scipy.io import wavfile
from matplotlib import pyplot as plt


def wav2spectrogram(source_path: Path, destination_path: Path, fft_window_length: int, resolution: tuple):
    """
    Converts sound file (sorce_path) to its spectrogram and save it to destination_path folder.
    Filename is the same as source sound file (but with .png extension).
    :param fft_window_length: lenght of FFT window (Hamming)
    :param source_path: path to sound file (pathlib)
    :param destination_path: path to folder where the spectrogram is saved. (pathlib)
    :return: None
    """
    sample_rate, samples = wavfile.read(source_path)
    # frequencies, times, spectrogram = signal.spectrogram(samples, fs=sample_rate,
    #                                                      scaling="spectrum", nfft=None,
    #                                                      mode="psd", noverlap=fft_window_length // 2,
    #                                                      window=np.hamming(fft_window_length))
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, window=np.hamming(fft_window_length),
                                                         noverlap=fft_window_length -1)
    inch_x = resolution[0] / 300
    inch_y = resolution[1] / 300
    fig = plt.figure(frameon=False)
    fig.set_size_inches(inch_y, inch_x)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.pcolormesh(times, frequencies, spectrogram, cmap="viridis")

    plt.savefig(destination_path.joinpath(source_path.stem + ".png"), format="png",
                bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close("all")


wav2spectrogram(Path("./voice001_nonhealthy_00005.wav"), Path("./"), 256,  (224, 224))
