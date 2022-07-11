"""
Module with various data preprocessing functions.
"""
from pathlib import Path
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np


def wav2spectrogram(source_path: Path, destination_path: Path, fft_window_length: int):
    """
    Converts sound file (sorce_path) to its spectrogram and save it to destination_path folder.
    Filename is the same as source sound file (but with .png extension).
    :param fft_window_length: lenght of FFT window (Hamming)
    :param source_path: path to sound file (pathlib)
    :param destination_path: path to folder where the spectrogram is saved. (pathlib)
    :return: None
    """
    sample_rate, samples = wavfile.read(source_path)
    frequencies, times, spectrogram = signal.spectrogram(samples, fs=sample_rate,
                                                         scaling="spectrum", nfft=None,
                                                         mode="psd", noverlap=128,
                                                         window=np.hamming(fft_window_length))
    plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram), cmap="viridis")
    plt.axis("off")
    plt.savefig(destination_path.joinpath(source_path.stem + ".png"), dpi=300, format="png",
                bbox_inches='tight', pad_inches=0)
    plt.close("all")


def stereo2mono(source_path: Path, destination_path: Path):
    """
    Converts stereo wav sound file to mono (single channel) wav file.
    :param source_path: path to stereo wav file
    :param destination_path: path to save converted mono wav file
    :return: None
    """
    sound = AudioSegment.from_wav(source_path)
    sound = sound.set_channels(1)
    sound.export(destination_path.joinpath(source_path.name), format="wav")


def txt2wav(source_path: Path, destination_path: Path):
    txt_data = np.loadtxt(source_path)
    print(txt_data.shape)
    wavfile.write(filename=destination_path.joinpath(source_path.stem + ".wav"), rate=8000, data=txt_data)
