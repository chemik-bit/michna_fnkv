"""
Module with various data preprocessing functions.
"""
from pathlib import Path
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
import shutil


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


def txt2wav(source_path: Path, destination_path: Path, sample_rate=8000, chunks=1):
    txt_data = np.loadtxt(source_path)
    wav_chunks = np.array_split(txt_data, chunks)
    if len(wav_chunks[:-1]) != len(wav_chunks[0]):
        wav_chunks.pop(0) # to remove bad data at start
        wav_chunks.pop(-1)

    for idx, wav_chunk in enumerate(wav_chunks):
        chunk_path = destination_path.joinpath(f"{source_path.stem}_ {idx:05d}.wav")
        wavfile.write(filename=chunk_path, rate=sample_rate, data=wav_chunk)


def rename_voiced(voiced_path: Path, destination_path: Path):
    """
    Add label (healty/nonhealty) to voiced txt files and save them to separate folder.
    :param voiced_path: path to original voiced database
    :param destination_path: path to destination folder with txt files
    :return: None
    """
    destination_path.mkdir(parents=True, exist_ok=True)
    for description_file in voiced_path.glob("*.hea"):
        with open(description_file, "r") as f:
            processed_filename = description_file.stem
            data = f.readlines()
            if "healthy" in data[-1]:
                destination_filename = processed_filename + "_healthy.txt"
                shutil.copy(voiced_path.joinpath(processed_filename + ".txt"),
                            destination_path.joinpath(processed_filename + "_healthy.txt"))
            else:
                destination_filename = processed_filename + "_nonhealthy.txt"
            shutil.copy(voiced_path.joinpath(processed_filename + ".txt"),
                        destination_path.joinpath(destination_filename))
