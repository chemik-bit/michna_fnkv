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
# import cv2
#from timeit import default_timer as timer
try:
    from utilities.octave_filter_bank import octave_filtering
except ImportError:
    from octave_filter_bank import octave_filtering

# import librosa
# import librosa.display
# import IPython.display as ipd

def wav2spectrogram(source_path: Path, destination_path: Path, fft_window_length: int, fft_overlap: int,
                    spectrogram_resolution: tuple, dpi: int = 100, octaves: list = None, standard_chunk: bool = False,
                    resampling_freq: float = None, lower_bound: int = 0, upper_bound: int = 0):
    """
    Converts sound file (source_path) to its spectrogram and save it to destination_path folder.
    Filename is the same as source sound file (but with .png extension).
    :param fft_window_length: length of FFT window (Hamming)
    :param fft_overlap: number of points overlapping between neighboring window
    :param source_path: path to sound file (pathlib)
    :param destination_path: path to folder where the spectrogram is saved. (pathlib)
    :param spectrogram_resolution: resolution of the resulting image in pixels
    :param dpi: resolution density (dots per inch)
    :param octaves: used for octave filtering, ignored if not defined when calling the function
    :param standard_chunk: used 1 chunk wav partitioning, so the last second of wav file is used
    :param resampling_freq: used for resampling recordings to the same frequency
    :return: None
    """
    # Convert the dimensions from pixels to inches
    inch_x = spectrogram_resolution[0] / dpi
    inch_y = max((spectrogram_resolution[1] - upper_bound - lower_bound),240) / dpi

    # print("sample source for spectrogram\n\n\n\n", source_path, "\n\n\n\n")
    # print("inch_x\n\n\n\n", inch_x, "\n\n\n\n")
    # print("inch_y\n\n\n\n", inch_y, "\n\n\n\n")
    # Create spectrogram
    if octaves is None:
        octaves = []

    sample_rate, samples = wavfile.read(source_path)
    #right now it is one channel
    # print("sample \n\n\n\n", samples, "\n\n\n\n")
    
    # print("1")
    if resampling_freq is not None:
        number_of_samples = round(len(samples) * resampling_freq / sample_rate)
        samples = signal.resample(samples, number_of_samples)
        sample_rate = resampling_freq
    # print("2")
    if octaves is not None:
        samples = octave_filtering(octaves, samples)
    # print("3")
    if standard_chunk:
        # print("standard_chunk\n\n\n\n", standard_chunk, "\n\n\n\n")
        # print("sample_rate\n\n\n\n", sample_rate, "\n\n\n\n")
        # print("samples\n\n\n\n", samples, "\n\n\n\n")
        # print("len samples\n\n\n\n", len(samples), "\n\n\n\n")
        if len(samples) > sample_rate + 1:
            # print("success")
            middle_point = int(len(samples) / 2)
            samples = samples[- middle_point - int(sample_rate / 2): - middle_point + int(sample_rate / 2)]
            # print("new_samples", len(samples))
        else:
            # print("Not enough data to create standard chunk.")
            return
    # print("4")
    frequencies, times, spectrogram = signal.spectrogram(samples,
                                                         fs=sample_rate,
                                                         scaling="spectrum", nfft=None, mode="psd",
                                                         window=np.hamming(fft_window_length),
                                                         noverlap=fft_overlap)
    # print("frequencies\n\n\n\n", frequencies.size, "\n\n\n\n")
    # print("times\n\n\n\n", times.size, "\n\n\n\n")
    # print("spectrogram\n\n\n\n", spectrogram.size, "\n\n\n\n")
    
    fig = plt.figure(frameon=False, figsize=(inch_x, inch_y))
    #fig.set_size_inches(inch_x, inch_y)  # first arg sets the width, second the height
    plot_axes = plt.Axes(fig, [0., 0., 1., 1.])
    plot_axes.set_axis_off()

    # Add the custom axes to the figure
    fig.add_axes(plot_axes)
    
    # Generate the plot
    upper_bound = len(frequencies) - upper_bound
    # Slice the frequencies and spectrogram to include only the lower 500 frequency bins
    lower_frequencies = frequencies[lower_bound:upper_bound]
    lower_spectrogram = spectrogram[lower_bound:upper_bound, :]
    plt.pcolormesh(times, lower_frequencies, 10 * np.log10(lower_spectrogram), cmap="Greys")
    
    #standard
    #plot_axes.pcolormesh(times, frequencies, 10 * np.log10(spectrogram), cmap="Greys")
    # Save the figure
    plt.savefig(destination_path.joinpath(f"{source_path.stem}.png"), format="png", dpi=dpi, bbox_inches='tight', pad_inches=0)


    # Show the plot
    #plt.show()

    # Close the figure to free up memory
    plt.close(fig)  # It's better to close the figure explicitly by referencing it

#wav2spectrogram(Path("/Users/honzamichna/Documents/GitHub/michna_fnkv/data/wav/svdadult/1/svdadult0363_unhealthy_50000_00000.wav"), Path("/Users/honzamichna/Desktop"), \
#                1250, 625, (79, 626), dpi=200, octaves=[], standard_chunk=True, resampling_freq=None)

#wav2spectrogram(Path("/Users/honzamichna/Documents/GitHub/michna_fnkv/data/wav/svdadult/1/svdadult0363_unhealthy_50000_00000.wav"), Path("/Users/honzamichna/Desktop"), \
#                1250, 625, (79, 400), dpi=100, octaves=[], standard_chunk=True, resampling_freq=None)
"""
wav2spectrogram(sound_file, destination_path_spectrogram, 
                fft_len, fft_overlap,
                                spectrogram_resolution, octaves=octaves, standard_chunk=single_chunk,
                                resampling_freq=resampling_frequency)
                                """

def wav2spectrogram2(source_path: Path, destination_path: Path, fft_window_length: int, fft_overlap: int,
                    spectrogram_resolution: tuple, dpi: int = 300, octaves: list = None, standard_chunk: bool = False,
                    resampling_freq: float = None):
    plt.figure()
    scale, sr = librosa.load(source_path)
    S_scale = librosa.stft(scale, n_fft=fft_window_length, hop_length=fft_overlap)
    print("S_scale", S_scale.shape)
    print(type(S_scale[0][0]))
    Y_scale = np.abs(S_scale) ** 2
    print("Y_scale", Y_scale.shape)
    print(type(Y_scale[0][0]))
    plot_spectrogram(Y_scale, sr, HOP_SIZE)
    Y_log_scale = librosa.power_to_db(Y_scale)
    plot_spectrogram(Y_log_scale, sr, HOP_SIZE)

def plot_spectrogram(Y, sr, hop_length, y_axis="linear"):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(Y, 
                             sr=sr, 
                             hop_length=hop_length, 
                             x_axis="time", 
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")




def txt2wav(source_path: Path, destination_path: Path, sample_rate: int, chunks: int = 1):
    """
    Converts voiced db, where data files are text files, cointaining wav sample values.
    :param source_path: path to voiced database txt files
    :param destination_path: path to destination folder
    :param sample_rate: target wav sample rate
    :param chunks: number of chunks -> each txt is divided to multiple wav files
    :return: None
    """
    destination_path.mkdir(parents=True, exist_ok=True)
    #print("source_path\n\n\n\n", source_path, "\n\n\n\n")
    txt_data = np.loadtxt(source_path)
    #print("txt_data\n\n\n\n", txt_data, "\n\n\n\n")
    #print("len txt_data\n\n\n\n", len(txt_data), "\n\n\n\n")
    if chunks > 1:
        wav_chunks = np.array_split(txt_data, chunks)
        wav_chunks.pop(0)  # to remove bad data at start
    else:
        wav_chunks = np.array_split(txt_data, chunks)
        #print("chunk ==== single-1")
        #print(wav_chunks)
        #print("len wav_chunks\n\n\n\n", wav_chunks[0].size, "\n\n\n\n")
    
    
    for idx, wav_chunk in enumerate(wav_chunks):
        chunk_path = destination_path.joinpath(f"{source_path.stem}_{idx:05d}.wav")
        #otazka
        if not chunk_path.is_file():
            # print(f"creating {chunk_path}")
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


# def path2image(paths: list, size: tuple):
#     """
#     Load images given by paths and resize them to desired size
#     :param paths: list with pathlike objects to images
#     :param size: tuple with desired image size
#     :return: list with loaded and resized images
#     """
#     converted_images = []
#     for image_path in paths:
#         image = cv2.imread(image_path)
#         converted_images.append(cv2.resize(image, size, interpolation=cv2.INTER_AREA))
#     return converted_images
