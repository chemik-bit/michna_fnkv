import os
import sys
import csv

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.utils import image_dataset_from_directory
from scipy.io import wavfile
from scipy import signal
import numpy as np
from matplotlib import pyplot as plt

from src.conversion import txt2wav
from dev.octave_filters import octave_filtering

"""
parametry:
chunks - int
octaves - list
fft okno - int
fft překryv - int
rozlišení spektrogramu - tuple
výstup:
složka s připravenejma spektrogramama
záznam do DB
"""


def data_pipeline(wav_chunks: int, octaves: list, fft_len: int, fft_overlap: int, spectrogram_resolution: tuple):

    inch_x = spectrogram_resolution[0] / 300
    inch_y = spectrogram_resolution[1] / 300

    if os.name == "nt":
        from config import WINDOWS_PATHS as PATHS
    else:
        from config import CENTOS_PATHS as PATHS
    os.chdir(sys.path[1])

    CSV_PATH = PATHS["PATH_CSV"]
    SOURCE_PATH = PATHS["PATH_VOICED_RENAMED"]
    DESTINATION_WAV_PATH = PATHS["PATH_VOICED_WAV"].joinpath(f"{wav_chunks}")

    # convert txt voiced files to wav chunks
    for file in SOURCE_PATH.glob("*.txt"):
        txt2wav(file, DESTINATION_WAV_PATH, chunks=wav_chunks)
    # filter wav files and produce spectrograms
    subdir_name = "".join(map(str,octaves)) + f"_fft{fft_len}_overlap{fft_overlap}"
    DESTINATION_SPECTROGRAM_PATH = \
        PATHS["PATH_SPECTROGRAMS"].joinpath(
            f"ch{wav_chunks}_res{spectrogram_resolution[0]}x{spectrogram_resolution[1]}", subdir_name)

    if not DESTINATION_SPECTROGRAM_PATH.is_dir():
        DESTINATION_SPECTROGRAM_PATH.joinpath("nonhealthy").mkdir(parents=True, exist_ok=True)
        DESTINATION_SPECTROGRAM_PATH.joinpath("healthy").mkdir(parents=True, exist_ok=True)
        print(f"created folder {DESTINATION_SPECTROGRAM_PATH}")
        with open(CSV_PATH.joinpath("datasets_info.csv"), "a", encoding="UTF8", newline="") as csv_file:
            writer = csv.writer(csv_file, quoting=csv.QUOTE_NONE, escapechar='')
            writer.writerow([wav_chunks, "+".join(map(str, octaves)), fft_len, fft_overlap, "x".join(map(str, spectrogram_resolution))])
        for sound_file in DESTINATION_WAV_PATH.glob("*.wav"):
            print(f"processing {sound_file}")
            sample_rate, samples = wavfile.read(sound_file)
            samples = octave_filtering(octaves, samples)
            frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, window=np.hamming(fft_len),
                                                                 noverlap=fft_overlap)

            fig = plt.figure(frameon=False)
            fig.set_size_inches(inch_y, inch_x)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.pcolormesh(times, frequencies, spectrogram, cmap="gray")
            destination_path = DESTINATION_SPECTROGRAM_PATH.joinpath("healthy")
            if "nonhealthy" in str(sound_file):
                destination_path = DESTINATION_SPECTROGRAM_PATH.joinpath("nonhealthy")

            plt.savefig(destination_path.joinpath(sound_file.stem + ".png"), format="png",
                        bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close("all")

    return DESTINATION_SPECTROGRAM_PATH


path = data_pipeline(4, [4,6], 256, 128 // 2, (224, 224))

train = tf.keras.preprocessing.image_dataset_from_directory(
  path,
  seed=123,
  validation_split=0.2,
  subset="training",
  image_size=(224, 224),
  batch_size=128)

val = tf.keras.preprocessing.image_dataset_from_directory(
  path,
  seed=123,
  validation_split=0.2,
  subset="validation",
  image_size=(224, 224),
  batch_size=128)
model = tf.keras.Sequential()

# Build the model.
model.add(layers.Flatten(input_shape=(224, 224,3)))
model.add(layers.Dense(4096, activation='relu', input_shape=(224, 224,3)))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(1, activation="sigmoid"))

# Display the model summary.
model.summary()
focal_loss = tf.keras.losses.BinaryCrossentropy()
optimizer_cnn = tf.keras.optimizers.Adam(learning_rate=0.00001)
model.compile(loss=focal_loss, optimizer=optimizer_cnn, metrics=["accuracy"])
model.fit(train, validation_data=val, epochs=30)