import os
import sys
import csv
from random import shuffle
import tensorflow as tf


from scipy import signal
from scipy.io import wavfile
import numpy as np
from matplotlib import pyplot as plt

from src.cnn.models.cnn002 import create_model
from utilities.converters import txt2wav
from utilities.octave_filter_bank import octave_filtering


def transform_image(image, label):
  # normalizace a převod na float32
  return tf.cast(image, tf.float32) / 255., label

def data_pipeline(wav_chunks: int, octaves: list,
                  fft_len: int, fft_overlap: int, spectrogram_resolution: tuple):
    """
    Function providing the data pipelining.
    1. Split wav to chunks
    2. Apply octave filters
    3. Create spectrogram images and save them on a disk
    :param wav_chunks: number of wav chunks. First chunk is always droped, due to boundary effects.
    :param octaves: list of octave filters
    :param fft_len: lenght of fft window to generate spectrogram
    :param fft_overlap: overlaping of fft windows to generate spectrogram
    :param spectrogram_resolution: spectrogram image resolution
    :return: path to folder with training and validation set (spectrogram images)
    """
    inch_x = spectrogram_resolution[0] / 300  # 300 is value in plt.savefig..
    inch_y = spectrogram_resolution[1] / 300
    # pylint: disable=wrong-import-position
    if os.name == "nt":
        from config import WINDOWS_PATHS as PATHS
    else:
        from config import CENTOS_PATHS as PATHS
    os.chdir(sys.path[1])
    # pylint: enable=wrong-import-position

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
        DESTINATION_SPECTROGRAM_PATH.joinpath("training", "nonhealthy").mkdir(parents=True, exist_ok=True)
        DESTINATION_SPECTROGRAM_PATH.joinpath("training", "healthy").mkdir(parents=True, exist_ok=True)
        DESTINATION_SPECTROGRAM_PATH.joinpath("validation", "nonhealthy").mkdir(parents=True, exist_ok=True)
        DESTINATION_SPECTROGRAM_PATH.joinpath("validation", "healthy").mkdir(parents=True, exist_ok=True)
        print(f"created folder {DESTINATION_SPECTROGRAM_PATH}")
        # splitting
        complete_set = [item.stem[:8] for item in (PATHS["PATH_VOICED_RENAMED"].glob("*.txt"))]
        shuffle(complete_set)
        split_point = int(len(complete_set) * 0.2)
        training_set = complete_set[split_point:]
        validation_set = complete_set[:split_point]
        print(training_set)
        print(validation_set)
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
            if sound_file.stem[:8] in training_set:
                destination_path = DESTINATION_SPECTROGRAM_PATH.joinpath("training")
            else:
                destination_path = DESTINATION_SPECTROGRAM_PATH.joinpath("validation")
            if "nonhealthy" in str(sound_file):
                destination_path = destination_path.joinpath("nonhealthy")
            else:
                destination_path = destination_path.joinpath("healthy")
            plt.savefig(destination_path.joinpath(sound_file.stem + ".png"), format="png",
                        bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close("all")

    return DESTINATION_SPECTROGRAM_PATH

image_sizes = (224, 224)
path = data_pipeline(12, [3, 4, 5], 256, 128 // 2, image_sizes)

train = tf.keras.preprocessing.image_dataset_from_directory(
  path.joinpath("training"),

  image_size=image_sizes,
  batch_size=32)

val = tf.keras.preprocessing.image_dataset_from_directory(
  path.joinpath("validation"),
  image_size=image_sizes)

train = train.map(transform_image, num_parallel_calls=tf.data.AUTOTUNE)
val = val.map(transform_image, num_parallel_calls=tf.data.AUTOTUNE)

model = create_model(image_sizes[0])
# focal_loss = tf.keras.losses.BinaryCrossentropy()
focal_loss = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True)
optimizer_cnn = tf.keras.optimizers.Adam(learning_rate=0.000001)
model.compile(loss=focal_loss, optimizer=optimizer_cnn, metrics=["accuracy"])
model.summary()
# Display the model summary.
history = model.fit(train, validation_data=val, epochs=1000, batch_size=20).history
print("val acc max: {}".format(max(history["val_accuracy"][10:])))
