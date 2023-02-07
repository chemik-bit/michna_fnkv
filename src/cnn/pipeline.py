"""
Complete datapipeline for CNN classification.
1. Convert wavs to chunks
2. Convert chunks to spectrograms
3. Create training/validation datasets
3. Import CNN model.
4. Load training/validation datasets as tf.dataset.Data
5. Train CNN model and validate
"""
import os
import sys
import csv
from random import shuffle
import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
import numpy as np
from matplotlib import pyplot as plt

from src.cnn.models.cnn001 import create_model
from utilities.converters import txt2wav
from utilities.octave_filter_bank import octave_filtering


def transform_image(image, label):
    """
    Function to perform image transformation.
    :param image: image to be transformed
    :param label: image class label
    :return: normalized image as tf.float32 and its label
    """
    return tf.cast(image, tf.float32) / 255., label


def data_pipeline(wav_chunks: int, octaves: list, balanced: bool,
                  fft_len: int, fft_overlap: int, spectrogram_resolution: tuple):
    """
    Function providing the data pipelining.
    1. Split wav to chunks
    2. Apply octave filters
    3. Create spectrogram images and save them on a disk
    :param balanced: True to create balanced dataset
    :param wav_chunks: number of wav chunks. First chunk is always droped, due to boundary effects.
    :param octaves: list of octave filters
    :param fft_len: lenght of fft window to generate spectrogram
    :param fft_overlap: overlaping of fft windows to generate spectrogram
    :param spectrogram_resolution: spectrogram image resolution
    :return: path to folder with training and validation set (spectrogram images)
    """
    inch_x = spectrogram_resolution[0] / 300  # 300 is value in plt.savefig..
    inch_y = spectrogram_resolution[1] / 300


    csv_path = PATHS["PATH_CSV"]
    source_path = PATHS["PATH_VOICED_RENAMED"]
    destination_wav_path = PATHS["PATH_VOICED_WAV"].joinpath(f"{wav_chunks}")

    # convert txt voiced files to wav chunks
    for file in source_path.glob("*.txt"):
        txt2wav(file, destination_wav_path, chunks=wav_chunks)
    # filter wav files and produce spectrograms
    subdir_name = "".join(map(str,octaves)) + f"_fft{fft_len}_overlap{fft_overlap}"
    prefix = "nonbalanced"
    if balanced:
        prefix = "balanced"
    destination_spectrogram_path = \
        PATHS["PATH_SPECTROGRAMS"].joinpath(
            f"{prefix}_ch{wav_chunks}_res{spectrogram_resolution[0]}x{spectrogram_resolution[1]}",
            subdir_name)
    # number of healthy
    set_size = min(len(list(destination_wav_path.glob("*_healthy*"))),
                   len(list(destination_wav_path. glob("*_nonhealthy*"))))
    print("healthy wavs: ", len(list(destination_wav_path.glob("*_healthy*"))))
    print("nonhealthy wavs: ", len(list(destination_wav_path.glob("*_nonhealthy*"))))
    print("set size: ", str(set_size))
    # number of unhealhty
    if not destination_spectrogram_path.is_dir():
        destination_spectrogram_path.joinpath(
            "training", "nonhealthy").mkdir(parents=True, exist_ok=True)
        destination_spectrogram_path.joinpath(
            "training", "healthy").mkdir(parents=True, exist_ok=True)
        destination_spectrogram_path.joinpath(
            "validation", "nonhealthy").mkdir(parents=True, exist_ok=True)
        destination_spectrogram_path.joinpath(
            "validation", "healthy").mkdir(parents=True, exist_ok=True)
        print(f"created folder {destination_spectrogram_path}")

        complete_set = [item.stem[:8] for item in (PATHS["PATH_VOICED_RENAMED"].glob("*.txt"))]
        shuffle(complete_set)
        split_point = int(len(complete_set) * 0.2)
        training_set = complete_set[split_point:]

        validation_set = complete_set[:split_point]
        with open(csv_path.joinpath("datasets_info.csv"),
                  "a", encoding="UTF8", newline="") as csv_file:
            writer = csv.writer(csv_file, quoting=csv.QUOTE_NONE, escapechar='')
            writer.writerow([wav_chunks, "+".join(map(str, octaves)),
                             fft_len, fft_overlap, "x".join(map(str, spectrogram_resolution))])
        randomized_path = list(destination_wav_path.glob("*.wav"))
        shuffle(randomized_path)
        nonhealty_counter = 0
        healthy_counter = 0
        for sound_file in randomized_path:
            print(f"processing {sound_file}")
            sample_rate, samples = wavfile.read(sound_file)
            samples = octave_filtering(octaves, samples)
            frequencies, times, spectrogram = signal.spectrogram(samples,
                                                                 sample_rate,
                                                                 window=np.hamming(fft_len),
                                                                 noverlap=fft_overlap)

            fig = plt.figure(frameon=False)
            fig.set_size_inches(inch_y, inch_x)
            plot_axes = plt.Axes(fig, [0., 0., 1., 1.])
            plot_axes.set_axis_off()
            fig.add_axes(plot_axes)
            plot_axes.pcolormesh(times, frequencies, spectrogram, cmap="hsv")

            if sound_file.stem[:8] in training_set:
                destination_path = destination_spectrogram_path.joinpath("training")
            else:
                destination_path = destination_spectrogram_path.joinpath("validation")
            if "nonhealthy" in str(sound_file):
                destination_path = destination_path.joinpath("nonhealthy")
                nonhealty_counter += 1
            else:
                destination_path = destination_path.joinpath("healthy")
                healthy_counter += 1
            save = False
            if "nonhealthy" in str(sound_file):
                if nonhealty_counter <= set_size:
                    save = True
            else:
                if healthy_counter <= set_size:
                    save = True
            if save:
                plt.savefig(destination_path.joinpath(sound_file.stem + ".png"), format="png",
                            bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close("all")

    return destination_spectrogram_path


if os.name == "nt":
    from config import WINDOWS_PATHS as PATHS
else:
    from config import CENTOS_PATHS as PATHS
os.chdir(sys.path[1])
image_sizes = [(40, 40), (50, 50), (60, 60), (80, 80), (100, 100)]
chunks = [x for x in range(2, 15)]
balances = [True, False]
fft_lens = [256, 128, 64]
for fft_len in fft_lens:
    for balance in balances:
        for chunk in chunks:
            for image_size in image_sizes:

                print(f"Entering data_pipeline.... {image_size}")
                path = data_pipeline(chunk, [3, 4], balance, fft_len, fft_len // 2, image_size)
                print("Exited data_pipeline....")

                train = tf.keras.preprocessing.image_dataset_from_directory(
                  path.joinpath("training"),

                  image_size=image_size,
                  batch_size=32)

                print("Train set loaded")
                val = tf.keras.preprocessing.image_dataset_from_directory(
                  path.joinpath("validation"),
                  image_size=image_size)
                print("Validation set loaded")
                train = train.map(transform_image, num_parallel_calls=tf.data.AUTOTUNE)
                val = val.map(transform_image, num_parallel_calls=tf.data.AUTOTUNE)
                print("Sets transformed...")
                model = create_model(image_size[0])
                focal_loss = tf.keras.losses.BinaryCrossentropy()
                # focal_loss = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=False)
                lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=1e-2,
                    decay_steps=1000000,
                    decay_rate=0.99)
                optimizer_cnn = tf.keras.optimizers.Adam(learning_rate=0.00001)

                model.compile(loss=focal_loss, optimizer=optimizer_cnn, metrics=["accuracy"])
                model.summary()
                # Display the model summary.
                history = model.fit(train, validation_data=val, epochs=1000, batch_size=50).history
                with open("result.txt", "a") as result_file:
                    result_file.write(f"cnn01, val acc max: {max(history['val_accuracy'][10:])},"
                                      f" balance: {balance},"
                                      f" fft_len: {fft_len},"
                                      f" chunks: {chunk},"
                                      f" image_size: {image_size} \n")
