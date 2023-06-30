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
from sklearn.metrics import confusion_matrix
from src.cnn.models.obsolete.h_cnn001_gap import create_model
from utilities.converters import txt2wav
from utilities.octave_filter_bank import octave_filtering
import itertools
import io
import sklearn.metrics


def log_confusion_matrix(epoch, logs):
    # Use the model to predict the values from the test_images.

    test_pred_raw = model.predict(val) # val should be images

    test_pred = np.argmax(test_pred_raw, axis=1)



    # Calculate the confusion matrix using sklearn.metrics
    y = np.concatenate([y for x, y in val], axis=0)

    cm = sklearn.metrics.confusion_matrix(y, test_pred) #val should be labels
    figure = plot_confusion_matrix(cm, class_names=["healthy", "nonhealthy"])

    cm_image = plot_to_image(figure)

    # Log the confusion matrix as an image summary.
    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)

def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """

    buf = io.BytesIO()

    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format='png')

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Use tf.image.decode_png to convert the PNG buffer
    # to a TF image. Make sure you use 4 channels.
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # Use tf.expand_dims to add the batch dimension
    image = tf.expand_dims(image, 0)

    return image

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

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
        txt2wav(file, destination_wav_path, chunks=wav_chunks, sample_rate=8000)
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
    if balanced:
        set_size = min(len(list(destination_wav_path.glob("*_healthy*"))),
                       len(list(destination_wav_path.glob("*_nonhealthy*"))))
    else:
        set_size = len(list(destination_wav_path.glob("*.*")))
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
image_sizes = [(80, 80)]
chunks = [x for x in range(10, 11)]
balances = [False]
fft_lens = [256]
for fft_len in fft_lens:
    for balance in balances:
        for chunk in chunks:
            for image_size in image_sizes:

                print(f"Entering data_pipeline.... {image_size}")
                path = data_pipeline(chunk, [3, 4, 5, 6], balance, fft_len, fft_len // 2, image_size)
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
                #focal_loss = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=False)
                lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=1e-2,
                    decay_steps=1000000,
                    decay_rate=0.99)
                optimizer_cnn = tf.keras.optimizers.Adam(learning_rate=0.00001)
                log_dir = "logs"
                # tensorboard stuff

                tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
                # callbacks = [TensorBoard(log_dir=log_dir,
                #                          histogram_freq=1,
                #                          write_graph=True,
                #                          write_images=False,
                #                          update_freq='epoch',
                #                          profile_batch=2,
                #                          embeddings_freq=1)]
                file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')
                cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
                metrics_list = ["accuracy",
                                tf.keras.metrics.TruePositives(),
                                tf.keras.metrics.TrueNegatives(),
                                tf.keras.metrics.FalsePositives(),
                                tf.keras.metrics.FalseNegatives(),
                                tf.keras.metrics.Precision(),
                                tf.keras.metrics.Recall(),
                                tf.keras.metrics.AUC()]
                model.compile(loss=focal_loss, optimizer=optimizer_cnn, metrics=metrics_list)
                model.summary()
                # Display the model summary.
                history = model.fit(train, validation_data=val, epochs=1000, batch_size=8, callbacks=[tensorboard_callback]).history
                healthy_validation = len(list(path.joinpath("validation", "healthy").glob("*")))
                nonhealthy_validation = len(list(path.joinpath("validation", "nonhealthy").glob("*")))

                with open("result_focal_exps.txt", "a") as result_file:
                    result_file.write(f"cnn01_005, val acc max: {max(history['val_accuracy'])}, acc max: {max(history['accuracy'])}"
                                      f" balance: {balance},"
                                      f" fft_len: {fft_len},"
                                      f" chunks: {chunk},"
                                      f" image_size: {image_size},"
                                      f"val_ratio: {nonhealthy_validation / (nonhealthy_validation + healthy_validation)}\n")
                print(history.keys())