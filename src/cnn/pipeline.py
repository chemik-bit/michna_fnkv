# TODO implement YAML config file
"""
Complete datapipeline for CNN classification.
1. Convert wavs to chunks
2. Convert chunks to spectrograms
3. Create training/validation datasets
3. Import CNN model.
4. Load training/validation datasets as tf.dataset.Data
5. Train CNN model and validate
"""
import importlib
import os
import sys
import shutil
import re
import itertools
import io
import csv
from random import shuffle
import yaml
import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
# from src.cnn.models.cnn003 import create_model
#import src.cnn.models.cnn001 as classifier
from utilities.converters import txt2wav, wav2spectrogram
from utilities.octave_filter_bank import octave_filtering

import sklearn.metrics
import tensorflow_addons as tfa


from tensorflow.keras.callbacks import TensorBoard


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
    image = tf.image.rgb_to_grayscale(image)
    return tf.cast(image, tf.float32) / 255., label


def data_pipeline(wav_chunks: int, octaves: list, balanced: bool,
                  fft_len: int, fft_overlap: int, spectrogram_resolution: tuple,
                  **options):
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
    :param training_db: selection from databases for training dataset. Options 'voiced', 'svd', 'mixed'
    :param validation_db: selection from databases for validation dataset. Options 'voiced', 'svd', 'mixed'
    :return: path to folder with training and validation set (spectrogram images)
    """
    inch_x = spectrogram_resolution[0] / 300  # 300 is value in plt.savefig..
    inch_y = spectrogram_resolution[1] / 300

    if ("options" in locals()) and ("train_ratio" in options.keys()):
        train_to_val_ratio = options["train_ratio"] / (1 - options["train_ratio"])
    elif ("options" in locals()) and ("val_ratio" in options.keys()):
        train_to_val_ratio = (1 - options["val_ratio"]) / options["val_ratio"]
    else:
        train_to_val_ratio = 4

    subdir_name = f"ch{wav_chunks}_" \
                  f"res{spectrogram_resolution[0]}x{spectrogram_resolution[1]}_" \
                  f"octaves{''.join(map(str, octaves))}_" \
                  f"fft{fft_len}_" \
                  f"overlap{fft_overlap}"

    # 1. Convert wavs to chunks
    # Check options
    if ("options" in locals()) and ("training" in options.keys()) and ("validation" in options.keys()):
        used_dbs = list({options["training"], options["validation"]})
        # Prepare wav files for mentioned databases in training and validation
    else:
        print("converting both databases")
        used_dbs = ["voiced", "svd"]
        # Prepare wav files for both databases

    for db in used_dbs:
        source_path = PATHS[f"PATH_{db.upper()}_RENAMED"]
        destination_path_wav = PATHS["PATH_WAV"].joinpath(db).joinpath(str(wav_chunks))
        print(f"Converting PATH_{db.upper()}_RENAMED to WAV...")
        for file in source_path.iterdir():
            sample_rate = int(file.stem.split("_")[-1])
            txt2wav(file, destination_path_wav, sample_rate, wav_chunks)

    #2. Convert chunks to spectrograms
        destination_path_spectrogram = PATHS["PATH_SPECTROGRAMS"].joinpath(db).joinpath(subdir_name)
        destination_path_spectrogram.mkdir(parents=True, exist_ok=True)
        print("Converting WAV files to spectrograms...")
        for sound_file in destination_path_wav.iterdir():
            if not destination_path_spectrogram.joinpath(f"{sound_file.stem}.png").exists():
                # Create spectrogram
                wav2spectrogram(sound_file, destination_path_spectrogram, fft_len, fft_overlap,
                                spectrogram_resolution, octaves=octaves)

    print("Dataset splitting...")
    # 3. Create training/validation datasets
    if ("options" in locals()) and ("training" in options.keys()) and ("validation" in options.keys()):
        destination_path_dataset = PATHS["PATH_DATASET"]\
            .joinpath(f"{subdir_name}_t_{options['training']}_v_{options['validation']}")
        if options["training"] == options["validation"] and not destination_path_dataset.exists():

            source_files = list(PATHS["PATH_SPECTROGRAMS"].joinpath(options["training"]).joinpath(subdir_name).iterdir())

            # Obtaining unique samples (humans) in random order
            unique_samples_all = list({str(name.name).split("_")[0] for name in source_files})
            unique_samples_all.sort()
            shuffle(unique_samples_all)
            unique_samples = {
                "training": unique_samples_all[
                            :int(train_to_val_ratio / (train_to_val_ratio + 1) * len(unique_samples_all))],
                "validation": unique_samples_all[
                              int(train_to_val_ratio / (train_to_val_ratio + 1) * len(unique_samples_all)) - 1:]
            }

            # Splitting samples
            for key in unique_samples.keys():
                destination_path_dataset.joinpath(key).joinpath("healthy").mkdir(exist_ok=True, parents=True)
                destination_path_dataset.joinpath(key).joinpath("nonhealthy").mkdir(exist_ok=True, parents=True)
                # for fft_len, balance, chunk, image_size in itertools.product(fft_lens, balances, chunks, image_sizes):
                for sample, file in itertools.product(unique_samples[key], source_files):
                    if sample in str(file):
                        if "_healthy_" in str(file):
                            shutil.copy(file, destination_path_dataset.joinpath(key).joinpath("healthy")
                                        .joinpath(file.name))
                        else:
                            shutil.copy(file, destination_path_dataset.joinpath(key).joinpath("nonhealthy")
                                        .joinpath(file.name))

        elif not destination_path_dataset.exists():
            # All training and validation files
            source_files = {
                "training": list(PATHS["PATH_SPECTROGRAMS"].joinpath(options["training"])
                    .joinpath(subdir_name).iterdir()),
                "validation": list(PATHS["PATH_SPECTROGRAMS"].joinpath(options["validation"])
                    .joinpath(subdir_name).iterdir()),
            }
            # Obtaining unique samples (humans) in random order
            for key in source_files.keys():
                destination_path_dataset.joinpath(key).joinpath("healthy").mkdir(exist_ok=True, parents=True)
                destination_path_dataset.joinpath(key).joinpath("nonhealthy").mkdir(exist_ok=True, parents=True)
                for file in source_files[key]:
                    if "_healthy_" in str(file):
                        shutil.copy(file, destination_path_dataset.joinpath(key).joinpath("healthy")
                                    .joinpath(file.name))
                    else:
                        shutil.copy(file, destination_path_dataset.joinpath(key).joinpath("nonhealthy")
                                    .joinpath(file.name))
        else:
            print(f"{destination_path_dataset.name} configuration already existing, skipping dataset creation...")

    elif not destination_path_dataset.exists(): # non-specified
        destination_path_dataset = PATHS["PATH_DATASET"]\
            .joinpath(f"{subdir_name}_t_mixed_v_mixed")
        source_files = list(PATHS["PATH_SPECTROGRAMS"].joinpath("svd").joinpath(subdir_name).iterdir()) + \
                       list(PATHS["PATH_SPECTROGRAMS"].joinpath("voiced").joinpath(subdir_name).iterdir())

        # Obtaining unique samples (humans) in random order
        unique_samples_all = list({str(name.name).split("_")[0] for name in source_files})
        unique_samples_all.sort()
        shuffle(unique_samples_all)
        unique_samples = {
            "training": unique_samples_all[:int(train_to_val_ratio / (train_to_val_ratio + 1) * len(unique_samples_all))],
            "validation": unique_samples_all[int(train_to_val_ratio / (train_to_val_ratio + 1) * len(unique_samples_all)) - 1:]
        }

        # Splitting samples
        for key in unique_samples.keys():
            print(key)
            destination_path_dataset.joinpath(key).joinpath("healthy").mkdir(exist_ok=True, parents=True)
            destination_path_dataset.joinpath(key).joinpath("nonhealthy").mkdir(exist_ok=True, parents=True)
            for sample, file in itertools.product(unique_samples[key], source_files):
                if sample in str(file):
                    if "_healthy_" in str(file):
                        shutil.copy(file, destination_path_dataset.joinpath(key).joinpath("healthy")
                                    .joinpath(file.name))
                    else:
                        shutil.copy(file, destination_path_dataset.joinpath(key).joinpath("nonhealthy")
                                    .joinpath(file.name))
    else:
        print(f"{destination_path_dataset.name} configuration already existing, skipping dataset creation...")

    return destination_path_dataset



if os.name == "nt":
    from config import WINDOWS_PATHS as PATHS
else:
    from config import CENTOS_PATHS as PATHS
os.chdir(sys.path[1])
with open("./src/cnn/configs/h_config.yaml") as file:
    config = yaml.safe_load(file)

image_sizes = []
losses = {"binary_crossentropy": tf.keras.losses.BinaryCrossentropy,
          "focal_loss":  tf.keras.losses.BinaryFocalCrossentropy}
optimizers = {"adam": tf.keras.optimizers.Adam,
              "sgd": tf.keras.optimizers.SGD,
              "rmsprop": tf.keras.optimizers.RMSprop}
for key in config["image_size"]:
    image_sizes.append(tuple(config["image_size"][key]))

chunks = config["wav_chunks"]
balances = config["balances"]
fft_lens = config["fft_lens"]
fft_overlaps = config["fft_overlaps"]
training_db = config["training_db"]
validation_db = config["validation_db"]
batch_size_exp = config["batch_size_exp"]
max_epochs = config["max_epochs"]
learning_rate_exp = config["lr"]
models = config["models"]
loss_function = losses[config["loss"]]()
# TODO handle lr schedule and different params for optimizers
optimizer_cnn = optimizers[config["optimizer"]](learning_rate=learning_rate_exp)

for eval_model in models:
    classifier = importlib.import_module(eval_model)
    for fft_len, balance, chunk, image_size in itertools.product(fft_lens, balances, chunks, image_sizes):
        print(f"Entering data_pipeline.... {image_size}")
        path = data_pipeline(chunk, [], balance, fft_len, fft_len // 2, image_size,
                             training=training_db, validation=validation_db)
        # path = data_pipeline(chunk, [3, 4, 5, 6], balance, fft_len, fft_len // 2, image_size)
        print("Exited data_pipeline....")
        print("Loading datasets....")
        train = tf.keras.preprocessing.image_dataset_from_directory(
            path.joinpath("training"),

            image_size=image_size,
            batch_size=batch_size_exp)

        print("Train set loaded")
        val = tf.keras.preprocessing.image_dataset_from_directory(
            path.joinpath("validation"),
            image_size=image_size)
        print("Validation set loaded")
        train = train.map(transform_image, num_parallel_calls=tf.data.AUTOTUNE)
        val = val.map(transform_image, num_parallel_calls=tf.data.AUTOTUNE)
        print("Sets transformed...")
        print(f"Model.. {classifier.__file__}")
        model = classifier.create_model(image_size)
        #loss_function = tf.keras.losses.BinaryCrossentropy()
        #focal_loss = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=False)
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=1e-2,
        #     decay_steps=1000000,
        #     decay_rate=0.99)

        #optimizer_cnn = tf.keras.optimizers.SGD(lr=0.01)
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
                        tf.keras.metrics.TruePositives(name="TP"),
                        tf.keras.metrics.TrueNegatives(name="TN"),
                        tf.keras.metrics.FalsePositives(name="FP"),
                        tf.keras.metrics.FalseNegatives(name="FN"),
                        tf.keras.metrics.Precision(name="Precision"),
                        tf.keras.metrics.Recall(name="Recall"),
                        tf.keras.metrics.AUC(name="AUC")]


        model.compile(loss=loss_function, optimizer=optimizer_cnn, metrics=metrics_list)
        model.summary()
        # Display the model summary.

        history = model.fit(train, validation_data=val, epochs=max_epochs, batch_size=batch_size_exp, callbacks=[tensorboard_callback]).history
        healthy_validation = len(list(path.joinpath("validation", "healthy").glob("*")))
        nonhealthy_validation = len(list(path.joinpath("validation", "nonhealthy").glob("*")))
        print(f"history keys {history.keys()}")
        benchmark_value = 9999999999999
        benchmark_auc = 0
        for idx, fp in enumerate(history["val_FP"]):
            if history["val_FN"][idx] + fp < benchmark_value:
                benchmark_value = fp + history["val_FN"][idx]
                benchmark_auc = history["val_AUC"][idx]
                benchmark_tp = history["val_TP"][idx]
                benchmark_tn = history["val_TN"][idx]
                benchmark_fp = fp
                benchmark_fn = history["val_FN"][idx]
        with open("results.txt", "a") as result_file:
            result_file.write(f"val auc max: {max(history['val_AUC'])}, auc max: {max(history['AUC'])},"
                              f"benchmark_value: {benchmark_value} - AUC {benchmark_auc} - val TP {benchmark_tp} - val TN {benchmark_tn} - val FP {benchmark_fp} - val FN {benchmark_fn} "
                              f"{classifier.__name__},"
                              f"training set: {path.joinpath('training')},"
                              f"val set: {path.joinpath('validation')},"
                              f"{loss_function._name_scope},"
                              f"{optimizer_cnn._name},"
                              f"lr: {learning_rate_exp},"
                              f" balance: {balance},"
                              f" fft_len: {fft_len},"
                              f" chunks: {chunk},"
                              f" image_size: {image_size},"
                              f"val_ratio: {nonhealthy_validation / (nonhealthy_validation + healthy_validation)}\n")
        # print(history)
        # print(history.keys())
