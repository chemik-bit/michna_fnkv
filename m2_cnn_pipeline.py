import socket
import importlib
import argparse
import os
import sys
from pathlib import Path
import shutil
import re
import itertools
import io
import csv
from random import shuffle
import yaml
import uuid
import json

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import sklearn.metrics

from utilities.converters import txt2wav, wav2spectrogram
from utilities.octave_filter_bank import octave_filtering

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#tf.config.threading.set_inter_op_parallelism_threads(0)
#tf.config.set_visible_devices([], 'GPU')

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

from sklearn.metrics import confusion_matrix

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

class Benchmark(tf.keras.metrics.Metric):
    """
    A custom metric that sums up the false positive and false negative results.
    """
    def __init__(self, name='benchmark', **kwargs):
        super(Benchmark, self).__init__(**kwargs)
        self.false_positives = self.add_weight(name='false_positives', initializer='zeros')
        self.false_negatives = self.add_weight(name='false_negatives', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.round(y_pred)  # Convert probabilities to binary predictions
        y_true = tf.cast(y_true, dtype=tf.bool)
        y_pred = tf.cast(y_pred, dtype=tf.bool)

        false_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_true), y_pred), dtype=tf.float32))
        false_negatives = tf.reduce_sum(tf.cast(tf.logical_and(y_true, tf.logical_not(y_pred)), dtype=tf.float32))

        self.false_positives.assign_add(false_positives)
        self.false_negatives.assign_add(false_negatives)

    def result(self):
        return self.false_positives + self.false_negatives

    def reset_state(self):
        self.false_positives.assign(0)
        self.false_negatives.assign(0)


def transform_image(image, label):
    """
    Function to perform image transformation.
    :param image: image to be transformed
    :param label: image class label
    :return: normalized image as tf.float32 and its label
    """
    image = tf.image.rgb_to_grayscale(image)
    return tf.cast(image, tf.float32) / 255., label

def transform_image2(image, label):
    """
    Function to perform image transformation.
    :param image: image to be transformed
    :param label: image class label
    :return: normalized image (-1,+1) as tf.float32 and its label
    """
    image = tf.image.rgb_to_grayscale(image)
    return (tf.cast(image, tf.float32) - 127.5) / 127.5, label

def balance_by_duplication(dataset_path):
    """
    Args:
        dataset_path: path to the dataset that needs to be augmented

    Returns:
        None
    """
    # List of healthy and nonhealthy samples
    healthy_set = list(dataset_path.joinpath("training").joinpath("healthy").iterdir())
    nonhealthy_set = list(dataset_path.joinpath("training").joinpath("nonhealthy").iterdir())
    healthy_count = len(healthy_set)
    nonhealthy_count = len(nonhealthy_set)

    # Choosing the part of the dataset that needs augmentation
    if nonhealthy_count > healthy_count:
        short_dataset = healthy_set
        long_dataset_count = nonhealthy_count
    else:
        short_dataset = nonhealthy_set
        long_dataset_count = healthy_count

    # Declaring lists of duplicated files with original and augmented names
    source_paths = []
    destination_paths = []

    # Duplicating files until the number of less-occurring files is balanced
    count = 0
    while len(destination_paths) < long_dataset_count:
        if len(destination_paths) + 2 * len(short_dataset) < long_dataset_count:
            source_paths += short_dataset
            destination_paths += [path.parent.joinpath(path.stem + "aug" + str(count) + ".png") for path in
                                  short_dataset]
            count += 1
        else:
            shuffle(short_dataset)
            source_paths += short_dataset[:(long_dataset_count - len(short_dataset) - len(destination_paths))]
            destination_paths += [path.parent.joinpath(path.stem + "aug" + str(count) + ".png") for path in
                                  short_dataset[:(long_dataset_count - len(short_dataset) - len(destination_paths))]]
            break

    # Copying the final list of files
    for source, destination in zip(source_paths, destination_paths):
        shutil.copy(source, destination)

def data_pipeline(wav_chunks: int, octaves: list, balanced: bool,
                  fft_len: int, fft_overlap: int, spectrogram_resolution: tuple, resampling_frequency: float,
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
    :param resampling_frequency: frequency to which the samples are to resample
    :param training_db: selection from databases for training dataset. Options 'voiced', 'svd', 'mixed'
    :param validation_db: selection from databases for validation dataset. Options 'voiced', 'svd', 'mixed'
    :return: path to folder with training and validation set (spectrogram images)
    """
    if os.name == "nt":
        from config import WINDOWS_PATHS as PATHS
    else:
        from config import CENTOS_PATHS as PATHS
    
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
                  f"overlap{fft_overlap}_resample{resampling_frequency}"

    # 1. Convert wavs to chunks
    # Check options
    if ("options" in locals()) and ("training" in options.keys()) and ("validation" in options.keys()):
        used_dbs = list({options["training"], options["validation"]})
        # Prepare wav files for mentioned databases in training and validation
    else:
        print("using voiced and svdadult databases")
        #used_dbs = ["voiced", "svdadult"] předěláno na níže uvedené
        used_dbs = ["svdadult"]
        # Prepare wav files for both databases

    for db in used_dbs:
        source_path = PATHS[f"PATH_{db.upper()}_RENAMED"]
        destination_path_wav = PATHS["PATH_WAV"].joinpath(db).joinpath(str(wav_chunks))
        print(f"Converting PATH_{db.upper()}_RENAMED to WAV...")
        for file in source_path.iterdir():
            print("file \n\n\n\n", file, "\n\n\n\n")
            sample_rate = int(file.stem.split("_")[-1])
            txt2wav(file, destination_path_wav, sample_rate, wav_chunks)

    #2. Convert chunks to spectrograms
        destination_path_spectrogram = PATHS["PATH_SPECTROGRAMS"].joinpath(db).joinpath(subdir_name)
        destination_path_spectrogram.mkdir(parents=True, exist_ok=True)
        print("Converting WAV files to spectrograms...")
        single_chunk = True if wav_chunks == 1 else False
        print(f" single chunk {single_chunk}")
        for sound_file in destination_path_wav.iterdir():
            # e.g.  /Users/honzamichna/Documents/GitHub/michna_fnkv/data/wav/svdadult/1/svdadult1081_healthy_50000_00000.wav
            print("sound_file in destination path wav\n\n\n\n", sound_file, "\n\n\n\n")
            #print("fft_len \n\n\n\n", fft_len, "\n\n\n\n")
            #print("fft_overlap \n\n\n\n", fft_overlap, "\n\n\n\n")
            if not destination_path_spectrogram.joinpath(f"{sound_file.stem}.png").exists():
                # Create spectrogram
                print("creating spectrogram, args: ", sound_file, destination_path_spectrogram, fft_len, fft_overlap,
                      spectrogram_resolution, octaves, single_chunk, resampling_frequency)
                wav2spectrogram(sound_file, destination_path_spectrogram, fft_len, fft_overlap,
                                spectrogram_resolution, octaves=octaves, standard_chunk=single_chunk,
                                resampling_freq=resampling_frequency)

    print("Dataset splitting...")
    # 3. Create training/validation datasets
    if ("options" in locals()) and ("training" in options.keys()) and ("validation" in options.keys()):
        destination_path_dataset = PATHS["PATH_DATASET"]\
            .joinpath(f"{subdir_name}_t_{options['training']}_v_{options['validation']}_balanced_{balanced}")
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
                              int(train_to_val_ratio / (train_to_val_ratio + 1) * len(unique_samples_all)):]
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

        if balanced:
            print("Balancing the training set...")
            balance_by_duplication(destination_path_dataset)

    elif not destination_path_dataset.exists(): # non-specified
        destination_path_dataset = PATHS["PATH_DATASET"]\
            .joinpath(f"{subdir_name}_t_mixed_v_mixed")
        source_files = list(PATHS["PATH_SPECTROGRAMS"].joinpath("svdadult").joinpath(subdir_name).iterdir()) + \
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

        if balanced:
            print("Balancing the training set...")
            balance_by_duplication(destination_path_dataset)

    else:
        print(f"{destination_path_dataset.name} configuration already existing, skipping dataset creation...")

    return destination_path_dataset


def pipeline(configfile: Path, generation: int, individual, ga: bool = False):

    if os.name == "nt":
        from config import WINDOWS_PATHS as PATHS
        # os.chdir(sys.path[1])
    else:
        from config import CENTOS_PATHS as PATHS

    benchmark_metric = Benchmark()

    with open(Path(__file__).parent.joinpath(configfile)) as file:
        config = yaml.safe_load(file)

    image_sizes = []
    try:
        losses = {"binary_crossentropy": tf.keras.losses.BinaryCrossentropy,
                  "focal_loss":  tf.keras.losses.BinaryFocalCrossentropy}
    except AttributeError:
        losses = {"binary_crossentropy": tf.keras.losses.BinaryCrossentropy,
                  "focal_loss": tf.keras.losses.BinaryCrossentropy}
    optimizers = {"adam": tf.keras.optimizers.legacy.Adam,
                  "sgd": tf.keras.optimizers.SGD,
                  "rmsprop": tf.keras.optimizers.RMSprop,
                  "adagrad": tf.keras.optimizers.Adagrad}
    transform = {"v1": transform_image,
                "v2": transform_image2}
    
    #FFT spectrogram sizes
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
    learning_rate_exp = float(config["lr"])
    models = config["models"]
    #print("models\n\n\n", models)
    binary = config["binary"] if "binary" in config else ""
    print("binary", binary)

    if config["loss"] == "focal_loss":
        loss_function = losses[config["loss"]](gamma=config["focal_loss_gamma"], alpha=0.25)
    else:
        loss_function = losses[config["loss"]]()

    if "resampling_frequency" in config.keys():
        resampling_frequency = config["resampling_frequency"]
    else:
        resampling_frequency = None
    # transform_function = transform[config["transform"]]()
    # TODO handle lr schedule and different params for optimizers

    for individual_index, eval_model in enumerate(models):
        try:
            classifier = importlib.import_module(eval_model)
            print("classifier", classifier)
            for fft_len, fft_overlap, balance, chunk, image_size in itertools.product(fft_lens, fft_overlaps, balances, chunks, image_sizes):
                print(f"Entering data_pipeline.... {image_size}")
                
                if fft_overlap >= fft_len:
                    print(f"window length of {fft_len} is smaller or equal to the overlap of {fft_overlap}. "
                          f"Skipping this setup.")
                else:
                    print("Entering data_pipeline....")
                    path = data_pipeline(chunk, [], balance, fft_len, fft_overlap, image_size,
                                         training=training_db, validation=validation_db,
                                         resampling_frequency=resampling_frequency)
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
                    train = train.map(transform[config["transform"]], num_parallel_calls=tf.data.AUTOTUNE)
                    val = val.map(transform[config["transform"]], num_parallel_calls=tf.data.AUTOTUNE)
                    print("Sets transformed...")
                    print(f"Model.. {classifier.__file__}")
                    model = classifier.create_model(image_size)
                    log_dir = "logs"
                    history_file = str(uuid.uuid4())

                    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f"logs/selected/{history_file}")
                    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                                               patience=20,
                                                                               verbose=1 )#,
                                                                               #mode="min")
                    metrics_list = ["accuracy",
                                    benchmark_metric,
                                    tf.keras.metrics.TruePositives(name="TP"),
                                    tf.keras.metrics.TrueNegatives(name="TN"),
                                    tf.keras.metrics.FalsePositives(name="FP"),
                                    tf.keras.metrics.FalseNegatives(name="FN"),
                                    tf.keras.metrics.Precision(name="Precision"),
                                    tf.keras.metrics.Recall(name="Recall"),
                                    tf.keras.metrics.AUC(name="AUC")]
                    #
                    initial_learning_rate = learning_rate_exp
                    final_learning_rate = learning_rate_exp / 1000
                    learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (1 / max_epochs)
                    steps_per_epoch = int(len(list(train)) / batch_size_exp)

                    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=initial_learning_rate,
                        decay_steps=steps_per_epoch,
                        decay_rate=learning_rate_decay_factor,
                        staircase=True)
                    #
                    lr_schedule = learning_rate_exp
                    optimizer_cnn = optimizers[config["optimizer"]](learning_rate=lr_schedule)

                    model.compile(loss=loss_function, optimizer=optimizer_cnn, metrics=metrics_list)
                    model.summary()

                    history = model.fit(train, validation_data=val,
                                        epochs=max_epochs,
                                        batch_size=batch_size_exp,
                                        callbacks=[tensorboard_callback, early_stopping_callback], 
                                        verbose = 1).history
                    
                    healthy_validation = len(list(path.joinpath("validation", "healthy").glob("*")))
                    nonhealthy_validation = len(list(path.joinpath("validation", "nonhealthy").glob("*")))
                    print(f"history keys {history.keys()}")


                    results_to_write = {"model": f"{classifier.__name__}",
                                        "benchmark_value": 9999999,
                                        "TP": 0,
                                        "TN": 0,
                                        "FP": 0,
                                        "FN": 0,
                                        "AUC": 0,
                                        "training_set": f"{path.joinpath('training')}",
                                        "val_set": f"{path.joinpath('validation')}",
                                        "loss": f"{loss_function._name_scope}",
                                        "optimizer":  optimizer_cnn.get_config()["name"],
                                        "lr": f"{learning_rate_exp}",
                                        "epochs": f"{max_epochs}",
                                        "batch_size": f"{batch_size_exp}",
                                        "balance":  f"{balance}",
                                        "fft_len":  f"{fft_len}",
                                        "fft_overlap": f"{fft_overlap}",
                                        "chunks": f"{chunk}",
                                        "image_size": f"{image_size}",
                                        "val_ratio": f"{nonhealthy_validation / (nonhealthy_validation + healthy_validation)}",
                                        "resampling": f"{resampling_frequency}"}
                    
                    for idx, fp in enumerate(history["val_FP"]):
                        if history["val_FN"][idx] + fp < results_to_write["benchmark_value"]:
                            results_to_write["benchmark_value"] =  history["val_FN"][idx] + fp
                            results_to_write["BENCHMARK_AUC"] = history["val_AUC"][idx]
                            results_to_write["TP"] = history["val_TP"][idx]
                            results_to_write["TN"] = history["val_TN"][idx]
                            results_to_write["FP"] = fp
                            results_to_write["FN"] = history["val_FN"][idx]

                    results_to_write["VAL_AUC_MAX"] = max(history["val_AUC"])
                    results_to_write["AUC_MAX"] = max(history["AUC"])
                    results_to_write["history_file"] = f"{history_file}.json"
                    results_to_write["configfile"] = configfile.name

                    #print("part 1")
                    with open(PATHS["PATH_RESULTS"].joinpath(socket.gethostname() + "_results_v3.csv"), "a", newline="") as csvfile:
                        fieldnames = [key for key in results_to_write.keys()]
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        if csvfile.tell() == 0:
                            writer.writeheader()
                        writer.writerow(results_to_write)

                    # Concatenate configfile.stem and "_original" to form a single directory name
                    directory_name = f"{configfile.stem}_original"
                    # Use joinpath to append this new directory name to PATH_RESULTS
                    PATHS["PATH_RESULTS"].joinpath(directory_name).mkdir(exist_ok=True)
                    #print("part 2")
                    with open(PATHS["PATH_RESULTS"].joinpath(directory_name, history_file + ".json"), "w") as fp:
                        json.dump(history, fp)

                    if ga:
                        #print("part 3")
                        GA_history = {}
                        GA_history["val_acc"] = max(history["val_accuracy"])
                        GA_history["binary"] = binary
                        print("GA_binary", binary)
                        #print("part 3")
                        PATHS["PATH_RESULTS"].joinpath("ga", str(generation+1)).mkdir(parents=True, exist_ok=True)
                        #print("part 4")
                        with open(PATHS["PATH_RESULTS"].joinpath("ga", str(generation+1), str(individual+1) + ".json"), "a") as fp:
                            json.dump(GA_history, fp)
                        #print("part 5")
                        print("generation", f"{generation+1}")
                        print("individual_index", f"{individual+1}")
        
        except Exception as e:
            print(f"Error training model {eval_model}: {e}")
            if ga:
                PATHS["PATH_RESULTS"].joinpath("ga", str(generation+1)).mkdir(parents=True, exist_ok=True)
                GA_history = {}
                GA_history["val_acc"] = 0
                GA_history["binary"] = binary
                with open(PATHS["PATH_RESULTS"].joinpath("ga", str(generation+1), str(individual+1) + ".json"), "a") as fp:
                    json.dump(GA_history, fp)

                continue


def main(configfile_path, generation, individual, ga=False):
    configfile = Path(configfile_path)
    pipeline(configfile, generation, individual, ga)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configfile", type=str, required=True)  # Notice type is changed to str
    args = parser.parse_args()
    main(args.configfile, 0, 0)

