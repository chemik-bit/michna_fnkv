import pickle
import os
import sys
import uuid
import inspect

import numpy as np
from src.conversion import convert_voiced
from src.voiced_to_lists import voiced_to_list
from src.cnn.models.h_cnn001_gap import create_model
import tensorflow as tf
from utilities.converters import path2image

if os.name == "nt":
    from config import WINDOWS_PATHS as PATHS
else:
    from config import CENTOS_PATHS as PATHS
os.chdir(sys.path[1])

# disable CUDA -1 if needed!!!
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
"""
Hyperparameters section
"""
MODEL_NAME = inspect.getmodule(create_model).__name__
EXPERIMENT_UUID = str(uuid.uuid4())
CHUNKS = 12
# TODO number of patients, instead of histograms!
VALIDATION_SAMPLE_SIZE = 40 * (CHUNKS) # patients number
TEST_SAMPLE_SIZE = 0

INPUT_SIZE = 224
BATCH_SIZE = 100
EPOCHS = 100
PATH_TO_SAVE = PATHS["PATH_EXPERIMENTS"].joinpath(EXPERIMENT_UUID)
PATH_TO_SAVE_MODEL = PATH_TO_SAVE.joinpath("model")
PATH_TO_SAVE.mkdir(parents=True, exist_ok=True)
PATH_TO_SAVE_MODEL.mkdir(parents=True, exist_ok=True)

# set True to prepare spectrogram images
PREPROCESSING = False

# this is helper variable to save model after each epoch
first_run = False
# TODO implement tensorflow data pipline to avoid all the pain with training

experiment_info = {
    "uuid": EXPERIMENT_UUID,
    "chunks": CHUNKS,
    "val_sample_size": VALIDATION_SAMPLE_SIZE,
    "test_sample_size": TEST_SAMPLE_SIZE,
    "input_size": INPUT_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "model_name": MODEL_NAME
}

if VALIDATION_SAMPLE_SIZE % CHUNKS != 0:
    raise Exception

with open(PATH_TO_SAVE.joinpath("experiment_info"), "wb") as f:
    pickle.dump(experiment_info, f)


if PREPROCESSING:
    # 1. rename voiced, convert it to wav and then to spectrograms
    convert_voiced(wav_chunks=CHUNKS) # 5 produce 3 spectrograms, outer spectrograms are not used (
    # boundary effects)

    # 2. split spectrograms to training/validation sets.
    # !!! zkontroluj že validation_sample_size a test_sample_size jsou v násobkách wav_chunks-2 !!!!!!!!!!!
histogram_paths, histogram_targets = voiced_to_list()
    # if VALIDATION_SAMPLE_SIZE % (CHUNKS - 2) != 0:
    #     raise Exception


# 4. create and save model
mlp = create_model(INPUT_SIZE)
focal_loss = tf.keras.losses.BinaryCrossentropy()
optimizer_cnn = tf.keras.optimizers.Adam(learning_rate=0.00001)
mlp.compile(loss=focal_loss, optimizer=optimizer_cnn, metrics=["accuracy"])
mlp.summary()
mlp.save(PATH_TO_SAVE_MODEL)
mlp = tf.keras.models.load_model(PATH_TO_SAVE_MODEL)

checkpoint_filepath = PATH_TO_SAVE_MODEL.joinpath("checkpoints")
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1,initial_value_threshold=0.7)

# 5. load validation set
print("Loading validation set....")
validation_set = []
training_set = []
validation_labels = np.asarray(histogram_targets[-VALIDATION_SAMPLE_SIZE:], dtype=np.float32)
training_labels = np.asarray(histogram_targets[:-VALIDATION_SAMPLE_SIZE], dtype=np.float32)
for path in histogram_paths[-VALIDATION_SAMPLE_SIZE:]:
    validation_set.append(path2image([path], (INPUT_SIZE, INPUT_SIZE))[0])
for path in histogram_paths[:-VALIDATION_SAMPLE_SIZE]:
    training_set.append(path2image([path], (INPUT_SIZE, INPUT_SIZE))[0])
print(len(training_labels), len(training_set))
print(len(validation_labels), len(validation_set))
training_set = np.asarray(training_set)
validation_set = np.asarray(validation_set)
# 6. train model
for iteration in range(10):

    if first_run:
        siamese = tf.keras.models.load_model(PATH_TO_SAVE.joinpath("model"))
    print(f"Iteration {iteration}...")

    history = mlp.fit([training_set], training_labels,
        validation_data=([validation_set], validation_labels),
                      epochs=EPOCHS,
                      callbacks=[model_checkpoint_callback])
    mlp.save(PATH_TO_SAVE_MODEL)
    with open(PATH_TO_SAVE.joinpath(f"history_{iteration:05d}.pickled"), "wb") as f:
        pickle.dump(history, f)
