import pickle
import os
import sys
import uuid

import numpy as np
from src.conversion import convert_voiced
from src.voiced_to_lists import voiced_to_lists
from src.unique_pairs_paths import unique_pairs
from src.siamese.models.vgg16_simplifed import create_model
from src.siamese.losses import contrastive_loss
import tensorflow as tf
from utilities.converters import path2image

if os.name == "nt":
    from config import WINDOWS_PATHS as PATHS
else:
    from config import CENTOS_PATHS as PATHS
os.chdir(sys.path[1])

# disable CUDA!!!
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
"""
Hyperparameters section
"""
MODEL_NAME = "vgg16_simplified"
EXPERIMENT_UUID = str(uuid.uuid4())
CHUNKS = 9
VALIDATION_SAMPLE_SIZE = 140
TEST_SAMPLE_SIZE = 0
TRAINING_SUBSET_SIZE = 1000
INPUT_SIZE = 224
BATCH_SIZE = 60
EPOCHS = 5
PATH_TO_SAVE = PATHS["PATH_EXPERIMENTS"].joinpath(EXPERIMENT_UUID)
PATH_TO_SAVE_MODEL = PATH_TO_SAVE.joinpath("model")
PATH_TO_SAVE.mkdir(parents=True, exist_ok=True)
PATH_TO_SAVE_MODEL.mkdir(parents=True, exist_ok=True)
first_run = False

experiment_info = {
    "uuid": EXPERIMENT_UUID,
    "chunks": CHUNKS,
    "val_sample_size": VALIDATION_SAMPLE_SIZE,
    "test_sample_size": TEST_SAMPLE_SIZE,
    "training_subset_size": TRAINING_SUBSET_SIZE,
    "input_size": INPUT_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "model_name": MODEL_NAME
}
if VALIDATION_SAMPLE_SIZE % (CHUNKS - 2) != 0:
    raise Exception

with open(PATH_TO_SAVE.joinpath("experiment_info"), "wb") as f:
    pickle.dump(experiment_info, f)

# 1. rename voiced, convert it to wav and then to spectrograms
convert_voiced(wav_chunks=CHUNKS) # 5 produce 3 spectrograms.. outer spectrograms are not used (
# boundary effects)

# 2. split spectrograms to training/validation sets.
# !!! zkontroluj že validation_sample_size a test_sample_size jsou v násobkách wav_chunks-2 !!!!!!!!!!!
voiced_to_lists(validation_sample_size=VALIDATION_SAMPLE_SIZE, test_sample_size=TEST_SAMPLE_SIZE)
if VALIDATION_SAMPLE_SIZE % (CHUNKS - 2) != 0:
    raise Exception

# 3. create spectrogram pairs for siamese network
unique_pairs(pairs_in_file=TRAINING_SUBSET_SIZE)

# 4. create and save model
siamese = create_model(INPUT_SIZE)
siamese.compile(loss=contrastive_loss, optimizer="RMSprop", metrics=["accuracy"])
siamese.summary()
siamese.save(PATH_TO_SAVE_MODEL)
siamese = tf.keras.models.load_model(PATH_TO_SAVE_MODEL, custom_objects=({
            "contrastive_loss": contrastive_loss,
        }))

checkpoint_filepath = PATH_TO_SAVE_MODEL.joinpath("checkpoints")
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1)

# 5. load validation set
print("Loading validation set....")
with open(PATHS["PATH_DATASET_VAL"].joinpath("voiced_pairs_00001.pickled"), "rb") as f:
    data = pickle.load(f)
    pairs_val_paths = data["data"]
    pairs_val = []
    for item in pairs_val_paths:
        pairs_val.append(path2image(item, (INPUT_SIZE,INPUT_SIZE)))
    labels_val = np.asarray(data["labels"], dtype=np.float32)
    pairs_val = np.asarray(pairs_val)

x_val_1 = pairs_val[:, 0]  # x_val_1.shape = (60000, 28, 28)
x_val_2 = pairs_val[:, 1]

# 6. train model
for iteration, train_dataset_path in enumerate(PATHS["PATH_DATASET_TRAIN"].glob("voiced_pairs_path*.pickled")):

    if first_run:
        siamese = tf.keras.models.load_model(PATH_TO_SAVE.joinpath("model"), custom_objects=({
            "contrastive_loss": contrastive_loss,
        }))
    print(f"Iteration {iteration}.... loading training dataset {train_dataset_path}")
    with open(train_dataset_path, "rb") as f:
        data = pickle.load(f)
        pairs_train_paths = data["data"]
        labels_train = np.asarray(data["labels"], dtype=np.float32)
        pairs_train = []
        for item in pairs_train_paths:
            pairs_train.append(path2image(item, (INPUT_SIZE, INPUT_SIZE)))
    first_run = True
    pairs_train = np.asarray(pairs_train)
    x_train_1 = pairs_train[:, 0]  # x_train_1.shape is (60000, 28, 28)
    x_train_2 = pairs_train[:, 1]

    history = siamese.fit(
        [x_train_1, x_train_2], labels_train,
        validation_data=([x_val_1, x_val_2], labels_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS, callbacks=[model_checkpoint_callback]
    )
    siamese.save(PATH_TO_SAVE_MODEL)
    with open(PATH_TO_SAVE.joinpath(f"history_{iteration:05d}.pickled"), "wb") as f:
        pickle.dump(history, f)
