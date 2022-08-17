"""
## Setup
"""
import sys
import os
import pickle
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras.backend import epsilon
from tensorflow.math import reduce_sum, square, maximum, sqrt

import matplotlib.pyplot as plt

from utilities.converters import path2image


if os.name == "nt":
    from config import WINDOWS_PATHS as PATHS
else:
    from config import CENTOS_PATHS as PATHS
os.chdir(sys.path[1])

"""
## Hyperparameters
"""
input_size = 224 #272 TOP
batch_size = 16
first_run = False
epochs = 5
margin = 1  # Margin for constrastive loss.

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# MODEL
input = layers.Input((input_size, input_size, 3))
x = tf.keras.layers.BatchNormalization()(input)
x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", strides=2)(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)


x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D(pool_size=(2, 2), padding="same", strides=2)(x)

x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D(pool_size=(2, 2), padding="same", strides=2)(x)


x = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(x)
x = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(x)
x = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(x)

x = layers.MaxPooling2D(pool_size=(2, 2), padding="same", strides=2)(x)
# x = layers.Dropout(0.1)(x)

x = layers.Flatten()(x)
x = layers.Dense(4096, activation="relu")(x)
x = layers.Dense(4096, activation="relu")(x)
x = layers.Dense(4096, activation="relu")(x)
embedding_network = keras.Model(input, x)

input_1 = layers.Input((input_size, input_size, 3))
input_2 = layers.Input((input_size, input_size, 3))

tower_1 = embedding_network(input_1)
tower_2 = embedding_network(input_2)

merge_layer = layers.Lambda(lambda x: sqrt(maximum(reduce_sum(square(x[0]-x[1]), axis=1, keepdims=True), epsilon())))([tower_1, tower_2])

normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
output_layer = layers.Dense(1, activation="sigmoid")(normal_layer)
siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)

"""
## Define the constrastive Loss
"""
def loss(margin=1):
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.
    Arguments:
        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar. - (default is 1).
    Returns:
        'constrastive_loss' function with data ('margin') attached.
    """
    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.
        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.
        Returns:
            A tensor containing constrastive loss as floating point value.
        """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive_loss


def contrastive_loss(y_true, y_pred):
    """Calculates the constrastive loss.
    Arguments:
        y_true: List of labels, each label is of type float32.
        y_pred: List of predictions of same length as of y_true,
                each label is of type float32.
    Returns:
        A tensor containing constrastive loss as floating point value.
    """
    margin=1
    square_pred = tf.math.square(y_pred)
    margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
    return tf.math.reduce_mean(
        (1 - y_true) * square_pred + (y_true) * margin_square
    )


"""
Compile the model with the contrastive loss
"""
siamese.compile(loss=contrastive_loss, optimizer="RMSprop", metrics=["accuracy"])
siamese.summary()
siamese.save("./siamese_tf")
siamese = tf.keras.models.load_model("./siamese_tf", custom_objects=({
            "contrastive_loss": contrastive_loss,
            # "euclidean_distance": euclidean_distance,
            # "euclidean_distance_output_shape": euclidean_distance_output_shape
        }))

checkpoint_filepath = './siamese_tf_checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='min',
    save_best_only=True)

"""
## Train the model
"""
with open(PATHS["PATH_DATASET_VAL"].joinpath("voiced_pairs.pickled"), "rb") as f:
    data = pickle.load(f)
    pairs_val_paths = data["data"]
    pairs_val = []
    for item in pairs_val_paths:
        pairs_val.append(path2image(item, (input_size, input_size)))
    labels_val = np.asarray(data["labels"], dtype=np.float32)
    pairs_val = np.asarray(pairs_val)

x_val_1 = pairs_val[:, 0]  # x_val_1.shape = (60000, 28, 28)
x_val_2 = pairs_val[:, 1]

for train_dataset_path in PATHS["PATH_DATASET_TRAIN"].glob("voiced_pairs_path*.pickled"):
    print(f"Training dataset: {train_dataset_path}")
    if first_run:
        siamese = tf.keras.models.load_model("./siamese_tf", custom_objects=({
            "contrastive_loss": contrastive_loss,
        }))

    with open(train_dataset_path, "rb") as f:
        data = pickle.load(f)
        pairs_train_paths = data["data"]
        labels_train = np.asarray(data["labels"], dtype=np.float32)
        pairs_train = []
        for item in pairs_train_paths:
            pairs_train.append(path2image(item, (input_size, input_size)))
    first_run = True
    pairs_train = np.asarray(pairs_train)
    x_train_1 = pairs_train[:, 0]  # x_train_1.shape is (60000, 28, 28)
    x_train_2 = pairs_train[:, 1]

    history = siamese.fit(
        [x_train_1, x_train_2], labels_train,
        validation_data=([x_val_1, x_val_2], labels_val),
        batch_size=batch_size,
        epochs=epochs, callbacks=[model_checkpoint_callback]
    )
    siamese.save("./siamese_tf")

"""
## Visualize results
"""


def plt_metric(history, metric, title, has_valid=True):
    """Plots the given 'metric' from 'history'.
    Arguments:
        history: history attribute of History object returned from Model.fit.
        metric: Metric to plot, a string value present as key in 'history'.
        title: A string to be used as title of plot.
        has_valid: Boolean, true if valid data was passed to Model.fit else false.
    Returns:
        None.
    """
    plt.plot(history[metric])
    if has_valid:
        plt.plot(history["val_" + metric])
        plt.legend(["train", "validation"], loc="upper left")
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.show()


# Plot the accuracy
plt_metric(history=history.history, metric="accuracy", title="Model accuracy")

# Plot the constrastive loss
plt_metric(history=history.history, metric="loss", title="Constrastive Loss")

"""
## Evaluate the model
"""

for test_dataset_path in Path("../../data/dataset/test").glob("voiced_pairs_path*.pickled"):
    print(f"Testing dataset: {test_dataset_path}")
    with open(test_dataset_path, "rb") as f:
        pairs_test_paths = np.asarray(pickle.load(f)["data"])
        pairs_test = []
        for item in pairs_test_paths:
            pairs_test.append(path2image(item, (input_size, input_size)))
        labels_test = np.asarray(pickle.load(f)["labels"], dtype=np.float32)
        pairs_test = np.asarray(pairs_test)
    #
    x_test_1 = pairs_test[:, 0]  # x_test_1.shape = (20000, 28, 28)
    x_test_2 = pairs_test[:, 1]
    results = siamese.evaluate([x_test_1, x_test_2], labels_test)

    print("test loss, test acc:", results)

"""
## Visualize the predictions
"""

# predictions = siamese.predict([x_test_1, x_test_2], labels_test)
# visualize(pairs_test, labels_test, to_show=3, predictions=predictions, test=True)
