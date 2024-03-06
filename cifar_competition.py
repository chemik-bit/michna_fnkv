#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 23:20:39 2023

@author: honzamichna
"""

#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from cifar10 import CIFAR10

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=..., type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=..., type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        # tf.data.experimental.enable_debug_mode()

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load data
    cifar = CIFAR10()

    # TODO: Create the model and train it
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C]))

    model.add(tf.keras.layers.Conv2D(16, kernel_size=3, strides=2,
                                         padding="same", activation="leaky_relu"))
    
    model.add(tf.keras.layers.BatchNormalization())
   
      
    model.add(tf.keras.layers.Conv2D(32, kernel_size=3, strides=2,
                                         padding="same", activation="leaky_relu"))
    
    model.add(tf.keras.layers.BatchNormalization())

    
    model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=2,
                                         padding="same", activation="leaky_relu"))
    
    model.add(tf.keras.layers.BatchNormalization())
    
    model.add(tf.keras.layers.Conv2D(128, kernel_size=3, strides=2,
                                         padding="same", activation="leaky_relu"))
    
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten())
    
    model.add(tf.keras.layers.Dense(500, activation="sigmoid"))
    
    model.add(tf.keras.layers.Dense(CIFAR10.LABELS, activation="softmax"))

    model.compile(
        optimizer=tf.optimizers.legacy.Adam(),
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    model.save(os.path.join(logdir, "model.h5"))

    tb_callback = tf.keras.callbacks.TensorBoard(logdir)
    checkpoint = tf.keras.callbacks.ModelCheckpoint("my_final_model.h5", \
                                                    save_best_only = True)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor = "val_accuracy", patience = 4)


    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Perform the prediction on the test data.
        for probs in model.predict(...):
            print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)