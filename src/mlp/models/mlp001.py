from tensorflow.keras import layers
from tensorflow.keras.backend import epsilon

import tensorflow as tf
from tensorflow import keras
# MODEL

def create_model(input_size):
    model = tf.keras.Sequential()

    # Build the model.
    model.add(layers.Flatten(input_shape=(input_size, input_size, 3)))
    model.add(layers.Dense(4096, activation='relu', input_shape=(input_size, input_size, 3)))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(1, activation="sigmoid"))

    # Display the model summary.
    model.summary()

    checkpoint_filepath = './siamese_tf_checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='min',
        save_best_only=True)

    return model
