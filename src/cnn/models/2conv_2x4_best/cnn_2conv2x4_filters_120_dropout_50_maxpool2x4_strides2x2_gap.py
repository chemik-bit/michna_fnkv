from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(120, (2, 4), activation="relu", padding="same", input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(2, 4), strides=(2, 2)))
    x.add(layers.Conv2D(120, (2, 4), activation="relu", padding="same", input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(2, 4), strides=(2, 2)))
    x.add(layers.Dropout(50))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1, activation="sigmoid"))
    x.compile(loss=tf.losses.binary_crossentropy, optimizer="adam", metrics=["val_accuracy"])
    return x
