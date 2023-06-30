"""
3x Convolutional layers with 100 filters of size (2, 7), stride (1, 1) and no padding.
The last layer has stride (2, 2).
3x MaxPooling layers after Conv layers with pool size (2, 2) and stride (2, 2)
Dropout 50%.
PRESUMABLY!!! AveragePooling with pool size (2, 2) and stride (1, 1) after the last layer.
PRESUMABLY!!! Linear layer with no activation function instead of Dense.
"""

from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()

    x.add(layers.Conv2D(100, (2, 7), activation="relu", padding="valid", input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
    x.add(layers.Conv2D(100, (2, 7), activation="relu", padding="valid"))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
    x.add(layers.Conv2D(100, (2, 7), strides=(2, 2), activation="relu", padding="valid"))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
    x.add(layers.Dropout(0.5))
    x.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1)))
    x.add(layers.Flatten())
    x.add(layers.Dense(1, activation="linear"))
    x.compile(loss=tf.losses.binary_crossentropy, optimizer="adam", metrics=["val_accuracy"])
    return x

