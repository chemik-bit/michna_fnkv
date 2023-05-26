from tensorflow.keras import layers
from tensorflow.keras.backend import epsilon
from tensorflow.math import reduce_sum, square, maximum, sqrt
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf
from tensorflow import keras

def create_model(input_size):
    x = tf.keras.Sequential()
    #x.add(layers.Input((input_size, input_size, 1)))
    #    x = tf.keras.layers.BatchNormalization()(input)
    x.add(layers.Conv2D(100, (3, 3), activation="relu", padding="same", input_shape = (input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1)))
    x.add(layers.Conv2D(100, (3, 3), activation="relu", padding="same"))
    x.add(layers.BatchNormalization())
    x.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    x.add(layers.Conv2D(100, (3, 3), activation="relu", padding="same"))
    x.add(layers.BatchNormalization())
    x.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1)))
    x.add(layers.Dropout(0.5))
    x.add(layers.Flatten())
    x.add(layers.Dense(1, activation="sigmoid"))
    x.compile(loss=tf.losses.binary_crossentropy, optimizer="adam", metrics=["val_accuracy"])
    return x