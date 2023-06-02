from tensorflow.keras import layers
from tensorflow.keras.backend import epsilon
from tensorflow.math import reduce_sum, square, maximum, sqrt
from tensorflow.keras.applications.vgg16 import VGG16
from keras.layers import Conv2D, LeakyReLU
import tensorflow as tf
from tensorflow import keras

def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(100, (3, 3), activation=LeakyReLU(), padding="same", input_shape = (input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1)))
    x.add(layers.Conv2D(100, (3, 3), activation=LeakyReLU(), padding="same"))
    x.add(layers.BatchNormalization())
    x.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    x.add(layers.Conv2D(100, (3, 3), activation=LeakyReLU(), padding="same"))
    x.add(layers.BatchNormalization())
    x.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1)))
    x.add(layers.Dropout(0.6))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1, activation="sigmoid"))
    x.compile(loss=tf.losses.binary_crossentropy, optimizer="adam", metrics=["val_accuracy"])
    return x