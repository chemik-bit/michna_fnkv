from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(55, (1, 1), activation='tanh', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(46, (1, 1), activation='relu', padding='same'))
    x.add(layers.Conv2D(45, (2, 2), activation='relu', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1783, activation='tanh'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x