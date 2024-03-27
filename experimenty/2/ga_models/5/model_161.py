from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(51, (7, 7), activation='leaky_relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(406, activation='tanh'))
    x.add(layers.Dropout(0.7))
    x.add(layers.Dense(1462, activation='leaky_relu'))
    x.add(layers.Dropout(0.9))
    x.add(layers.Dense(43, activation='tanh'))
    x.add(layers.Dropout(0.4))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x