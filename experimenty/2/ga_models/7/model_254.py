from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(33, (6, 6), activation='tanh', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1169, activation='leaky_relu'))
    x.add(layers.Dropout(0.3))
    x.add(layers.Dense(1276, activation='tanh'))
    x.add(layers.Dense(1953, activation='tanh'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x