from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(62, (6, 6), activation='tanh', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(541, activation='relu'))
    x.add(layers.Dropout(0.2))
    x.add(layers.Dense(1708, activation='tanh'))
    x.add(layers.Dropout(0.8))
    x.add(layers.Dense(1292, activation='tanh'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x