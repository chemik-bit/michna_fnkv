from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(49, (6, 6), activation='sigmoid', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1663, activation='leaky_relu'))
    x.add(layers.Dense(1804, activation='relu'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x