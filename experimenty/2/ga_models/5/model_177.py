from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(49, (1, 1), activation='relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(755, activation='relu'))
    x.add(layers.Dense(1280, activation='sigmoid'))
    x.add(layers.Dense(845, activation='leaky_relu'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x