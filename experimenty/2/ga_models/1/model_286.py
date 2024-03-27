from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(24, (5, 5), activation='tanh', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1033, activation='relu'))
    x.add(layers.Dropout(0.7))
    x.add(layers.Dense(1532, activation='relu'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x