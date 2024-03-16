from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(28, (2, 2), activation='tanh', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1213, activation='sigmoid'))
    x.add(layers.Dropout(0.9))
    x.add(layers.Dense(1502, activation='tanh'))
    x.add(layers.Dropout(0.7))
    x.add(layers.Dense(731, activation='leaky_relu'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x