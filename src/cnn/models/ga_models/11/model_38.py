from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(39, (1, 1), activation='leaky_relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(73, activation='sigmoid'))
    x.add(layers.Dense(1170, activation='relu'))
    x.add(layers.Dense(1273, activation='tanh'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x