from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(46, (7, 7), activation='tanh', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.Conv2D(59, (2, 2), activation='tanh', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(7, 7), strides=(2, 2)))
    x.add(layers.Conv2D(26, (7, 7), activation='sigmoid', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(4, 4), strides=(2, 2)))
    x.add(layers.Conv2D(23, (6, 6), activation='sigmoid', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(7, 7), strides=(2, 2)))
    x.add(layers.Conv2D(63, (3, 3), activation='tanh', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(4, 4), strides=(2, 2)))
    x.add(layers.Conv2D(27, (3, 3), activation='sigmoid', padding='same'))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(999, activation='leaky_relu'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x