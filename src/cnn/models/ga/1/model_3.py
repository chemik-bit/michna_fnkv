from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(37, (2, 2), activation='tanh', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.Conv2D(23, (1, 1), activation='sigmoid', padding='same'))
    x.add(layers.Conv2D(22, (1, 1), activation='sigmoid', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(7, 7), strides=(2, 2)))
    x.add(layers.Conv2D(40, (6, 6), activation='sigmoid', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1126, activation='leaky_relu'))
    x.add(layers.Dense(751, activation='relu'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x