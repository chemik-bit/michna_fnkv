from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(27, (7, 7), activation='tanh', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.Conv2D(54, (5, 5), activation='sigmoid', padding='same'))
    x.add(layers.Conv2D(62, (1, 1), activation='tanh', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.Conv2D(61, (2, 2), activation='tanh', padding='same'))
    x.add(layers.Conv2D(57, (7, 7), activation='sigmoid', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.Conv2D(32, (3, 3), activation='sigmoid', padding='same'))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1709, activation='tanh'))
    x.add(layers.Dense(1853, activation='relu'))
    x.add(layers.Dense(259, activation='sigmoid'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x