from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(8, (5, 5), activation='sigmoid', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(49, (5, 5), activation='relu', padding='same'))
    x.add(layers.Conv2D(11, (1, 1), activation='leaky_relu', padding='same'))
    x.add(layers.Conv2D(1, (3, 3), activation='leaky_relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1637, activation='tanh'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x