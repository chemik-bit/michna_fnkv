from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(33, (7, 7), activation='relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(48, (7, 7), activation='leaky_relu', padding='same'))
    x.add(layers.Conv2D(20, (2, 2), activation='sigmoid', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(45, (2, 2), activation='relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(42, (2, 2), activation='leaky_relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(37, (6, 6), activation='tanh', padding='same'))
    x.add(layers.Conv2D(44, (4, 4), activation='leaky_relu', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(80, activation='tanh'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x