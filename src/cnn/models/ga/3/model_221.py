from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(10, (4, 4), activation='tanh', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.Conv2D(4, (2, 2), activation='leaky_relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.Conv2D(41, (1, 1), activation='leaky_relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(31, (1, 1), activation='leaky_relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.Conv2D(37, (7, 7), activation='tanh', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(10, activation='leaky_relu'))
    x.add(layers.Dropout(0.7))
    x.add(layers.Dense(1829, activation='sigmoid'))
    x.add(layers.Dropout(0.2))
    x.add(layers.Dense(1714, activation='sigmoid'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x