from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(52, (7, 7), activation='relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.Conv2D(48, (7, 7), activation='relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(36, (4, 4), activation='leaky_relu', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(28, (4, 4), activation='leaky_relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(9, (1, 1), activation='relu', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    x.add(layers.Conv2D(46, (6, 6), activation='tanh', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(16, (2, 2), activation='tanh', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1508, activation='tanh'))
    x.add(layers.Dropout(0.7))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x