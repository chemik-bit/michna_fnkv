from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(24, (4, 4), activation='leaky_relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.Conv2D(33, (3, 3), activation='sigmoid', padding='same'))
    x.add(layers.Conv2D(10, (6, 6), activation='leaky_relu', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.Conv2D(39, (1, 1), activation='sigmoid', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(55, (4, 4), activation='relu', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(867, activation='relu'))
    x.add(layers.Dropout(0.7))
    x.add(layers.Dense(2007, activation='relu'))
    x.add(layers.Dropout(0.9))
    x.add(layers.Dense(1817, activation='tanh'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x