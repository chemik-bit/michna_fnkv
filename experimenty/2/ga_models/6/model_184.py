from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(59, (1, 1), activation='relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    x.add(layers.Conv2D(36, (2, 2), activation='tanh', padding='same'))
    x.add(layers.Conv2D(11, (4, 4), activation='relu', padding='same'))
    x.add(layers.Conv2D(61, (3, 3), activation='relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1594, activation='sigmoid'))
    x.add(layers.Dense(35, activation='relu'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x