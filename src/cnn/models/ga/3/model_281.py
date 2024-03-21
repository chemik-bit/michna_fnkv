from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(28, (1, 1), activation='relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.Conv2D(63, (2, 2), activation='tanh', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(27, (5, 5), activation='tanh', padding='same'))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1513, activation='sigmoid'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x