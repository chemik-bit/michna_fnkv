from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(12, (5, 5), activation='sigmoid', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.Conv2D(42, (6, 6), activation='tanh', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1709, activation='sigmoid'))
    x.add(layers.Dropout(0.7))
    x.add(layers.Dense(1196, activation='tanh'))
    x.add(layers.Dropout(0.3))
    x.add(layers.Dense(1667, activation='sigmoid'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x