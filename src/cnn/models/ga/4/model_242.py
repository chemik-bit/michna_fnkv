from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(3, (2, 2), activation='sigmoid', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.Conv2D(23, (7, 7), activation='sigmoid', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(21, activation='leaky_relu'))
    x.add(layers.Dropout(0.6))
    x.add(layers.Dense(1715, activation='sigmoid'))
    x.add(layers.Dropout(0.7))
    x.add(layers.Dense(269, activation='sigmoid'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x