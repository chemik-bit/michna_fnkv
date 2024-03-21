from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(24, (2, 2), activation='sigmoid', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1436, activation='tanh'))
    x.add(layers.Dropout(0.8))
    x.add(layers.Dense(689, activation='tanh'))
    x.add(layers.Dropout(0.4))
    x.add(layers.Dense(593, activation='tanh'))
    x.add(layers.Dropout(0.9))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x