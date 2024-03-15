from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(19, (3, 3), activation='leaky_relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(215, activation='tanh'))
    x.add(layers.Dropout(0.2))
    x.add(layers.Dense(1951, activation='sigmoid'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x