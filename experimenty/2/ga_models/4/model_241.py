from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(2, (4, 4), activation='leaky_relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1076, activation='leaky_relu'))
    x.add(layers.Dropout(0.4))
    x.add(layers.Dense(354, activation='leaky_relu'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x