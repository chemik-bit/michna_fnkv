from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(20, (4, 4), activation='relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1645, activation='leaky_relu'))
    x.add(layers.Dropout(0.5))
    x.add(layers.Dense(1824, activation='leaky_relu'))
    x.add(layers.Dropout(0.9))
    x.add(layers.Dense(1100, activation='leaky_relu'))
    x.add(layers.Dropout(0.3))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x