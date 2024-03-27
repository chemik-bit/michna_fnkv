from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(18, (7, 7), activation='tanh', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1920, activation='tanh'))
    x.add(layers.Dropout(0.4))
    x.add(layers.Dense(919, activation='tanh'))
    x.add(layers.Dropout(0.8))
    x.add(layers.Dense(738, activation='sigmoid'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x