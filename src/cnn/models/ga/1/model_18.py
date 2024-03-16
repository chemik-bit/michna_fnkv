from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(28, (7, 7), activation='relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1645, activation='sigmoid'))
    x.add(layers.Dropout(0.7))
    x.add(layers.Dense(1971, activation='relu'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x