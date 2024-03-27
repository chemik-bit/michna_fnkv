from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(21, (6, 6), activation='sigmoid', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(639, activation='sigmoid'))
    x.add(layers.Dense(1307, activation='sigmoid'))
    x.add(layers.Dense(1151, activation='sigmoid'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x