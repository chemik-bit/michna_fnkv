from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(3, (1, 1), activation='sigmoid', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1580, activation='relu'))
    x.add(layers.Dense(353, activation='relu'))
    x.add(layers.Dropout(0.6))
    x.add(layers.Dense(1059, activation='leaky_relu'))
    x.add(layers.Dropout(0.7))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x