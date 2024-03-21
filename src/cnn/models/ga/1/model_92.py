from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(7, (2, 2), activation='sigmoid', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(33, (3, 3), activation='tanh', padding='same'))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1121, activation='relu'))
    x.add(layers.Dense(1363, activation='leaky_relu'))
    x.add(layers.Dropout(0.9))
    x.add(layers.Dense(1563, activation='relu'))
    x.add(layers.Dropout(0.8))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x