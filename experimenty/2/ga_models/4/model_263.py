from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(4, (5, 5), activation='sigmoid', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.Conv2D(17, (6, 6), activation='leaky_relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(13, (1, 1), activation='sigmoid', padding='same'))
    x.add(layers.Conv2D(18, (6, 6), activation='tanh', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(875, activation='sigmoid'))
    x.add(layers.Dense(1395, activation='leaky_relu'))
    x.add(layers.Dropout(0.6))
    x.add(layers.Dense(682, activation='leaky_relu'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x