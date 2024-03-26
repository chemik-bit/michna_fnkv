from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(28, (6, 6), activation='sigmoid', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(6, (5, 5), activation='leaky_relu', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    x.add(layers.Conv2D(63, (7, 7), activation='sigmoid', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.Conv2D(16, (7, 7), activation='tanh', padding='same'))
    x.add(layers.Conv2D(39, (6, 6), activation='tanh', padding='same'))
    x.add(layers.Conv2D(60, (1, 1), activation='tanh', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(4, (1, 1), activation='leaky_relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1076, activation='sigmoid'))
    x.add(layers.Dropout(0.2))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x