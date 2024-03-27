from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(2, (7, 7), activation='relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.Conv2D(37, (1, 1), activation='tanh', padding='same'))
    x.add(layers.Conv2D(7, (6, 6), activation='sigmoid', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    x.add(layers.Conv2D(6, (6, 6), activation='leaky_relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(10, (4, 4), activation='sigmoid', padding='same'))
    x.add(layers.Conv2D(31, (3, 3), activation='relu', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1823, activation='tanh'))
    x.add(layers.Dropout(0.2))
    x.add(layers.Dense(1216, activation='tanh'))
    x.add(layers.Dropout(0.4))
    x.add(layers.Dense(1523, activation='leaky_relu'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x