from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(16, (1, 1), activation='sigmoid', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(39, (6, 6), activation='relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(32, (2, 2), activation='relu', padding='same'))
    x.add(layers.Conv2D(1, (7, 7), activation='sigmoid', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    x.add(layers.Conv2D(24, (6, 6), activation='sigmoid', padding='same'))
    x.add(layers.Conv2D(9, (7, 7), activation='tanh', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(22, (1, 1), activation='tanh', padding='same'))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1449, activation='tanh'))
    x.add(layers.Dropout(0.5))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x