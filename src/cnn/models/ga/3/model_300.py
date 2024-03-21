from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(32, (2, 2), activation='sigmoid', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(59, (6, 6), activation='tanh', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(31, (7, 7), activation='sigmoid', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.Conv2D(60, (2, 2), activation='tanh', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1269, activation='leaky_relu'))
    x.add(layers.Dropout(0.9))
    x.add(layers.Dense(345, activation='sigmoid'))
    x.add(layers.Dense(247, activation='relu'))
    x.add(layers.Dropout(0.8))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x