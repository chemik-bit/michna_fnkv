from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(61, (2, 2), activation='relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(15, (3, 3), activation='leaky_relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(174, activation='relu'))
    x.add(layers.Dropout(0.9))
    x.add(layers.Dense(1205, activation='tanh'))
    x.add(layers.Dense(1923, activation='tanh'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x