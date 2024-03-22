from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(7, (1, 1), activation='tanh', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.Conv2D(1, (4, 4), activation='relu', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1253, activation='leaky_relu'))
    x.add(layers.Dropout(0.2))
    x.add(layers.Dense(641, activation='tanh'))
    x.add(layers.Dropout(0.9))
    x.add(layers.Dense(1001, activation='tanh'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x