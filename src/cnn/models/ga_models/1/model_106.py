from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(26, (6, 6), activation='leaky_relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.Conv2D(24, (3, 3), activation='tanh', padding='same'))
    x.add(layers.Conv2D(35, (6, 6), activation='sigmoid', padding='same'))
    x.add(layers.Conv2D(13, (5, 5), activation='relu', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(229, activation='leaky_relu'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x