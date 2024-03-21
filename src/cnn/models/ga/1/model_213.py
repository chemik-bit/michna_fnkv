from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(52, (1, 1), activation='leaky_relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(37, (5, 5), activation='leaky_relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(35, (5, 5), activation='sigmoid', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(45, (1, 1), activation='leaky_relu', padding='same'))
    x.add(layers.Conv2D(52, (7, 7), activation='tanh', padding='same'))
    x.add(layers.Conv2D(42, (7, 7), activation='sigmoid', padding='same'))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(128, activation='sigmoid'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x