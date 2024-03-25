from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(6, (1, 1), activation='leaky_relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.Conv2D(54, (4, 4), activation='tanh', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(53, (2, 2), activation='relu', padding='same'))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1424, activation='relu'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x