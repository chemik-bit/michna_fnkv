from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(9, (1, 1), activation='tanh', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.Conv2D(43, (2, 2), activation='sigmoid', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.Conv2D(19, (4, 4), activation='relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(494, activation='sigmoid'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x