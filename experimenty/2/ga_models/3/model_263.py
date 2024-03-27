from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(8, (4, 4), activation='relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.Conv2D(16, (2, 2), activation='sigmoid', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.Conv2D(38, (3, 3), activation='sigmoid', padding='same'))
    x.add(layers.Conv2D(32, (4, 4), activation='relu', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1991, activation='tanh'))
    x.add(layers.Dense(1009, activation='tanh'))
    x.add(layers.Dropout(0.3))
    x.add(layers.Dense(1245, activation='sigmoid'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x