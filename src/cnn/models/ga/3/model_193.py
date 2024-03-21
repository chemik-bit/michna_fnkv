from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(17, (2, 2), activation='sigmoid', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    x.add(layers.Conv2D(24, (7, 7), activation='leaky_relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(49, (3, 3), activation='relu', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(41, (7, 7), activation='tanh', padding='same'))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(438, activation='relu'))
    x.add(layers.Dense(311, activation='tanh'))
    x.add(layers.Dropout(0.2))
    x.add(layers.Dense(1285, activation='sigmoid'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x