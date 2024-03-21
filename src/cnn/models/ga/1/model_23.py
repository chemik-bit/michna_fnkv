from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(55, (2, 2), activation='relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(14, (4, 4), activation='leaky_relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.Conv2D(46, (2, 2), activation='sigmoid', padding='same'))
    x.add(layers.Conv2D(33, (5, 5), activation='leaky_relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(43, (6, 6), activation='sigmoid', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.Conv2D(5, (2, 2), activation='relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(41, activation='relu'))
    x.add(layers.Dense(1734, activation='tanh'))
    x.add(layers.Dropout(0.4))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x