from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(2, (1, 1), activation='leaky_relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(60, (2, 2), activation='tanh', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(6, (5, 5), activation='relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(25, (7, 7), activation='sigmoid', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(4, (7, 7), activation='sigmoid', padding='same'))
    x.add(layers.Conv2D(33, (1, 1), activation='sigmoid', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(61, (1, 1), activation='relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(310, activation='leaky_relu'))
    x.add(layers.Dropout(0.9))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x