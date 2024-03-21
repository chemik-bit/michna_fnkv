from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(6, (7, 7), activation='leaky_relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(3, (7, 7), activation='relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    x.add(layers.Conv2D(3, (6, 6), activation='tanh', padding='same'))
    x.add(layers.Conv2D(22, (6, 6), activation='sigmoid', padding='same'))
    x.add(layers.Conv2D(28, (5, 5), activation='sigmoid', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(38, (1, 1), activation='tanh', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.Conv2D(25, (4, 4), activation='leaky_relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1558, activation='relu'))
    x.add(layers.Dropout(0.7))
    x.add(layers.Dense(85, activation='leaky_relu'))
    x.add(layers.Dropout(0.5))
    x.add(layers.Dense(1598, activation='sigmoid'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x