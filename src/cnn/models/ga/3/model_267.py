from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(53, (2, 2), activation='leaky_relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(41, (2, 2), activation='tanh', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(14, (6, 6), activation='leaky_relu', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(14, (7, 7), activation='relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(63, (7, 7), activation='sigmoid', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(35, (7, 7), activation='relu', padding='same'))
    x.add(layers.Conv2D(32, (1, 1), activation='sigmoid', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(507, activation='relu'))
    x.add(layers.Dropout(0.2))
    x.add(layers.Dense(173, activation='leaky_relu'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x