from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(11, (1, 1), activation='sigmoid', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(53, (6, 6), activation='sigmoid', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(24, (6, 6), activation='leaky_relu', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.Conv2D(1, (2, 2), activation='leaky_relu', padding='same'))
    x.add(layers.Conv2D(31, (5, 5), activation='relu', padding='same'))
    x.add(layers.Conv2D(33, (3, 3), activation='relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(500, activation='leaky_relu'))
    x.add(layers.Dropout(0.4))
    x.add(layers.Dense(622, activation='sigmoid'))
    x.add(layers.Dropout(0.9))
    x.add(layers.Dense(1161, activation='leaky_relu'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x