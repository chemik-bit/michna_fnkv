from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(15, (5, 5), activation='sigmoid', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(2, (5, 5), activation='sigmoid', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(45, (4, 4), activation='tanh', padding='same'))
    x.add(layers.Conv2D(48, (2, 2), activation='leaky_relu', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(33, activation='leaky_relu'))
    x.add(layers.Dense(1162, activation='tanh'))
    x.add(layers.Dropout(0.5))
    x.add(layers.Dense(1725, activation='relu'))
    x.add(layers.Dropout(0.3))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x