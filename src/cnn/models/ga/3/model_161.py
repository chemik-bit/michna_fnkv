from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(29, (6, 6), activation='sigmoid', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    x.add(layers.Conv2D(38, (1, 1), activation='relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1595, activation='leaky_relu'))
    x.add(layers.Dropout(0.9))
    x.add(layers.Dense(964, activation='leaky_relu'))
    x.add(layers.Dropout(0.7))
    x.add(layers.Dense(1032, activation='relu'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x