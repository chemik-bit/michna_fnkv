from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(7, (3, 3), activation='sigmoid', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(31, (6, 6), activation='sigmoid', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(24, (4, 4), activation='relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(57, (5, 5), activation='sigmoid', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(6, (5, 5), activation='tanh', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1711, activation='tanh'))
    x.add(layers.Dropout(0.4))
    x.add(layers.Dense(785, activation='relu'))
    x.add(layers.Dropout(0.6))
    x.add(layers.Dense(533, activation='sigmoid'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x