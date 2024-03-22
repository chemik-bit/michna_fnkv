from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(1, (6, 6), activation='tanh', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.Conv2D(37, (1, 1), activation='sigmoid', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(3, (2, 2), activation='relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(58, (6, 6), activation='tanh', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(36, (7, 7), activation='tanh', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1369, activation='sigmoid'))
    x.add(layers.Dropout(0.6))
    x.add(layers.Dense(1057, activation='sigmoid'))
    x.add(layers.Dense(73, activation='tanh'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x